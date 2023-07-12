from glob import glob
import pandas as pd
import numpy as np
import cv2
import os
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt
import shutil
import seaborn as sns
from sklearn.metrics import *
import json
import requests
import sys
sys.path.append("/home/VirtualFlaw/RT_Project")
from config import *
import time



def load_test(data_path, target_size):
    test_image = glob(data_path + "/**/*.jpg")
    test_image_label = [int(i.split("/")[-2]) for i in test_image]
    test_image_list = []
    for i in test_image:
        img = cv2.imread(i, (cv2.IMREAD_GRAYSCALE if target_size[2] == 1 else cv2.IMREAD_COLOR))
        img = cv2.resize(img, (target_size[0], target_size[1]))
        test_image_list.append(img)
    test_image_list = np.array(test_image_list)
    test_image_list = test_image_list.reshape(-1, target_size[0], target_size[1], target_size[2])
    return test_image_list, test_image_label, test_image

def load_train(data_path, data_type, target_size, batch_size):
    path = os.path.join(data_path, data_type)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(path, "Train"),
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(target_size[0], target_size[1]),
        batch_size=batch_size,
        label_mode="binary",
        color_mode=("grayscale" if target_size[2] == 1 else "rgb"),
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(path, "Train"),
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(target_size[0], target_size[1]),
        batch_size=batch_size,
        label_mode="binary",
        color_mode=("grayscale" if target_size[2] == 1 else "rgb"),
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds

def init_train(**kwargs):
    tf.random.set_seed(42)
    os.environ["PYTHONHASHSEED"] = str(42)
    date = kwargs["date"]
    data = kwargs["data"]
    model_type = kwargs["model_type"]
    target_size = kwargs["target_size"]
    batch_size = kwargs["batch_size"]
    learning_rate = kwargs["learning_rate"]
    loss_func = kwargs["loss_func"]
    if loss_func == "binary_focal_crossentropy":
        loss_func = "focal_loss"
    class_weight = kwargs["class_weight"]
    class_weight_ = str(class_weight).replace("{", "(").replace("}", ")")
    os.makedirs(f"/home/VirtualFlaw/RT_Project/log/{date}/{data}/{model_type}/{target_size}_{batch_size}_{learning_rate}_{loss_func}_{class_weight_}/models", exist_ok=True)
    model_dir = f"/home/VirtualFlaw/RT_Project/log/{date}/{data}/{model_type}/{target_size}_{batch_size}_{learning_rate}_{loss_func}_{class_weight_}/models"
    os.makedirs(f"/home/VirtualFlaw/RT_Project/log/{date}/{data}/{model_type}/{target_size}_{batch_size}_{learning_rate}_{loss_func}_{class_weight_}/logs", exist_ok=True)
    log_dir = f"/home/VirtualFlaw/RT_Project/log/{date}/{data}/{model_type}/{target_size}_{batch_size}_{learning_rate}_{loss_func}_{class_weight_}/logs"
    os.makedirs(f"/home/VirtualFlaw/RT_Project/log/{date}/{data}/{model_type}/{target_size}_{batch_size}_{learning_rate}_{loss_func}_{class_weight_}/result", exist_ok=True)
    result_dir = f"/home/VirtualFlaw/RT_Project/log/{date}/{data}/{model_type}/{target_size}_{batch_size}_{learning_rate}_{loss_func}_{class_weight_}/result"
    return model_dir, log_dir, result_dir

def draw_learning_curve(history, result_dir, DATA_PATH, DATE):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["binary_accuracy"], label="accuracy")
    plt.plot(history.history["val_binary_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "learning_curve.png"), dpi=500)
    plt.close()
    shutil.copy(DATA_PATH + "/Dataset_Info/Dataset_Info.txt", result_dir)
    os.rename(result_dir +"/Dataset_Info.txt", result_dir +"/Dataset_Info_" + DATE + ".txt")
    #learningrate 그래프
    plt.figure(figsize=(12, 4))
    plt.plot(history.history["lr"], label="lr")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "learning_rate.png"), dpi=500)
    
def get_callbacks(model_type, model_dir, log_dir):
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.999,
        patience=16,
        verbose=0,
        min_delta=1e-4,
        min_lr=1e-7,
        mode="auto",
    )

    checkpoint = ModelCheckpoint(
        f"{model_dir}/{model_type}.h5",
        monitor="val_loss",
        verbose=2,
        save_best_only=True,
        mode="auto",

    )

    early_stopping = EarlyStopping(
        monitor="val_loss", verbose=0, mode="auto",
        patience=48,
    )

    csv_logger = CSVLogger(
        f"{log_dir}/{model_type}.csv",
        append=True,
    )
    
    
    return [reduce_lr, checkpoint, early_stopping, csv_logger]

def generate_report(test_pred, test_label, test_image, result_dir, upload=False):
    start_time = time.time()
    print("Generating report...")
    df = pd.DataFrame(test_pred, columns=["pred"])
    df["label"] = test_label
    df.to_csv(os.path.join(result_dir, "label_pred.csv"), index=False)
    Threshold_list = []
    Precision_list = []
    Recall_list = []
    F1_score_list = []
    Accuracy_list = []
    for i in range(0, 10000):
        Threshold = i/10000
        Threshold_list.append(Threshold)
        test_pred_clip = np.where(test_pred > Threshold, 1, 0)
        report = classification_report(test_label, test_pred_clip, labels=[0, 1], target_names=["class 0", "class 1"], digits=4, zero_division=0, output_dict=True)
        F1_score_list.append(report["class 1"]["f1-score"])
        Recall_list.append(report["class 1"]["recall"])
        Precision_list.append(report["class 1"]["precision"])
        Accuracy_list.append((report["class 1"]["precision"] * 400 + report["class 1"]["recall"] * 45) / (400 + 45))
    df_origin = pd.DataFrame({"Threshold": Threshold_list, "Precision": Precision_list, "Recall": Recall_list, "F1_score": F1_score_list, "Accuracy": Accuracy_list})
    #f1 score가 가장 높은 행의 값들을 리턴
    df = df_origin[df_origin["F1_score"] == df_origin["F1_score"].max()]

    threshold, precision, recall, f1, accuracy = df.iloc[-1]
    draw_roc_curve(test_label, test_pred, result_dir, threshold)
    draw_precision_recall_curve(test_label, test_pred, result_dir, threshold)
    test_pred = np.where(test_pred > threshold, 1, 0)
    draw_confusion_matrix(test_label, test_pred, result_dir, threshold)
    write_classification_report(test_label, test_pred, result_dir)
    #csv 파일 경로
    path = result_dir + "/label_pred.csv"
    upload_notion(threshold, precision, recall, f1, accuracy, path, NOTION_DATABASE_ID_FS) if upload else None
    
    shutil.rmtree(f"{result_dir}/False", ignore_errors=True)
    os.makedirs(f"{result_dir}/False", exist_ok=True)
    for pred, label, i in zip(test_pred, test_label, test_image):
        if pred == label:
            pass
        else:
            shutil.copy(i, f"{result_dir}/False")
    
    #recall이 1인 행의 값들을 리턴
    df = df_origin[df_origin["Recall"] == df_origin["Recall"].max()]
    threshold, precision, recall, f1, accuracy = df.iloc[-1]
    end_time = time.time()
    print(f"Report generated in {end_time - start_time:.2f} seconds.")
    upload_notion(threshold, precision, recall, f1, accuracy, path, NOTION_DATABASE_ID_RC) if upload else None    
    
   
def upload_notion(threshold, precision, recall, f1, accuracy, path, notion_database_id):
    model_name = path.split("/")[-4]
    info = path.split("/")[-3]
    Dataset = path.split("/")[-5]
    Input_Size = info.split("_")[0].replace(",","")
    Batch_Size = info.split("_")[1]
    Learning_Rate = info.split("_")[2]
    Loss_Function = info.split("_")[3] + "_" + info.split("_")[4]
    if Loss_Function == "binary_focal":
        Loss_Function = "binary_focal_" + info.split("_")[5]
        Class_Weight = "None" if len(info.split("_")) == 6 else info.split("_")[6]
    else:
        Class_Weight = "None" if len(info.split("_")) == 5 else info.split("_")[5]
    DATE = path.split("/")[-6]
    DATE = DATE[:4] + "-" + DATE[4:6] + "-" + DATE[6:8] + "T" + DATE[8:10] + ":" + DATE[10:12] + ":00.000+09:00"
    page_values = {
        "Model": model_name,
        "Dataset": Dataset,
        "Input Size": Input_Size,
        "Batch Size": Batch_Size,
        "Learning Rate": float(Learning_Rate),
        "Loss Func": Loss_Function,
        "Class Weight": Class_Weight,
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Accuracy": accuracy,
        "상태": "Done",
        "실행 일시": DATE
    }
    upload_test_result(page_values, notion_database_id)
    
def upload_test_result(page_values, notion_database_id):
    createdUrl = "https://api.notion.com/v1/pages"
    newPageData = {
        "parent": {"database_id": notion_database_id},
        "properties": {
                "Loss Func": {
                    "type": "select",
                    "select": {
                        "name": page_values["Loss Func"],
                    }
                },
                "상태": {
                    "type": "status",
                    "status": {
                        "name": page_values["상태"],
                    }
                },
                "Recall": {
                    "type": "number",
                    "number": round(page_values["Recall"], 4)
                },
                "Dataset": {
                    "type": "select",
                    "select": {
                        "name": page_values["Dataset"],
                    }
                },
                "Learning Rate": {
                    "type": "number",
                    "number": page_values["Learning Rate"]
                },
                "Accuracy": {
                    "type": "number",
                    "number": round(page_values["Accuracy"], 4)
                },
                "Input Size": {
                    "type": "select",
                    "select": {
                        "name": page_values["Input Size"],
                    }
                },
                "Precision": {
                    "type": "number",
                    "number": round(page_values["Precision"], 4)
                },
                "Batch Size": {
                    "type": "select",
                    "select": {
                        "name": page_values["Batch Size"],
                    }
                },
                "F1 Score": {
                    "type": "number",
                    "number": round(page_values["F1 Score"], 4)
                },
                "실행 일시": {
                    "type": "date",
                    "date": {
                        "start": page_values["실행 일시"],
                        "end": None,
                        "time_zone": None
                    }
                },
                "Model": {
                    "type": "title",
                    "title": [
                        {
                            "type": "text",
                            "text": {
                                "content": page_values["Model"],
                                "link": None
                            },
                            "annotations": {
                                "bold": False,
                                "italic": False,
                                "strikethrough": False,
                                "underline": False,
                                "code": False,
                                "color": "default"
                            },
                            "plain_text": page_values["Model"],
                            "href": None
                        }
                    ]
                },
                "Class Weight": {
                    "type": "rich_text",
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": page_values["Class Weight"],
                                "link": None
                            },
                            "annotations": {
                                "bold": False,
                                "italic": False,
                                "strikethrough": False,
                                "underline": False,
                                "code": False,
                                "color": "default"
                            },
                            "plain_text": page_values["Class Weight"],
                            "href": None
                        }
                    ]
                },
                "Threshold": {
                    "type": "number",
                    "number": page_values["Threshold"]
                },
                
            }
    }
    data = json.dumps(newPageData)
    res = requests.post(createdUrl, headers=HEADERS, data=data)
    if res.status_code == 200:
        block_url = res.json()["url"]
    else:
        print(res.status_code)
        print(res.text)
        
        
def draw_roc_curve(test_label, test_pred, result_dir, threshold):
    fpr, tpr, thresholds = roc_curve(test_label, test_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver operating characteristic - threshold : {threshold}')
    plt.legend(loc="lower right")
    plt.savefig(result_dir + f"/roc_curve.png", dpi=500)
    
def draw_confusion_matrix(test_label, test_pred, result_dir, threshold):
    cm = confusion_matrix(test_label, test_pred)
    plt.figure(figsize=(10, 10))
    plt.title(f'Confusion matrix - threshold : {threshold}')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues)
    plt.savefig(result_dir + f"/confusion_matrix.png", dpi=500)
    
def draw_precision_recall_curve(test_label, test_pred, result_dir, threshold):
    precision, recall, thresholds = precision_recall_curve(test_label, test_pred)
    plt.figure(figsize=(10, 10))
    plt.plot(recall, precision, label='Precision-Recall curve - ')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve - threshold : {threshold}')
    plt.legend(loc="lower right")
    plt.savefig(result_dir + f"/precision_recall_curve.png", dpi=500)
    
def draw_threshold_change_curves(test_pred, test_label, result_dir):
    Threshold_list = []
    Precision_list = []
    Recall_list = []
    F1_score_list = []
    Accuracy_list = []
    for i in range(0, 1000):
        Threshold = i/1000
        Threshold_list.append(Threshold)
        test_pred_clip = np.where(test_pred > Threshold, 1, 0)
        report = classification_report(test_label, test_pred_clip, labels=[0, 1], target_names=["class 0", "class 1"], digits=4, zero_division=0, output_dict=True)
        F1_score_list.append(report["class 1"]["f1-score"])
        Recall_list.append(report["class 1"]["recall"])
        Precision_list.append(report["class 1"]["precision"])
        Accuracy_list.append((report["class 1"]["precision"] * 400 + report["class 1"]["recall"] * 45) / (400 + 45))

    df = pd.DataFrame({"Threshold": Threshold_list, "Precision": Precision_list, "Recall": Recall_list, "F1_score": F1_score_list, "Accuracy": Accuracy_list})
    #f1 score가 가장 높은 행의 값들을 리턴
    df = df[df["F1_score"] == df["F1_score"].max()]
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(Threshold_list, Precision_list, label='Precision')
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.subplot(2, 2, 2)
    plt.plot(Threshold_list, Recall_list, label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Recall')
    plt.subplot(2, 2, 3)
    plt.plot(Threshold_list, F1_score_list, label='F1_score')
    plt.xlabel('Threshold')
    plt.ylabel('F1_score')
    plt.subplot(2, 2, 4)
    plt.plot(Threshold_list, Accuracy_list, label='Accuracy')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.suptitle('Threshold')
    plt.savefig(result_dir + f"/threshold.png", dpi=500)
    return df.iloc[0]

def write_classification_report(test_label, test_pred, result_dir):
    with open(result_dir + f"/classification_report.txt", "w") as f:
        f.write(classification_report(test_label, test_pred, target_names=['Accept', 'Reject']))