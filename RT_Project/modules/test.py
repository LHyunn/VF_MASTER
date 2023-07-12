import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *
#tfa focal loss
import tensorflow_addons as tfa
import shutil
import requests
import json

def load_test(self, path):
    test_image = glob(path + "/**/*.jpg")
    test_image_label = [int(i.split("/")[-2]) for i in test_image]
    test_image_list = []
    for i in test_image:
        img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (512, 512))
        test_image_list.append(img)
    test_image_list = np.array(test_image_list)
    test_image_list = test_image_list.reshape(-1, 512, 512, 1)
    return test_image_list, test_image_label, test_image

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))    

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


class Test:
    def __init__(self, NOTION_DATABASE_ID, NOTION_KEY, DATE, TEST_PATH, DATASET):
        print("Test initialized.")
        self.NOTION_DATABASE_ID = NOTION_DATABASE_ID
        self.NOTION_KEY = NOTION_KEY
        self.NOTION_URL = "https://api.notion.com/v1/databases/" + NOTION_DATABASE_ID
        self.DATE = DATE
        self.DATE_ = DATE[:4] + "-" + DATE[4:6] + "-" + DATE[6:8] + "T" + DATE[9:11] + ":" + DATE[11:13] + ":00.000+09:00"
        self.HEADERS = {
            "Authorization": "Bearer " + NOTION_KEY,
            "Content-Type": "application/json",
            "Notion-Version": "2022-02-22"
        }
        self.TEST_PATH = TEST_PATH
        self.DATASET = DATASET

    def upload_test_result(self, page_values):
        createdUrl = "https://api.notion.com/v1/pages"
        newPageData = {
            "parent": {"database_id": self.NOTION_DATABASE_ID},
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
                        "number": page_values["Recall"]
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
                        "number": page_values["Accuracy"]
                    },
                    "Input Size": {
                        "type": "select",
                        "select": {
                            "name": page_values["Input Size"],
                        }
                    },
                    "Precision": {
                        "type": "number",
                        "number": page_values["Precision"]
                    },
                    "Batch Size": {
                        "type": "select",
                        "select": {

                            "name": page_values["Batch Size"],
                        }
                    },
                    "F1 Score": {
                        "type": "number",
                        "number": page_values["F1 Score"]
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
        res = requests.post(createdUrl, headers=self.HEADERS, data=data)
        if res.status_code == 200:
            block_url = res.json()["url"]
        else:
            print(res.status_code)
            print(res.text)
            

    

    def draw_roc_curve(self, test_image_label, test_pred, path, model_name):
        fpr, tpr, thresholds = roc_curve(test_image_label, test_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(10, 10))
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver operating characteristic - threshold : {self.threshold}' + model_name)
        plt.legend(loc="lower right")
        plt.savefig(path + f"/roc_curve.png", dpi=500)
        
    def draw_confusion_matrix(self, test_image_label, test_pred, path, model_name):
        cm = confusion_matrix(test_image_label, test_pred)
        plt.figure(figsize=(10, 10))
        plt.title(f'Confusion matrix - threshold : {self.threshold}' + model_name)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues)
        plt.savefig(path + f"/confusion_matrix.png", dpi=500)
        
    def draw_precision_recall_curve(self, test_image_label, test_pred, path, model_name):
        precision, recall, thresholds = precision_recall_curve(test_image_label, test_pred)
        plt.figure(figsize=(10, 10))
        plt.plot(recall, precision, label='Precision-Recall curve - ')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall curve - threshold : {self.threshold}' + model_name)
        plt.legend(loc="lower right")
        plt.savefig(path + f"/precision_recall_curve.png", dpi=500)
        
    def draw_threshold_change_curves(self, test_image_label, test_pred_original, path, model_name):
        Threshold_list = []
        Precision_list = []
        Recall_list = []
        F1_score_list = []
        Accuracy_list = []
        for i in range(0, 1000):
            Threshold = i/1000
            Threshold_list.append(Threshold)
            test_pred = np.where(test_pred_original > Threshold, 1, 0)
            cm = confusion_matrix(test_image_label, test_pred)
            Precision = cm[1][1] / (cm[1][1] + cm[0][1])
            Recall = cm[1][1] / (cm[1][1] + cm[1][0])
            F1_score = 2 * (Precision * Recall) / (Precision + Recall)
            Accuracy = (cm[1][1] + cm[0][0]) / (cm[1][1] + cm[0][0] + cm[1][0] + cm[0][1])
            Precision_list.append(Precision)
            Recall_list.append(Recall)
            F1_score_list.append(F1_score)
            Accuracy_list.append(Accuracy)
        #가장 높은 F1_score를 가지는 Threshold를 찾는다.
        max_F1_score = max(F1_score_list)
        max_F1_score_index = F1_score_list.index(max_F1_score)
        max_F1_score_Threshold = Threshold_list[max_F1_score_index]
        
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
        plt.suptitle('Threshold - ' + model_name)
        plt.savefig(path + f"/threshold.png", dpi=500)
        return max_F1_score_Threshold

    def write_classification_report(self, test_image_label, test_pred, path, model_name):
        with open(path + f"/classification_report.txt", "w") as f:
            f.write(classification_report(test_image_label, test_pred, target_names=['Accept', 'Reject']))
            

            
    def model_test(self, upload = False):
        date = self.DATE
        path = self.TEST_PATH
        test_image_list, test_image_label, test_image = self.load_test(path)
        model_path = glob(f"/home/VirtualFlaw/RT_Project/log/{date}/**/*.h5", recursive=True)
        print("Total test count : ", len(model_path))
        
        for model_file in model_path:
            model_name = model_file.split("/")[-1].split(".")[0]
            class_weight = model_file.split("/")[-3].split("_")[-1]
            info = model_file.split("/")[-3]
            dataset = model_file.split("/")[-5]
            path = "/".join(model_file.split("/")[:-2]) + "/result"
            input_size = info.split("_")[0].replace(",", "")
            batch_size = info.split("_")[1]
            learning_rate = info.split("_")[2]
            loss = info.split("_")[3] + "_" + info.split("_")[4]
            model = None
            try:
                model = tf.keras.models.load_model(model_file)
            except:
                model = tf.keras.models.load_model(model_file, custom_objects={'Addons>SigmoidFocalCrossEntropy': tfa.losses.SigmoidFocalCrossEntropy(),'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m})
            test_pred = model.predict(test_image_list, verbose=2, batch_size=1)
            test_pred_original = test_pred.copy()
            
            
            
            
            shutil.rmtree("/".join(model_file.split("/")[:-2]) + f"/False", ignore_errors=True)
            os.makedirs("/".join(model_file.split("/")[:-2]) + f"/False", exist_ok=True)
            
            for pred, label, i in zip(test_pred, test_image_label, test_image):
                if pred == label:
                    pass
                else:
                    shutil.copy(i, "/".join(model_file.split("/")[:-2]) + "/False")
            self.threshold = self.draw_threshold_change_curves(test_image_label, test_pred_original, path, model_name)
            test_pred = np.where(test_pred > self.threshold, 1, 0)
            test_pred = test_pred.reshape(-1)
            self.draw_roc_curve(test_image_label, test_pred_original, path, model_name)
            self.draw_confusion_matrix(test_image_label, test_pred, path, model_name)
            self.draw_precision_recall_curve(test_image_label, test_pred_original, path, model_name)
            self.write_classification_report(test_image_label, test_pred, path, model_name)
            
            
            page_values = {
                "Loss Func": loss,
                "상태": "Done",
                "Recall": recall_score(test_image_label, test_pred).round(3),
                "Dataset": dataset,
                "Learning Rate": float(learning_rate),
                "Accuracy": accuracy_score(test_image_label, test_pred).round(3),
                "Input Size": input_size,
                "Precision": precision_score(test_image_label, test_pred).round(3),
                "Batch Size": batch_size,
                "F1 Score": f1_score(test_image_label, test_pred).round(3),
                "Model": model_name,
                "실행 일시": (date[:4] + "-" + date[4:6] + "-" + date[6:8] + "T" + date[8:10] + ":" + date[10:12] + ":00.000+09:00"),
                "Class Weight": class_weight,
                "Threshold": self.threshold,
                
            }

            self.upload_test_result(page_values) if upload else None
            
            
            