import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
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

def load_test(path):
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

def sigmoid_2x(x):
    return tf.nn.sigmoid(2*x)




class TestModel:
    def __init__(self, NOTION_DATABASE_ID, NOTION_KEY, DATE, TEST_PATH, DATASET, test_image_list=None, test_image_label=None, test_image=None):
        
        self.NOTION_DATABASE_ID = NOTION_DATABASE_ID
        self.NOTION_KEY = NOTION_KEY
        self.NOTION_URL = "https://api.notion.com/v1/databases/" + NOTION_DATABASE_ID
        self.DATE = DATE
        self.DATE_ = DATE[:4] + "-" + DATE[4:6] + "-" + DATE[6:8] + "T" + DATE[8:10] + ":" + DATE[10:12] + ":00.000+09:00"
        self.HEADERS = {
            "Authorization": "Bearer " + NOTION_KEY,
            "Content-Type": "application/json",
            "Notion-Version": "2022-02-22"
        }
        self.TEST_PATH = TEST_PATH
        self.DATASET = DATASET
        ##########################################################
        print("Test initialized. - DATE:", self.DATE)
        if test_image_list is None:
            self.test_image_list, self.test_image_label, self.test_image = self.load_test()
        else:
            self.test_image_list = test_image_list
            self.test_image_label = test_image_label
            self.test_image = test_image
        self.model_list = glob(f"/home/VirtualFlaw/RT_Project/log/{self.DATE}/**/*.h5", recursive=True)
        print("Test loaded. total model count:", len(self.model_list), "total test image count:", len(self.test_image_list))
        self.pred = None
        
        
    def load_test(self):
        test_image = glob(self.TEST_PATH + "/**/*.jpg")
        test_image_label = [int(i.split("/")[-2]) for i in test_image]
        test_image_list = []
        for i in test_image:
            img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (512, 512))
            test_image_list.append(img)
        test_image_list = np.array(test_image_list)
        test_image_list = test_image_list.reshape(-1, 512, 512, 1)
        return test_image_list, test_image_label, test_image    
        
    def test_model(self, verbose=True, upload=True):
        for model in self.model_list:
            model_name = model.split("/")[-1].split(".")[0]
            result_path = "/".join(model.split("/")[:-2]) + "/result"
            info = model.split("/")[-3]
            Dataset = model.split("/")[-5]
            Input_Size = info.split("_")[0].replace(",","")
            Batch_Size = info.split("_")[1]
            Learning_Rate = info.split("_")[2]
            Loss_Function = info.split("_")[3] + "_" + info.split("_")[4]
            Class_Weight = "None" if len(info.split("_")) == 5 else info.split("_")[5]
            print("Testing model:", model_name, "Dataset:", Dataset, "Input_Size:", Input_Size, "Batch_Size:", Batch_Size, "Learning_Rate:", Learning_Rate, "Loss_Function:", Loss_Function, "Class_Weight:", Class_Weight) if verbose else None
            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                tf_model = tf.keras.models.load_model(model, custom_objects={"f1_m": f1_m, "precision_m": precision_m, "recall_m": recall_m, "focal_loss": tfa.losses.SigmoidFocalCrossEntropy(), "sigmoid_2x": sigmoid_2x})
            self.test_pred = tf_model.predict(self.test_image_list) # 1. predict
            df = pd.DataFrame(self.test_pred)
            df["label"] = self.test_image_label
            df.to_csv(result_path + "/" + "label_pred.csv", index=False)
            self.threshold, f1_score_, recall_, precision_, accuracy_= self.draw_threshold_change_curves(model_name, result_path) # 2. draw threshold change curves and get best threshold
            print("Threshold:", self.threshold) if verbose else None
            self.draw_roc_curve(self.test_image_label, self.test_pred, model_name, result_path) # 3. draw roc curve
            self.draw_precision_recall_curve(self.test_image_label, self.test_pred, model_name, result_path) # 4. draw precision recall curve
            self.test_pred = np.where(self.test_pred > self.threshold, 1, 0) # 5. apply threshold
            self.draw_confusion_matrix(self.test_image_label, self.test_pred, model_name, result_path) # 6. draw confusion matrix
            self.write_classification_report(self.test_image_label, self.test_pred, result_path) # 7. write classification report
            page_values = {
                "Loss Func": Loss_Function,
                "상태": "Done",
                "Recall": recall_,
                "Dataset": Dataset,
                "Learning Rate": float(Learning_Rate),
                "Accuracy": accuracy_,
                "Input Size": Input_Size,
                "Precision": precision_,
                "Batch Size": Batch_Size,
                "F1 Score": f1_score_,
                "Model": model_name,
                "실행 일시": self.DATE_,
                "Class Weight": Class_Weight,
                "Threshold": self.threshold
                
            }
            self.upload_test_result(page_values) if upload else None
            
            shutil.rmtree("/".join(model.split("/")[:-2]) + f"/False", ignore_errors=True)
            os.makedirs("/".join(model.split("/")[:-2]) + f"/False", exist_ok=True)
            
            for pred, label, i in zip(self.test_pred, self.test_image_label, self.test_image):
                if pred == label:
                    pass
                else:
                    shutil.copy(i, "/".join(model.split("/")[:-2]) + "/False")
            
            
            
            
            
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
        res = requests.post(createdUrl, headers=self.HEADERS, data=data)
        if res.status_code == 200:
            block_url = res.json()["url"]
        else:
            print(res.status_code)
            print(res.text)
            
    def draw_roc_curve(self, test_image_label, test_pred, model_name, result_path):
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
        plt.savefig(result_path + f"/roc_curve.png", dpi=500)
        
    def draw_confusion_matrix(self, test_image_label, test_pred, model_name, result_path):
        cm = confusion_matrix(test_image_label, test_pred)
        plt.figure(figsize=(10, 10))
        plt.title(f'Confusion matrix - threshold : {self.threshold}' + model_name)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues)
        plt.savefig(result_path + f"/confusion_matrix.png", dpi=500)
        
    def draw_precision_recall_curve(self, test_image_label, test_pred, model_name, result_path):
        precision, recall, thresholds = precision_recall_curve(test_image_label, test_pred)
        plt.figure(figsize=(10, 10))
        plt.plot(recall, precision, label='Precision-Recall curve - ')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall curve - threshold : {self.threshold}' + model_name)
        plt.legend(loc="lower right")
        plt.savefig(result_path + f"/precision_recall_curve.png", dpi=500)
        
    def draw_threshold_change_curves(self, model_name, result_path):
        Threshold_list = []
        Precision_list = []
        Recall_list = []
        F1_score_list = []
        Accuracy_list = []
        len_0 = self.test_image_label.count(0)
        len_1 = self.test_image_label.count(1)
        for i in range(0, 1000):
            Threshold = i/1000
            Threshold_list.append(Threshold)
            test_pred = np.where(self.test_pred > Threshold, 1, 0)
            report = classification_report(self.test_image_label, test_pred, labels=[0, 1], target_names=["class 0", "class 1"], digits=4, zero_division=0, output_dict=True)
            F1_score_list.append(report["class 1"]["f1-score"])
            Recall_list.append(report["class 1"]["recall"])
            Precision_list.append(report["class 1"]["precision"])
            Accuracy_list.append((report["class 1"]["precision"] * len_0 + report["class 1"]["recall"] * len_1) / (len_0 + len_1))


        #recall이 1인 것 중에서 precision이 가장 높은 것을 찾는다.
        df = pd.DataFrame({"Threshold": Threshold_list, "Precision": Precision_list, "Recall": Recall_list, "F1_score": F1_score_list, "Accuracy": Accuracy_list})
        df.to_csv(result_path + f"/threshold_recall_1.csv", index=False)
        try:
            df = df[df["Recall"] == 1]
            #마지막 행이 가장 높은 precision을 가지는 행이다.
            max_F1_score_Threshold, f1_score_, recall_, precision_, accuracy_ = df.iloc[-1]
        except:
            max_F1_score_Threshold, f1_score_, recall_, precision_, accuracy_ = -1, -1, -1, -1, -1
    
        
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
        plt.savefig(result_path + f"/threshold.png", dpi=500)
        return max_F1_score_Threshold, f1_score_, recall_, precision_, accuracy_
  
    def write_classification_report(self, test_image_label, test_pred, result_path):
        with open(result_path + f"/classification_report.txt", "w") as f:
            f.write(classification_report(test_image_label, test_pred, target_names=['Accept', 'Reject'], digits=4, zero_division=0))        
            
            
        
    

    