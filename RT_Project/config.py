import datetime

#모드
MODE = "train"
#

#날짜 설정
DATE = datetime.datetime.now().strftime("%Y%m%d%H%M")
#DATE = "202305130626"
#

#모델 설정
#MODEL = ["CNN", "VGG16", "ResNet50", "InceptionV3", "InceptionResNetV2", "MobileNetV2", "DenseNet121", "Xception", "EfficientNetB0", "NASNetMobile", "NASNetLarge"]
MODEL = ["CNN2",]
LOSS_FUNC = ["binary_crossentropy"]
LEARNING_RATE = [0.0005]
WEIGHT = [{0: 1, 1: 1}]
OPTIMIZER = ["Adam"]
WEIGHT_DECAY = [0.0001]
MOMENTUM = [0.9]
#

#데이터 설정
DATA = [  "VF_4000","VF_6000","VF_8000","VF_10000"]
TARGET_SIZE = [(512, 512, 3)]
BATCH_SIZE = [64, 128, 256]
EPOCHS = 300
#

#데이터 경로 설정
DATA_PATH = "/home/VirtualFlaw/RT_Project/data/Dataset_VF_20000"
TEST_PATH = "/home/VirtualFlaw/RT_Project/data/Test"
#


#Notion 설정
NOTION_DATABASE_ID_RC = "0c9c9940cae5420b9fca645154167c32"
NOTION_DATABASE_ID_FS = "4c0a155472254cbd9c2f1907cfa9d22b"
NOTION_KEY = ""
HEADERS = {
            "Authorization": "Bearer " + NOTION_KEY,
            "Content-Type": "application/json",
            "Notion-Version": "2022-02-22"
        }
#NOTION_URL = "https://api.notion.com/v1/databases/" + NOTION_DATABASE_ID