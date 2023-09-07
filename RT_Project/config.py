import datetime

#모드
MODE = "train"
#
MODEL_PATH = ""
#날짜 설정
DATE = datetime.datetime.now().strftime("%Y%m%d%H%M")
#DATE = "202305130626"
#

#모델 설정
#MODEL = ["CNN", "VGG16", "ResNet50", "InceptionV3", "InceptionResNetV2", "MobileNetV2", "DenseNet121", "Xception", "EfficientNetB0", "NASNetMobile", "NASNetLarge"]
MODEL = ["CNN5"]
LOSS_FUNC = ["binary_crossentropy"]
LEARNING_RATE = [0.0005]
WEIGHT = [{0: 1, 1: 1}]
OPTIMIZER = ["Adam"]
WEIGHT_DECAY = [0.00001]
MOMENTUM = [0.9]
#

#데이터 설정
DATA = ["PO"]
TARGET_SIZE = [(512, 512, 1)]
BATCH_SIZE = [128]
EPOCHS = 9999
#

#데이터 경로 설정
DATA_PATH = "/home/RT_Paper/data"

#


#로그 저장 폴더 설정
LOG_DIR = "/home/RT_Paper/log"
#


#Notion 설정
NOTION_DATABASE_ID_RC = "0c9c9940cae5420b9fca645154167c32"
NOTION_DATABASE_ID_FS = "1f959176342148768bff07bacec5c0b7"
NOTION_KEY = "secret_2nVjIaYGdiJJbz7VpKwF0kqsdbZyqgjgLHPQWfyEXzF"
HEADERS = {
            "Authorization": "Bearer " + NOTION_KEY,
            "Content-Type": "application/json",
            "Notion-Version": "2022-02-22"
        }
#NOTION_URL = "https://api.notion.com/v1/databases/" + NOTION_DATABASE_ID