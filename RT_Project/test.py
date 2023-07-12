from modules.test_v3 import *
import sys
import shutil
import tensorflow as tf
########## Notion API ##########
NOTION_DATABASE_ID = ""
NOTION_KEY = ""
################################
if len(sys.argv) == 3:
    DATE = sys.argv[1]
    DATASET = sys.argv[2]
else:
    DATE = "202305080443"
    DATASET = "Dataset_DeformedVFBasedOnVanilla"

if __name__ == "__main__":
    tf.keras.backend.clear_session()
    TEST_PATH = "/home/VirtualFlaw/RT_Project/data/Test_V2"
    test_image_list, test_image_label, test_image = load_test(TEST_PATH)
    check = 0
    for DATE in os.listdir("/home/VirtualFlaw/RT_Project/log"):
        if DATE == "Legacy":
            continue
        if DATE == "202305090623":
            check = 1
        if check == 0:
            continue
        
        test = TestModel(NOTION_DATABASE_ID, NOTION_KEY, DATE, TEST_PATH, DATASET, test_image_list=test_image_list, test_image_label=test_image_label, test_image=test_image)
        test.test_model(verbose=False, upload=False)    
    
    
    
    