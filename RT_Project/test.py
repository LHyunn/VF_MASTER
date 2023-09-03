from modules.test_v3 import *
import sys
import shutil
import tensorflow as tf
from config import *
########## Notion API ##########
NOTION_DATABASE_ID = NOTION_DATABASE_ID_FS
NOTION_KEY = "secret_2nVjIaYGdiJJbz7VpKwF0kqsdbZyqgjgLHPQWfyEXzF"
################################
if len(sys.argv) == 3:
    DATE = sys.argv[1]
    DATASET = sys.argv[2]
else:
    DATE = "202309030702"
    DATASET = "PO"

if __name__ == "__main__":
    tf.keras.backend.clear_session()
    test_image_list, test_image_label, test_image = load_test(TEST_PATH, TARGET_SIZE[0])
    test = TestModel(NOTION_DATABASE_ID, NOTION_KEY, DATE, TEST_PATH, DATASET, test_image_list=test_image_list, test_image_label=test_image_label, test_image=test_image)
    test.test_model(verbose=False, upload=True)    
    
    
    
    