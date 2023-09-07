import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import random
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from skimage.io import imread, imshow
from skimage.transform import resize
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from models.unet_model import *

warnings.filterwarnings("ignore")
from config import *

def main(**kwargs):
    TRAIN_PATH = '/home/RT_Paper/data/IP/train/1/'
    TRAIN_MASK_PATH = '/home/RT_Paper/data/IP/Diff/'
    FLAW_TYPE = 'IP'

    train_ids = os.listdir(TRAIN_PATH)

    print(len(train_ids))
    n_images = 4000
    train_ids = train_ids[0:n_images]


    X = np.zeros((len(train_ids), TARGET_SIZE[0][0], TARGET_SIZE[0][1], 3), dtype=np.uint8)
    y = np.zeros((len(train_ids), TARGET_SIZE[0][0], TARGET_SIZE[0][1], 1), dtype=np.bool)

    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = cv2.imread(path)
        img = cv2.resize(img, (TARGET_SIZE[0][0], TARGET_SIZE[0][1]))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        X[n] = img
        mask_path = TRAIN_MASK_PATH + id_.replace(".png", "_diff.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (TARGET_SIZE[0][0], TARGET_SIZE[0][1]))
        mask = np.expand_dims(mask, axis=-1)
        y[n] = mask
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = Unet().get_model()
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=0.00001, first_decay_steps=150, t_mul=2, m_mul=0.9, alpha=0.0, name=None)
        checkpoint = ModelCheckpoint(f'/home/RT_Paper/model/unet_IP.h5', monitor='val_loss', save_best_only=True, mode="auto")
        early_stop = EarlyStopping(monitor='val_loss', patience=100, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.99, patience=4, mode='min', min_lr=1e-8)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='binary_focal_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.FalseNegatives(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.TruePositives()])
    
    history = model.fit(X_train, y_train, validation_data=(X_test,y_test), batch_size=8, epochs=300, callbacks=[checkpoint, early_stop, reduce_lr])
    TEST_PATH = '/home/RT_Paper/data/IP/test/1/'
    test_ids = os.listdir(TEST_PATH)
    test_images = np.zeros((len(test_ids), TARGET_SIZE[0][0], TARGET_SIZE[0][1], 3), dtype=np.uint8)
    image_name = []
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_
        img = cv2.imread(path)
        img = cv2.resize(img, (TARGET_SIZE[0][0], TARGET_SIZE[0][1]))
        test_images[n] = img
        image_name.append(id_)
    os.makedirs(f'/home/RT_Paper/log/Unet_{DATE}/', exist_ok=True)
    for i in range(len(test_images)):
        prediction = model.predict(test_images[i][tf.newaxis, ...])[0]
        predicted_mask = (prediction > 0.5).astype(np.uint8)
        #3채널로 변경
        predicted_mask = np.concatenate((predicted_mask, predicted_mask, predicted_mask), axis=2)
        predicted_mask = cv2.normalize(predicted_mask, None, 0, 255, cv2.NORM_MINMAX)
        original_image = test_images[i]
        concat_image = np.concatenate((original_image, predicted_mask), axis=1)
        cv2.imwrite(f'/home/RT_Paper/log/Unet_{DATE}/'+image_name[i], concat_image)
        

if __name__ == "__main__":
    main()
