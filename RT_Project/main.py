import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from models import model as MM
import datetime
import matplotlib.pyplot as plt
import time
import shutil
import sys
import tensorflow_addons as tfa
def main(**kwargs):
    global DATASET_HISTORY
    global train_ds
    global val_ds
    model_name = kwargs["model"]
    data_name = kwargs["data"]
    target_size = kwargs["target_size"]
    batch_size = kwargs["batch_size"]
    learning_rate = kwargs["learning_rate"]
    loss_func = kwargs["loss_func"]
    class_weight = kwargs["weight"]
    class_weight_ = str(class_weight).replace("{", "(").replace("}", ")")
    # init model, result, log    
    os.makedirs(f"/home/VirtualFlaw/RT_Project/log/{DATE}/{data_name}/{model_name}/{target_size}_{batch_size}_{learning_rate}_{loss_func}_{class_weight_}/models", exist_ok=True)
    model_dir = f"/home/VirtualFlaw/RT_Project/log/{DATE}/{data_name}/{model_name}/{target_size}_{batch_size}_{learning_rate}_{loss_func}_{class_weight_}/models"
    os.makedirs(f"/home/VirtualFlaw/RT_Project/log/{DATE}/{data_name}/{model_name}/{target_size}_{batch_size}_{learning_rate}_{loss_func}_{class_weight_}/logs", exist_ok=True)
    log_dir = f"/home/VirtualFlaw/RT_Project/log/{DATE}/{data_name}/{model_name}/{target_size}_{batch_size}_{learning_rate}_{loss_func}_{class_weight_}/logs"
    os.makedirs(f"/home/VirtualFlaw/RT_Project/log/{DATE}/{data_name}/{model_name}/{target_size}_{batch_size}_{learning_rate}_{loss_func}_{class_weight_}/result", exist_ok=True)
    result_dir = f"/home/VirtualFlaw/RT_Project/log/{DATE}/{data_name}/{model_name}/{target_size}_{batch_size}_{learning_rate}_{loss_func}_{class_weight_}/result"


    # init model

    if data_name == "Vanilla":
        data_path = os.path.join(DATA_PATH, "Vanilla")
    elif data_name == "Augmented":
        data_path = os.path.join(DATA_PATH, "Augmented")
    elif data_name == "VF":
        data_path = os.path.join(DATA_PATH, "VF")
    elif data_name == "VF_Deformed":
        data_path = os.path.join(DATA_PATH, "VF_Deformed")
    model = None
    
    if model_name == "CNN":
        model = MM.CNN_Model(
            target_size,
            learning_rate,
            loss_func,
            model_dir,
            data_name,
            model_name,
            log_dir,
        )

    elif model_name == "VGG16":
        model = MM.VGG16_Model(
            target_size,
            learning_rate,
            loss_func,
            model_dir,
            data_name,
            model_name,
            log_dir,
        )
    
    elif model_name == "ResNet50":
        model = MM.ResNet50_Model(
            target_size,
            learning_rate,
            loss_func,
            model_dir,
            data_name,
            model_name,
            log_dir,
        )

    elif model_name == "EfficientNetB0":
        model = MM.EfficientNetB0_Model(
            target_size,
            learning_rate,
            loss_func,
            model_dir,
            data_name,
            model_name,
            log_dir,
        )
    
    elif model_name == "Xception":
        model = MM.Xception_Model(
            target_size,
            learning_rate,
            loss_func,
            model_dir,
            data_name,
            model_name,
            log_dir,
        )

    elif model_name == "MobileNetV2":
        model = MM.MobileNetV2_Model(
            target_size,
            learning_rate,
            loss_func,
            model_dir,
            data_name,
            model_name,
            log_dir,
        )

    elif model_name == "DenseNet121":
        model = MM.DenseNet121_Model(
            target_size,
            learning_rate,
            loss_func,
            model_dir,
            data_name,
            model_name,
            log_dir,
        )

    elif model_name == "NASNetLarge":
        model = MM.NASNetLarge_Model(
            target_size,
            learning_rate,
            loss_func,
            model_dir,
            data_name,
            model_name,
            log_dir,
        )

    elif model_name == "NASNetMobile":
        model = MM.NASNetMobile_Model(
            target_size,
            learning_rate,
            loss_func,
            model_dir,
            data_name,
            model_name,
            log_dir,
        )
        
    elif model_name == "InceptionV3":
        model = MM.InceptionV3_Model(
            target_size,
            learning_rate,
            loss_func,
            model_dir,
            data_name,
            model_name,
            log_dir,
        )
    
    elif model_name == "InceptionResNetV2":
        model = MM.InceptionResNetV2_Model(
            target_size,
            learning_rate,
            loss_func,
            model_dir,
            data_name,
            model_name,
            log_dir,
        )
        
    else:
        print("Wrong model name")
        sys.exit(1)
    model, callbacks = model.build_model()

    if DATASET_HISTORY != [data_name, target_size, batch_size]:

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(data_path, "Train"),
            validation_split=0.3,
            subset="training",
            seed=123,
            image_size=(target_size[0], target_size[1]),
            batch_size=batch_size,
            label_mode="binary",
            color_mode="grayscale",
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(data_path, "Train"),
            validation_split=0.3,
            subset="validation",
            seed=123,
            image_size=(target_size[0], target_size[1]),
            batch_size=batch_size,
            label_mode="binary",
            color_mode="grayscale",
        )

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        DATASET_HISTORY = [data_name, target_size, batch_size]

    elif DATASET_HISTORY == [data_name, target_size, batch_size]:
        pass

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCH, callbacks=callbacks, verbose=1, class_weight=class_weight
    )

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
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


if __name__ == "__main__":
    
    tf.random.set_seed(42)
    os.environ["PYTHONHASHSEED"] = str(42)

    DATE = datetime.datetime.now().strftime("%Y%m%d%H%M")
    MODEL = ["CNN", "VGG16", "ResNet50", "InceptionV3", "MobileNetV2", "DenseNet121", "Xception", "EfficientNetB0"]
    #Model = ["CNN", "VGG16", "ResNet50", "InceptionV3", "InceptionResNetV2", "MobileNetV2", "DenseNet121", "Xception", "EfficientNetB0", "NASNetMobile", "NASNetLarge"]
    DATA = ["Augmented", "VF"]  # ["Vanilla", "Augmented", "VF","VF_Deformed"]
    TARGET_SIZE = [(512, 512, 1)]  # [(512, 512, 1), (1024, 1024, 1)]
    BATCH_SIZE = [64]  # [64, 128]
    EPOCH = 1000
    LOSS = ["binary_crossentropy"]  # ["binary_crossentropy", "focal_loss"]
    LEARNING_RATE = [0.0001]  # [0.0001, 0.0002, 0.0003]
    CLASS_WEIGHT = [{0: 1, 1: 1}]  # {0: 1, 1:  1}
    # --------------------------------------------
    DATA_PATH = "/home/VirtualFlaw/RT_Project/data/Dataset_DeformedVFBasedOnVanilla"
    
    DATASET = DATA_PATH.split("/")[-1]
    DATASET_HISTORY = None
    for data in DATA:
        for image_size in TARGET_SIZE:
            for batch_size in BATCH_SIZE:
                train_ds = None
                val_ds = None
                for model in MODEL:
                    for learning_rate in LEARNING_RATE:
                        for loss_func in LOSS:
                            for weight in CLASS_WEIGHT:
                                kwargs = {
                                    "model": model,
                                    "data": data,
                                    "target_size": image_size,
                                    "batch_size": batch_size,
                                    "learning_rate": learning_rate,
                                    "loss_func": loss_func,
                                    "weight": weight,
                                }
                                print(kwargs)
                                main(**kwargs)
                                

                                
                                
                                
    
    
                                
    

#nohup /usr/bin/python3 /home/VirtualFlaw/RT_Project/main.py
#실행하는 시나리오 수가 20개가 넘어가면 프로세스가 죽음.