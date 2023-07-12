import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import warnings
warnings.filterwarnings("ignore")
from config import *
from modules.util import *
from models.model_v3 import *


def main(**kwargs):
    model_dir, log_dir, result_dir = init_train(**kwargs)
    #모델 생성. multi gpu 사용 가능.
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        if MODE == "train":
            model = get_model(kwargs["model_type"])
            lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=kwargs["learning_rate"], first_decay_steps=150, t_mul=2, m_mul=0.9, alpha=0.0, name=None)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                        loss=kwargs["loss_func"],
                        metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.FalseNegatives(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.TruePositives()])
            history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, class_weight=kwargs["class_weight"], callbacks=get_callbacks(kwargs["model_type"], model_dir, log_dir), verbose=1)
            draw_learning_curve(history, result_dir, DATA_PATH, kwargs["date"])
            
        elif MODE == "alpha train":
            model = tf.keras.models.load_model(model_dir + f"/{kwargs['model_type']}.h5")
            lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=kwargs["learning_rate"], first_decay_steps=150, t_mul=2, m_mul=0.9, alpha=0.0, name=None)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                        loss=kwargs["loss_func"],
                        metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.FalseNegatives(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.TruePositives()])
            history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, class_weight=kwargs["class_weight"], callbacks=get_callbacks(kwargs["model_type"], model_dir, log_dir), verbose=1)
            draw_learning_curve(history, result_dir, DATA_PATH, kwargs["date"])
            
        
        
    
    
    
    #모델 평가.
    model = tf.keras.models.load_model(model_dir + f"/{kwargs['model_type']}.h5")
    #model.summary()
    test_pred = model.predict(test_image_list)
    #threading.Thread(target=generate_report, args=(test_pred, test_image_label, test_image, result_dir), kwargs={"upload": True}).start()
    generate_report(test_pred, test_image_label, test_image, result_dir, upload=True)


    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    for target_size in TARGET_SIZE:
    #시작 전에 공통적으로 사용되는 테스트 데이터 로드.
        print(f"test data loading...", end="")
        test_image_list, test_image_label, test_image = load_test(TEST_PATH, target_size)
        ######################################
        print(f"done")
        for data in DATA:
            for batch_size in BATCH_SIZE:
                #학습 데이터 로드.######################
                if MODE == "train" or MODE == "alpha train":
                    train_ds, val_ds = load_train(DATA_PATH, data, target_size, batch_size)
                ####################################
                for model_type in MODEL:
                    for learning_rate in LEARNING_RATE:
                        for loss_func in LOSS_FUNC:
                            for weight in WEIGHT:
                                #학습 시작.
                                kwargs = {
                                    "model_type": model_type,
                                    "data": data,
                                    "target_size": target_size,
                                    "batch_size": batch_size,
                                    "learning_rate": learning_rate,
                                    "loss_func": loss_func,
                                    "class_weight": weight,
                                    "date": DATE,
                                }
                                main(**kwargs)
                                #학습 종료.
                                #학습 결과 저장.
                                #학습 결과 그래프로 출력.
                                #학습 결과 테스트 데이터로 평가.
                                

