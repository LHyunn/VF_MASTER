import os

import optuna
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import warnings
warnings.filterwarnings("ignore")
from config import *
from modules.util import *
from models.model_v3 import *

def main(**kwargs):
    model_dir, log_dir, result_dir = init_train(**kwargs)
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        def objective(trial):
            
            n_cnnlayer = trial.suggest_int("n_layer", 1, 5)
            n_denselayer = trial.suggest_int("n_layer", 1, 3)
            n_filter = trial.suggest_int("n_filter", 16, 64)
            n_dense = trial.suggest_int("n_dense", 1024, 2048)
            
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Rescaling(1./255, input_shape=target_size))
            for i in range(n_cnnlayer):
                model.add(tf.keras.layers.Conv2D(n_filter, (3,3), activation='relu'))
                model.add(tf.keras.layers.MaxPooling2D())
                
            model.add(tf.keras.layers.Flatten())
            for i in range(n_denselayer):
                model.add(tf.keras.layers.Dense(n_dense, activation='relu'))
            
            
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
                    initial_learning_rate=kwargs["learning_rate"], 
                    first_decay_steps=150, 
                    t_mul=2, 
                    m_mul=0.9, 
                    alpha=0.0, 
                    name=None
                )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                loss=kwargs["loss_func"],
                metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.FalseNegatives(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.TruePositives()]
            )
            test_callback = TestCallback(model, test_image_list, test_image_label, test_image, result_dir)
            history = model.fit(
                    train_ds, 
                    validation_data=val_ds, 
                    epochs=EPOCHS, 
                    class_weight=kwargs["class_weight"], 
                    callbacks=get_callbacks(kwargs["model_type"], model_dir, log_dir)+ [test_callback],  # TestCallback 추가
                    verbose=2, 
                    workers=40, 
                    use_multiprocessing=True
                )
            draw_learning_curve(history, result_dir, DATA_PATH, kwargs["date"])
            test_pred = model.predict(test_image_list)
            return generate_report(test_pred, test_image_label, test_image, result_dir, upload=True)
    
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100)

        
    

if __name__ == "__main__":
    for target_size in TARGET_SIZE:
            # 시작 전에 공통적으로 사용되는 테스트 데이터 로드.
            print(f"test data loading...", end="")
            test_image_list, test_image_label, test_image = load_test(TEST_PATH, target_size)
            print(f"done")
            
            for data in DATA:
                for batch_size in BATCH_SIZE:
                    # 학습 데이터 로드.
                    if MODE == "train" or MODE == "alpha train":
                        train_ds, val_ds = load_train(DATA_PATH, data, target_size, batch_size)
                        
                    for model_type in MODEL:
                        for learning_rate in LEARNING_RATE:
                            for loss_func in LOSS_FUNC:
                                for weight in WEIGHT:
                                    # 학습 시작.
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



