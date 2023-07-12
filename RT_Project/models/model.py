import tensorflow as tf

import warnings

warnings.filterwarnings("ignore")
import os
#텐서플로우 출력 로그 레벨 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow_addons as tfa
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import tensorflow.keras.backend as K
import numpy as np

#Model = ["CNN", "VGG16", "ResNet50", "InceptionV3", "InceptionResNetV2", "MobileNetV2", "DenseNet121", "Xception", "EfficientNetB0", "NASNetMobile", "NASNetLarge"]
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



class BaseModel:
    def __init__(
        self,
        target_size,
        learning_rate,
        loss_func,
        model_dir,
        data_name,
        model_name,
        log_dir
    ):
        self.target_size = target_size
        self.learning_rate = learning_rate
        self.model_dir = model_dir
        self.data_name = data_name
        self.model_name = model_name
        self.log_dir = log_dir

        if loss_func == "focal_loss":
            self.loss_func = tfa.losses.SigmoidFocalCrossEntropy()
        elif loss_func == "binary_crossentropy":
            self.loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def callbacks(self):
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.8,
            patience=14,
            verbose=2,
            min_delta=1e-3,
            min_lr=1e-7,
            mode="auto",
        )

        checkpoint = ModelCheckpoint(
            f"{self.model_dir}/{self.model_name}.h5",
            monitor="val_loss",
            verbose=2,
            save_best_only=True,
            mode="auto",
        )

        early_stopping = EarlyStopping(
            monitor="val_loss", verbose=2, mode="auto",
            patience=28,
        )

        csv_logger = tf.keras.callbacks.CSVLogger(
            f"{self.log_dir}/{self.model_name}.csv",
            append=True,
        )
        return [reduce_lr, checkpoint, early_stopping, csv_logger]
    
    def build_model(self):
        raise NotImplementedError("build_model method not implemented.")
        
class CNN_Model(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def build_model(self):
        mirrored_strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))

        with mirrored_strategy.scope():
            model = tf.keras.models.Sequential([
                tf.keras.layers.experimental.preprocessing.Rescaling(
                    1.0 / 255,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
                tf.keras.layers.MaxPool2D(2, 2),
                tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
                tf.keras.layers.MaxPool2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPool2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPool2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPool2D(2, 2),tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4096, activation="gelu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(4096, activation="gelu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation=sigmoid_2x),
            ])
            metrics = ["accuracy", f1_m, precision_m, recall_m]
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, weight_decay=0.8914520666889757)
            model.compile(optimizer=optimizer, loss=self.loss_func, metrics=metrics)

        return model, self.callbacks()
    
class VGG16_Model(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def build_model(self):
        mirrored_strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))

        with mirrored_strategy.scope():
            model = tf.keras.models.Sequential([
                tf.keras.layers.experimental.preprocessing.Rescaling(
                    1.0 / 255,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.applications.VGG16(
                    include_top=False,
                    weights=None,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4096, activation="gelu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(4096, activation="gelu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ])
            metrics = ["accuracy", f1_m, precision_m, recall_m]
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            model.compile(optimizer=optimizer, loss=self.loss_func, metrics=metrics)

        return model, self.callbacks()
    
class ResNet50_Model(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def build_model(self):
        mirrored_strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))

        with mirrored_strategy.scope():
            model = tf.keras.models.Sequential([
                tf.keras.layers.experimental.preprocessing.Rescaling(
                    1.0 / 255,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.applications.ResNet50(
                    include_top=False,
                    weights=None,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation="gelu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ])
            metrics = ["accuracy", f1_m, precision_m, recall_m]
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            model.compile(optimizer=optimizer, loss=self.loss_func, metrics=metrics)

        return model, self.callbacks()
    
class InceptionV3_Model(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def build_model(self):
        mirrored_strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))

        with mirrored_strategy.scope():
            model = tf.keras.models.Sequential([
                tf.keras.layers.experimental.preprocessing.Rescaling(
                    1.0 / 255,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.applications.InceptionV3(
                    include_top=False,
                    weights=None,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation="gelu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ])
            metrics = ["accuracy", f1_m, precision_m, recall_m]
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            model.compile(optimizer=optimizer, loss=self.loss_func, metrics=metrics)

        return model, self.callbacks()
    
class InceptionResNetV2_Model(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def build_model(self):
        mirrored_strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))

        with mirrored_strategy.scope():
            model = tf.keras.models.Sequential([
                tf.keras.layers.experimental.preprocessing.Rescaling(
                    1.0 / 255,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.applications.InceptionResNetV2(
                    include_top=False,
                    weights=None,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation="gelu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ])
            metrics = ["accuracy", f1_m, precision_m, recall_m]
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            model.compile(optimizer=optimizer, loss=self.loss_func, metrics=metrics)

        return model, self.callbacks()
    
class MobileNetV2_Model(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def build_model(self):
        mirrored_strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))

        with mirrored_strategy.scope():
            model = tf.keras.models.Sequential([
                tf.keras.layers.experimental.preprocessing.Rescaling(
                    1.0 / 255,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.applications.MobileNetV2(
                    include_top=False,
                    weights=None,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation="gelu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ])
            metrics = ["accuracy", f1_m, precision_m, recall_m]
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            model.compile(optimizer=optimizer, loss=self.loss_func, metrics=metrics)

        return model, self.callbacks()
    
class DenseNet121_Model(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def build_model(self):
        mirrored_strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))

        with mirrored_strategy.scope():
            model = tf.keras.models.Sequential([
                tf.keras.layers.experimental.preprocessing.Rescaling(
                    1.0 / 255,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.applications.DenseNet121(
                    include_top=False,
                    weights=None,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation="gelu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ])
            metrics = ["accuracy", f1_m, precision_m, recall_m]
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            model.compile(optimizer=optimizer, loss=self.loss_func, metrics=metrics)

        return model, self.callbacks()

class Xception_Model(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def build_model(self):
        mirrored_strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))

        with mirrored_strategy.scope():
            model = tf.keras.models.Sequential([
                tf.keras.layers.experimental.preprocessing.Rescaling(
                    1.0 / 255,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.applications.Xception(
                    include_top=False,
                    weights=None,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation="gelu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ])
            metrics = ["accuracy", f1_m, precision_m, recall_m]
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            model.compile(optimizer=optimizer, loss=self.loss_func, metrics=metrics)

        return model, self.callbacks()
    
class EfficientNetB0_Model(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def build_model(self):
        mirrored_strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))

        with mirrored_strategy.scope():
            model = tf.keras.models.Sequential([
                tf.keras.layers.experimental.preprocessing.Rescaling(
                    1.0 / 255,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.applications.EfficientNetB0(
                    include_top=False,
                    weights=None,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation="gelu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ])
            metrics = ["accuracy", f1_m, precision_m, recall_m]
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            model.compile(optimizer=optimizer, loss=self.loss_func, metrics=metrics)

        return model, self.callbacks()
    
class NASNetMobile_Model(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def build_model(self):
        mirrored_strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))

        with mirrored_strategy.scope():
            model = tf.keras.models.Sequential([
                tf.keras.layers.experimental.preprocessing.Rescaling(
                    1.0 / 255,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.applications.NASNetMobile(
                    include_top=False,
                    weights=None,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation="gelu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ])
            metrics = ["accuracy", f1_m, precision_m, recall_m]
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            model.compile(optimizer=optimizer, loss=self.loss_func, metrics=metrics)

        return model, self.callbacks()
    
class NASNetLarge_Model(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def build_model(self):
        mirrored_strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))

        with mirrored_strategy.scope():
            model = tf.keras.models.Sequential([
                tf.keras.layers.experimental.preprocessing.Rescaling(
                    1.0 / 255,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.applications.NASNetLarge(
                    include_top=False,
                    weights=None,
                    input_shape=(
                        self.target_size[0],
                        self.target_size[1],
                        self.target_size[2],
                    ),
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation="gelu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ])
            metrics = ["accuracy", f1_m, precision_m, recall_m]
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            model.compile(optimizer=optimizer, loss=self.loss_func, metrics=metrics)

        return model, self.callbacks()