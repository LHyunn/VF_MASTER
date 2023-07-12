import tensorflow as tf

import warnings

warnings.filterwarnings("ignore")
import os
#텐서플로우 출력 로그 레벨 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow_addons as tfa
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.applications import EfficientNetB0, ResNet50, VGG16, Xception, MobileNet, DenseNet121, NASNetLarge
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling, Dropout, BatchNormalization, Activation


class CNN_Model:
    def __init__(
        self,
        target_size,
        learning_rate,
        loss_func,
        metrics,
        model_dir,
        data_name,
        model_name,
        log_dir
    ):
        """_summary_
        Args:
            target_size (tuple): _description_
            learning_rate (_type_): _description_
            loss (_type_): _description_
            metrics (_type_): _description_
        """
        self.target_size = target_size
        self.learning_rate = learning_rate

        if loss_func == "focal_loss":
            self.loss = tfa.losses.SigmoidFocalCrossEntropy()
        elif loss_func == "binary_crossentropy":
            self.loss = (tf.keras.losses.BinaryCrossentropy(from_logits=False),)
        self.metrics = metrics
        self.model_dir = model_dir
        self.data_name = data_name
        self.model_name = model_name
        self.log_dir = log_dir

    def model(self):
        mirrored_strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))
        with mirrored_strategy.scope():
            model = tf.keras.models.Sequential(
                [
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
                    tf.keras.layers.MaxPool2D(2, 2),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(512, activation="relu"),
                    tf.keras.layers.Dense(1, activation="sigmoid"),
                ]
            )

            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.6,
                patience=10,
                verbose=1,
                min_delta=1e-3,
                min_lr=1e-6,
                mode="max",
            )

            checkpoint = ModelCheckpoint(
                f"{self.model_dir}/{self.model_name}_{self.data_name}.h5",
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
                mode="auto",
            )
            earlystopping = EarlyStopping(monitor="val_loss", patience=30)
            adam_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            csv_logger = tf.keras.callbacks.CSVLogger(
                f"{self.log_dir}/{self.model_name}_{self.data_name}.csv",
                append=True,
            )
            progress_bar = tf.keras.callbacks.ProgbarLogger(count_mode="steps")
            
            

        model.compile(optimizer=adam_optimizer, loss=self.loss, metrics=self.metrics)
        return model, [reduce_lr, checkpoint, earlystopping, csv_logger, progress_bar]

class VGG16_Model:
    def __init__(
        self,
        target_size,
        learning_rate,
        loss_func,
        metrics,
        model_dir,
        data_name,
        model_name,
        log_dir
    ):
        """_summary_
        Args:
            target_size (tuple): _description_
            learning_rate (_type_): _description_
            loss (_type_): _description_
            metrics (_type_): _description_
        """
        self.target_size = target_size
        self.learning_rate = learning_rate

        if loss_func == "focal_loss":
            self.loss = tfa.losses.SigmoidFocalCrossEntropy()
        elif loss_func == "binary_crossentropy":
            self.loss = (tf.keras.losses.BinaryCrossentropy())
        self.metrics = metrics
        self.model_dir = model_dir
        self.data_name = data_name
        self.model_name = model_name
        self.log_dir = log_dir

    def model(self):
        mirrored_strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))
        with mirrored_strategy.scope():
            model = tf.keras.models.Sequential(
                            [
                    Rescaling(1.0 / 255, input_shape=(self.target_size[0], self.target_size[1], self.target_size[2])),
                    Conv2D(64, (3, 3), activation='relu', padding='same'),
                    Conv2D(64, (3, 3), activation='relu', padding='same'),
                    MaxPooling2D((2, 2), strides=(2, 2)),
                    
                    Conv2D(128, (3, 3), activation='relu', padding='same'),
                    Conv2D(128, (3, 3), activation='relu', padding='same'),
                    MaxPooling2D((2, 2), strides=(2, 2)),
                    
                    Conv2D(256, (3, 3), activation='relu', padding='same'),
                    Conv2D(256, (3, 3), activation='relu', padding='same'),
                    Conv2D(256, (3, 3), activation='relu', padding='same'),
                    MaxPooling2D((2, 2), strides=(2, 2)),
                    
                    Conv2D(512, (3, 3), activation='relu', padding='same'),
                    Conv2D(512, (3, 3), activation='relu', padding='same'),
                    Conv2D(512, (3, 3), activation='relu', padding='same'),
                    MaxPooling2D((2, 2), strides=(2, 2)),
                    
                    Conv2D(512, (3, 3), activation='relu', padding='same'),
                    Conv2D(512, (3, 3), activation='relu', padding='same'),
                    Conv2D(512, (3, 3), activation='relu', padding='same'),
                    MaxPooling2D((2, 2), strides=(2, 2)),
                    
                    Flatten(),
                    Dense(4096, activation='relu'),
                    Dense(4096, activation='relu'),
                    Dense(2048, activation='relu'),
                    Dense(1024, activation='relu'),
                    Dense(1, activation='sigmoid')
                ]
            )
                    
            
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.6,
                patience=10,
                verbose=1,
                min_delta=1e-3,
                min_lr=1e-6,
                mode="max",
            )

            checkpoint = ModelCheckpoint(
                f"{self.model_dir}/{self.model_name}_{self.data_name}.h5",
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
                mode="auto",
            )
            earlystopping = EarlyStopping(monitor="val_loss", patience=30)
            adam_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            csv_logger = tf.keras.callbacks.CSVLogger(
                f"{self.log_dir}/{self.model_name}_{self.data_name}.csv",
                append=True,
            )
            progress_bar = tf.keras.callbacks.ProgbarLogger(count_mode="steps")
            
            

        model.compile(optimizer=adam_optimizer, loss=self.loss, metrics=self.metrics)
        return model, [reduce_lr, checkpoint, earlystopping, csv_logger, progress_bar]
    
class ResNet50_Model:
    def __init__(
        self,
        target_size,
        learning_rate,
        loss_func,
        metrics,
        model_dir,
        data_name,
        model_name,
        log_dir
    ):
        """_summary_
        Args:
            target_size (tuple): _description_
            learning_rate (_type_): _description_
            loss (_type_): _description_
            metrics (_type_): _description_
        """
        self.target_size = target_size
        self.learning_rate = learning_rate

        if loss_func == "focal_loss":
            self.loss = tfa.losses.SigmoidFocalCrossEntropy()
        elif loss_func == "binary_crossentropy":
            self.loss = (tf.keras.losses.BinaryCrossentropy(from_logits=False),)
        self.metrics = metrics
        self.model_dir = model_dir
        self.data_name = data_name
        self.model_name = model_name
        self.log_dir = log_dir

    def model(self):
        mirrored_strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))
        with mirrored_strategy.scope():
            ResNet50 = tf.keras.applications.ResNet50(
                include_top=False,
                weights=None,
                input_shape=(
                    self.target_size[0],
                    self.target_size[1],
                    self.target_size[2],
                ),
            )
            ResNet50.trainable = False
            model = tf.keras.models.Sequential(
                [   
                    tf.keras.layers.experimental.preprocessing.Rescaling(
                            1.0 / 255,
                            input_shape=(
                                self.target_size[0],
                                self.target_size[1],
                                self.target_size[2],
                            ),
                        ),
                    ResNet50,
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(1)
                ]
            )
            
            
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.6,
                patience=10,
                verbose=1,
                min_delta=1e-3,
                min_lr=1e-6,
                mode="max",
            )

            checkpoint = ModelCheckpoint(
                f"{self.model_dir}/{self.model_name}_{self.data_name}.h5",
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
                mode="auto",
            )
            earlystopping = EarlyStopping(monitor="val_loss", patience=30)
            adam_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            csv_logger = tf.keras.callbacks.CSVLogger(
                f"{self.log_dir}/{self.model_name}_{self.data_name}.csv",
                append=True,
            )
            progress_bar = tf.keras.callbacks.ProgbarLogger(count_mode="steps")
            
            

        model.compile(optimizer=adam_optimizer, loss=self.loss, metrics=self.metrics)
        return model, [reduce_lr, checkpoint, earlystopping, csv_logger, progress_bar]
    
class EfficientNetB0_Model:
    def __init__(
        self,
        target_size,
        learning_rate,
        loss_func,
        metrics,
        model_dir,
        data_name,
        model_name,
        log_dir
    ):
        """_summary_
        Args:
            target_size (tuple): _description_
            learning_rate (_type_): _description_
            loss (_type_): _description_
            metrics (_type_): _description_
        """
        self.target_size = target_size
        self.learning_rate = learning_rate

        if loss_func == "focal_loss":
            self.loss = tfa.losses.SigmoidFocalCrossEntropy()
        elif loss_func == "binary_crossentropy":
            self.loss = (tf.keras.losses.BinaryCrossentropy(from_logits=False),)
        self.metrics = metrics
        self.model_dir = model_dir
        self.data_name = data_name
        self.model_name = model_name
        self.log_dir = log_dir

    def model(self):
        mirrored_strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))
        with mirrored_strategy.scope():
            EfficientNetB0 = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=(
                    self.target_size[0],
                    self.target_size[1],
                    self.target_size[2],
                ),
            )
            EfficientNetB0.trainable = False
            model = tf.keras.models.Sequential(
                [   
                    tf.keras.layers.experimental.preprocessing.Rescaling(
                            1.0 / 255,
                            input_shape=(
                                self.target_size[0],
                                self.target_size[1],
                                self.target_size[2],
                            ),
                        ),
                    EfficientNetB0,
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(1)
                ]
            )
            
            
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.6,
                patience=10,
                verbose=1,
                min_delta=1e-3,
                min_lr=1e-6,
                mode="max",
            )

            checkpoint = ModelCheckpoint(
                f"{self.model_dir}/{self.model_name}_{self.data_name}.h5",
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
                mode="auto",
            )
            earlystopping = EarlyStopping(monitor="val_loss", patience=30)
            adam_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            csv_logger = tf.keras.callbacks.CSVLogger(
                f"{self.log_dir}/{self.model_name}_{self.data_name}.csv",
                append=True,
            )
            progress_bar = tf.keras.callbacks.ProgbarLogger(count_mode="steps")
            
            

        model.compile(optimizer=adam_optimizer, loss=self.loss, metrics=self.metrics)
        return model, [reduce_lr, checkpoint, earlystopping, csv_logger, progress_bar]
    
class Xception_Model:
    def __init__(
        self,
        target_size,
        learning_rate,
        loss_func,
        metrics,
        model_dir,
        data_name,
        model_name,
        log_dir
    ):
        """_summary_
        Args:
            target_size (tuple): _description_
            learning_rate (_type_): _description_
            loss (_type_): _description_
            metrics (_type_): _description_
        """
        self.target_size = target_size
        self.learning_rate = learning_rate

        if loss_func == "focal_loss":
            self.loss = tfa.losses.SigmoidFocalCrossEntropy()
        elif loss_func == "binary_crossentropy":
            self.loss = (tf.keras.losses.BinaryCrossentropy(from_logits=False),)
        self.metrics = metrics
        self.model_dir = model_dir
        self.data_name = data_name
        self.model_name = model_name
        self.log_dir = log_dir

    def model(self):
        mirrored_strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))
        with mirrored_strategy.scope():
            Xception = tf.keras.applications.Xception(
                include_top=False,
                weights=None,
                input_shape=(
                    self.target_size[0],
                    self.target_size[1],
                    self.target_size[2],
                ),
            )
            Xception.trainable = False
            model = tf.keras.models.Sequential(
                [   
                    tf.keras.layers.experimental.preprocessing.Rescaling(
                            1.0 / 127.5, offset=-1,
                            input_shape=(
                                self.target_size[0],
                                self.target_size[1],
                                self.target_size[2],
                            ),
                        ),
                    Xception,
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(1)
                ]
            )
            
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.6,
                patience=10,
                verbose=1,
                min_delta=1e-3,
                min_lr=1e-6,
                mode="max",
            )

            checkpoint = ModelCheckpoint(
                f"{self.model_dir}/{self.model_name}_{self.data_name}.h5",
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
                mode="auto",
            )
            earlystopping = EarlyStopping(monitor="val_loss", patience=30)
            adam_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            csv_logger = tf.keras.callbacks.CSVLogger(
                f"{self.log_dir}/{self.model_name}_{self.data_name}.csv",
                append=True,
            )
            progress_bar = tf.keras.callbacks.ProgbarLogger(count_mode="steps")
            
            

        model.compile(optimizer=adam_optimizer, loss=self.loss, metrics=self.metrics)
        return model, [reduce_lr, checkpoint, earlystopping, csv_logger, progress_bar]
    
class MobileNet_Model:
    def __init__(
        self,
        target_size,
        learning_rate,
        loss_func,
        metrics,
        model_dir,
        data_name,
        model_name,
        log_dir
    ):
        """_summary_
        Args:
            target_size (tuple): _description_
            learning_rate (_type_): _description_
            loss (_type_): _description_
            metrics (_type_): _description_
        """
        self.target_size = target_size
        self.learning_rate = learning_rate

        if loss_func == "focal_loss":
            self.loss = tfa.losses.SigmoidFocalCrossEntropy()
        elif loss_func == "binary_crossentropy":
            self.loss = (tf.keras.losses.BinaryCrossentropy(from_logits=False),)
        self.metrics = metrics
        self.model_dir = model_dir
        self.data_name = data_name
        self.model_name = model_name
        self.log_dir = log_dir

    def model(self):
        mirrored_strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))
        with mirrored_strategy.scope():
            MobileNet = tf.keras.applications.MobileNet(
                include_top=False,
                weights=None,
                input_shape=(
                    self.target_size[0],
                    self.target_size[1],
                    self.target_size[2],
                ),
            )
            MobileNet.trainable = False
            model = tf.keras.models.Sequential(
                [   
                    tf.keras.layers.experimental.preprocessing.Rescaling(
                            1.0 / 255,
                            input_shape=(
                                self.target_size[0],
                                self.target_size[1],
                                self.target_size[2],
                            ),
                        ),
                    MobileNet,
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(1)
                ]
            )
            
            
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.6,
                patience=10,
                verbose=1,
                min_delta=1e-3,
                min_lr=1e-6,
                mode="max",
            )

            checkpoint = ModelCheckpoint(
                f"{self.model_dir}/{self.model_name}_{self.data_name}.h5",
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
                mode="auto",
            )
            earlystopping = EarlyStopping(monitor="val_loss", patience=30)
            adam_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            csv_logger = tf.keras.callbacks.CSVLogger(
                f"{self.log_dir}/{self.model_name}_{self.data_name}.csv",
                append=True,
            )
            progress_bar = tf.keras.callbacks.ProgbarLogger(count_mode="steps")
            
            

        model.compile(optimizer=adam_optimizer, loss=self.loss, metrics=self.metrics)
        return model, [reduce_lr, checkpoint, earlystopping, csv_logger, progress_bar]
    
class DenseNet121_Model:
    def __init__(
        self,
        target_size,
        learning_rate,
        loss_func,
        metrics,
        model_dir,
        data_name,
        model_name,
        log_dir
    ):
        """_summary_
        Args:
            target_size (tuple): _description_
            learning_rate (_type_): _description_
            loss (_type_): _description_
            metrics (_type_): _description_
        """
        self.target_size = target_size
        self.learning_rate = learning_rate

        if loss_func == "focal_loss":
            self.loss = tfa.losses.SigmoidFocalCrossEntropy()
        elif loss_func == "binary_crossentropy":
            self.loss = (tf.keras.losses.BinaryCrossentropy(from_logits=False),)
        self.metrics = metrics
        self.model_dir = model_dir
        self.data_name = data_name
        self.model_name = model_name
        self.log_dir = log_dir

    def model(self):
        mirrored_strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))
        with mirrored_strategy.scope():
            DenseNet121 = tf.keras.applications.DenseNet121(
                include_top=False,
                weights=None,
                input_shape=(
                    self.target_size[0],
                    self.target_size[1],
                    self.target_size[2],
                ),
            )
            DenseNet121.trainable = False
            model = tf.keras.models.Sequential(
                [   
                    tf.keras.layers.experimental.preprocessing.Rescaling(
                            1.0 / 255,
                            input_shape=(
                                self.target_size[0],
                                self.target_size[1],
                                self.target_size[2],
                            ),
                        ),
                    DenseNet121,
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(1)
                ]
            )
            
            
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.6,
                patience=10,
                verbose=1,
                min_delta=1e-3,
                min_lr=1e-6,
                mode="max",
            )

            checkpoint = ModelCheckpoint(
                f"{self.model_dir}/{self.model_name}_{self.data_name}.h5",
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
                mode="auto",
            )
            earlystopping = EarlyStopping(monitor="val_loss", patience=30)
            adam_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            csv_logger = tf.keras.callbacks.CSVLogger(
                f"{self.log_dir}/{self.model_name}_{self.data_name}.csv",
                append=True,
            )
            progress_bar = tf.keras.callbacks.ProgbarLogger(count_mode="steps")
            
            

        model.compile(optimizer=adam_optimizer, loss=self.loss, metrics=self.metrics)
        return model, [reduce_lr, checkpoint, earlystopping, csv_logger, progress_bar]
    
class NASNetLarge_Model:
    def __init__(
        self,
        target_size,
        learning_rate,
        loss_func,
        metrics,
        model_dir,
        data_name,
        model_name,
        log_dir
    ):
        """_summary_
        Args:
            target_size (tuple): _description_
            learning_rate (_type_): _description_
            loss (_type_): _description_
            metrics (_type_): _description_
        """
        self.target_size = target_size
        self.learning_rate = learning_rate

        if loss_func == "focal_loss":
            self.loss = tfa.losses.SigmoidFocalCrossEntropy()
        elif loss_func == "binary_crossentropy":
            self.loss = (tf.keras.losses.BinaryCrossentropy(from_logits=False),)
        self.metrics = metrics
        self.model_dir = model_dir
        self.data_name = data_name
        self.model_name = model_name
        self.log_dir = log_dir

    def model(self):
        mirrored_strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))
        with mirrored_strategy.scope():
            NASNetLarge = tf.keras.applications.NASNetLarge(
                include_top=False,
                weights=None,
                input_shape=(
                    self.target_size[0],
                    self.target_size[1],
                    self.target_size[2],
                ),
            )
            NASNetLarge.trainable = False
            model = tf.keras.models.Sequential(
                [   
                    tf.keras.layers.experimental.preprocessing.Rescaling(
                        1.0 / 255,
                        input_shape=(
                            self.target_size[0],
                            self.target_size[1],
                            self.target_size[2],
                        ),
                    ),
                    NASNetLarge,
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(1)
                ]
            )
            
            
            
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.6,
                patience=10,
                verbose=1,
                min_delta=1e-3,
                min_lr=1e-6,
                mode="max",
            )

            checkpoint = ModelCheckpoint(
                f"{self.model_dir}/{self.model_name}_{self.data_name}.h5",
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
                mode="auto",
            )
            earlystopping = EarlyStopping(monitor="val_loss", patience=30)
            adam_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            csv_logger = tf.keras.callbacks.CSVLogger(
                f"{self.log_dir}/{self.model_name}_{self.data_name}.csv",
                append=True,
            )
            progress_bar = tf.keras.callbacks.ProgbarLogger(count_mode="steps")
            
            

        model.compile(optimizer=adam_optimizer, loss=self.loss, metrics=self.metrics)
        return model, [reduce_lr, checkpoint, earlystopping, csv_logger, progress_bar]