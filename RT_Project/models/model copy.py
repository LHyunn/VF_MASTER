import tensorflow as tf

import warnings

warnings.filterwarnings("ignore")
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tensorflow_addons as tfa
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback


class CNN:
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
                os.path.join(
                    self.model_dir,
                    self.data_name,
                    f"{self.model_name}_{self.data_name}.h5",
                ),
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

