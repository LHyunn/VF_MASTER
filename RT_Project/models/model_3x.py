import os
#텐서플로우 출력 로그 레벨 설정
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
import tensorflow_addons as tfa
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import tensorflow.keras.backend as K
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import config

INPUT_SHAPE = config.TARGET_SIZE[0]

def sigmoid_3x(x):
    return tf.nn.sigmoid(x) * 3

class BaseModel:
    def __init__(self):
        self.target_size = INPUT_SHAPE
        
    def get_model(self):
        raise NotImplementedError
    
class CNN(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_model(self):
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
                tf.keras.layers.MaxPool2D(2, 2),
                tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
                #tf.keras.layers.BatchNormalization(),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4096, activation="gelu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(4096, activation="gelu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation=sigmoid_3x),
            ])
        return model
    
class VGG16(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_model(self):
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
                tf.keras.layers.Dense(1, activation=sigmoid_3x),
            ])
        return model
       
class ResNet50(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_model(self):
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
                tf.keras.layers.Dense(1, activation=sigmoid_3x),
            ])
        return model
    
class InceptionV3(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_model(self):
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
                tf.keras.layers.Dense(1, activation=sigmoid_3x),
            ])
        return model
    
class InceptionResNetV2(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_model(self):
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
                tf.keras.layers.Dense(1, activation=sigmoid_3x),
            ])
        return model
    
class MobileNetV2(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_model(self):
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
                tf.keras.layers.Dense(1, activation=sigmoid_3x),
            ])
        return model
    
class DenseNet121(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_model(self):
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
                tf.keras.layers.Dense(1, activation=sigmoid_3x),
            ])
        return model
    
class Xception(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_model(self):
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
                tf.keras.layers.Dense(1, activation=sigmoid_3x),
            ])
        return model
    
class EfficientNetB0(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_model(self):
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
                tf.keras.layers.Dense(1, activation=sigmoid_3x),
            ])
        return model
    
class NASNetMobile(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_model(self):
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
                tf.keras.layers.Dense(1, activation=sigmoid_3x),
            ])
        return model
    
class NASNetLarge(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_model(self):
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
                tf.keras.layers.Dense(1, activation=sigmoid_3x),
            ])
        return model
               
        
def get_model(model_name):
    if model_name == 'CNN':
        return CNN().get_model()
    elif model_name == 'VGG16':
        return VGG16().get_model()
    elif model_name == 'ResNet50':
        return ResNet50().get_model()
    elif model_name == 'InceptionV3':
        return InceptionV3().get_model()
    elif model_name == 'InceptionResNetV2':
        return InceptionResNetV2().get_model()
    elif model_name == 'MobileNetV2':
        return MobileNetV2().get_model()
    elif model_name == 'DenseNet121':
        return DenseNet121().get_model()
    elif model_name == 'Xception':
        return Xception().get_model()
    elif model_name == 'EfficientNetB0':
        return EfficientNetB0().get_model()
    elif model_name == 'NASNetMobile':
        return NASNetMobile().get_model()
    elif model_name == 'NASNetLarge':
        return NASNetLarge().get_model()
    else:
        raise ValueError('Invalid model name: ', model_name)
        