import os
#텐서플로우 출력 로그 레벨 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
#Model
import config

class BaseModel:
    def __init__(self):
        self.target_size = config.TARGET_SIZE
        
    def get_model(self):
        raise NotImplementedError
    
class Unet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_filters = 128
        
    def filter_size(self, filters):
        return self.init_filters * (2 ** filters)
        
    def get_model(self):
        inputs = tf.keras.layers.Input((self.target_size[0][0], self.target_size[0][1], 3))
        inputs = tf.keras.layers.Rescaling(1./255)(inputs)
        #Contraction path
        c1 = tf.keras.layers.Conv2D(self.filter_size(0), (3, 3), activation='relu', padding='same')(inputs)
        c1 = tf.keras.layers.Conv2D(self.filter_size(0), (3, 3), activation='relu', padding='same')(c1)
        r1 = tf.keras.layers.ReLU()(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(r1)
        
        c2 = tf.keras.layers.Conv2D(self.filter_size(1), (3, 3), activation='relu', padding='same')(p1)
        c2 = tf.keras.layers.Conv2D(self.filter_size(1), (3, 3), activation='relu', padding='same')(c2)
        r2 = tf.keras.layers.ReLU()(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(r2)
        
        c3 = tf.keras.layers.Conv2D(self.filter_size(2), (3, 3), activation='relu', padding='same')(p2)
        c3 = tf.keras.layers.Conv2D(self.filter_size(2), (3, 3), activation='relu', padding='same')(c3)
        r3 = tf.keras.layers.ReLU()(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(r3)
        
        c4 = tf.keras.layers.Conv2D(self.filter_size(3), (3, 3), activation='relu', padding='same')(p3)
        c4 = tf.keras.layers.Conv2D(self.filter_size(3), (3, 3), activation='relu', padding='same')(c4)
        r4 = tf.keras.layers.ReLU()(c4)
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(r4)
        
        c5 = tf.keras.layers.Conv2D(self.filter_size(4), (3, 3), activation='relu', padding='same')(p4)
        r5 = tf.keras.layers.ReLU()(c5)
        c5 = tf.keras.layers.Conv2D(self.filter_size(4), (3, 3), activation='relu', padding='same')(r5)

        #Expansive path 
        u6 = tf.keras.layers.Conv2DTranspose(self.filter_size(3), (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        u6 = tf.keras.layers.ReLU()(u6)

        
        u7 = tf.keras.layers.Conv2DTranspose(self.filter_size(2), (2, 2), strides=(2, 2), padding='same')(u6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        u7 = tf.keras.layers.ReLU()(u7)

        
        u8 = tf.keras.layers.Conv2DTranspose(self.filter_size(1), (2, 2), strides=(2, 2), padding='same')(u7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        u8 = tf.keras.layers.ReLU()(u8)
        
        u9 = tf.keras.layers.Conv2DTranspose(self.filter_size(0), (2, 2), strides=(2, 2), padding='same')(u8)
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
        u9 = tf.keras.layers.ReLU()(u9)

        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(u9)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        return model