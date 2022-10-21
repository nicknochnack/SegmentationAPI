import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Loss
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, Cropping2D, Dropout

class SegmentationModel(): 
    def __init__(self): 
        inputData = Input(shape=(256,256,3))
        c1 = Conv2D(64, 3, activation='relu', padding='same')(inputData)
        c2 = Conv2D(64, 3, activation='relu', padding='same')(c1)

        c3 = MaxPooling2D()(c2)
        c4 = Conv2D(128, 3, activation='relu', padding='same')(c3)
        c5 = Conv2D(128, 3, activation='relu', padding='same')(c4)
        c5 = Dropout(0.5)(c5)

        c6 = MaxPooling2D()(c5)
        c7 = Conv2D(256, 3, activation='relu', padding='same')(c6)
        c8 = Conv2D(256, 3, activation='relu', padding='same')(c7)
        c8 = Dropout(0.5)(c8)

        c9 = MaxPooling2D()(c8)
        c10 = Conv2D(512, 3, activation='relu', padding='same')(c9)
        c11 = Conv2D(512, 3, activation='relu', padding='same')(c10)
        c11 = Dropout(0.5)(c11)

        c12 = Conv2D(1024, 3, activation='relu', padding='same')(c11)
        c12 = Dropout(0.5)(c12)

        c12 = Concatenate()([c12, Cropping2D(cropping=((16, 16), (16, 16)))(c8)])
        u1 = Conv2DTranspose(1024, 2, (2,2))(c12)
        u2 = Conv2D(512, 3, activation='relu', padding='same')(u1)
        u3 = Conv2D(512, 3, activation='relu', padding='same')(u2)
        u3 = Dropout(0.5)(u3)

        u3 = Concatenate()([u3, Cropping2D(cropping=((32, 32), (32, 32)))(c5)])
        u4 = Conv2DTranspose(512, 2, (2,2))(u3)
        u5 = Conv2D(256, 3, activation='relu', padding='same')(u4)
        u6 = Conv2D(256, 3, activation='relu', padding='same')(u5)
        u6 = Dropout(0.5)(u6)

        u6 = Concatenate()([u6, Cropping2D(cropping=((64, 64), (64, 64)))(c2)])
        u7 = Conv2DTranspose(256, 2, (2,2))(u6)
        u8 = Conv2D(128, 3, activation='relu', padding='same')(u7)
        u9 = Conv2D(6, 1, activation='softmax', padding='same')(u8)

        self.model = Model(inputs=inputData, outputs=u9)
