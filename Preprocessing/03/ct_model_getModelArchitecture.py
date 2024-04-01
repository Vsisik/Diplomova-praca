# import tensorflow as tf

from keras import Model
from keras.layers import *


def get_model(width, height, depth, type_=1):
    """
    Function creates a model architecture based on given type_ parameter
    :param width: Width number
    :param height: Height number
    :param depth: Depth number
    :param type_: Type of architecture
    """
    ############### TYPE 1 ############################
    if type_ == 1:
      inputs = Input((depth, width, height, 1))

      x = Conv3D(filters=256, kernel_size=3, activation='relu')(inputs)
      x = AveragePooling3D(pool_size=2, padding='same')(x)

      x = Conv3D(filters=128, kernel_size=3, activation='relu')(x)

      x = Conv3D(filters=64, kernel_size=3, activation='relu')(x)
      x = MaxPooling3D(pool_size=2)(x)
      x = BatchNormalization()(x)

      x = GlobalAveragePooling3D()(x)
      x = Dense(units=64, activation='relu')(x)
      x = Dropout(0.3)(x)

      outputs = Dense(units=1, activation='sigmoid')(x)

      model = Model(inputs, outputs, name='CT_Model')
      model.summary()

    ############### TYPE 2 ############################
    if type_ == 2:
      inputs = Input((depth, width, height, 1))

      x = Conv3D(filters=64, kernel_size=3, activation='relu')(inputs)
      x = AveragePooling3D(pool_size=2, padding='same')(x)
      x = BatchNormalization()(x)

      x = Conv3D(filters=128, kernel_size=3, activation='relu')(x)
      x = BatchNormalization()(x)

      x = Conv3D(filters=256, kernel_size=3, activation='relu')(x)
      x = MaxPooling3D(pool_size=2)(x)
      x = BatchNormalization()(x)

      x = GlobalAveragePooling3D()(x)
      x = Dense(units=256, activation='relu')(x)
      x = Dropout(0.3)(x)

      outputs = Dense(units=1, activation='sigmoid')(x)

      model = Model(inputs, outputs, name='CT_Model')
      model.summary()

    ############### TYPE 3 ############################
    if type_ == 3:
      inputs = Input((depth, width, height, 1))

      x = Conv3D(filters=64, kernel_size=3, activation='relu')(inputs)
      x = MaxPooling3D(pool_size=2)(x)

      x = Conv3D(filters=128, kernel_size=3, activation='relu')(x)
      x = AveragePooling3D(pool_size=2, padding='same')(x)
      x = BatchNormalization()(x)

      x = GlobalMaxPooling3D()(x)
      x = Dense(units=256, activation='relu')(x)
      x = Dropout(0.3)(x)

      outputs = Dense(units=1, activation='sigmoid')(x)

      model = Model(inputs, outputs, name='CT_Model')
      model.summary()
    return model
