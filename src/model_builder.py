"""
Model Builder:

# Things to consider  in keras while building model:
# - No Dropout after Conv layer
# - Use dropout after dense layer (use mostly at the end of arch to not loose data)
# - Use BatchNorm before any activation function
#
# Regularizers:
# - Dropout
# - Weight Decay (L2) i.e. weights should be smaller. Penalizes model complexity.
# - BatchNorm (is a most)
# - Data Augmentation: e.g. fix lightning in images

# Basic CNN block
# conv > batch_norm > relu

# Room for improvement:
# - Progressive Resizing
# - Data Augmentation
"""

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, MaxPool2D, AveragePooling2D, Flatten, Add
# from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout, BatchNormalization, Activation
from src.config import config


# Extra Tips:
# use stride 2 in the middle to reduce size and increase no. of filters
# use avgpool at the end (not maxpool)

class ModelBuilder:

    def seq_conv_block(self, model, filters=32):
        model.add(Conv2D(filters=filters, kernel_size=(3, 3), strides=2, padding="same"))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation("relu"))
        return model

    def build_rmsprop_model21(self):
        # all conv layers  with strides=1
        model = Sequential(name="seq_conv_rmsprop")

        model.add(Conv2D(input_shape=(config['IMG_HEIGHT'], config['IMG_WIDTH'], 1), filters=16, kernel_size=(3, 3),
                         padding="same"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation("relu"))

        model = self.seq_conv_block(model, filters=32)
        model = self.seq_conv_block(model, filters=64)
        model = self.seq_conv_block(model, filters=128)

        model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dropout(0.5))

        model.add(Dense(500))
        model.add(Dropout(0.5))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation("relu"))

        model.add(Dense(10, activation="softmax"))

        # sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
        # optimizer = RMSprop(learning_rate=0.001)
        # adad = Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        return model

    def conv_layer(self, inputs, filters=16, num_strides=1):
        return Conv2D(filters=filters, kernel_size=(3, 3), strides=num_strides, padding='same')(inputs)

    def conv_block(self, inputs, filters=16, num_strides=1):
        x = self.conv_layer(inputs, filters, num_strides)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        return x

    def resnet_block(self, inputs, filters=16):
        x_shortcut = inputs
        x = self.conv_block(inputs, filters)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([x, x_shortcut])  # skip connection
        x = Activation('relu')(x)
        return x

    def build_resnetlike_model(self):
        inputs = Input(shape=(config['IMG_HEIGHT'], config['IMG_WIDTH'], 1))

        output_0 = self.conv_block(inputs=inputs, filters=16)
        output_0 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(output_0)

        output_1 = self.conv_block(output_0, filters=32, num_strides=2)
        output_1 = self.resnet_block(output_1, filters=32)
        output_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(output_1)

        output_2 = self.conv_block(output_1, filters=64, num_strides=2)
        output_2 = self.resnet_block(output_2, filters=64)
        output_2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(output_2)

        output_3 = Flatten()(output_2)
        output_3 = Dropout(0.5)(output_3)

        output_4 = Dense(500)(output_3)
        output_4 = Dropout(0.5)(output_4)
        output_4 = BatchNormalization(axis=-1)(output_4)
        output_4 = Activation('relu')(output_4)

        output_5 = Dense(10, activation='softmax')(output_4)

        res_model = Model(inputs=inputs, outputs=output_5, name="res_model")

        res_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        return res_model
