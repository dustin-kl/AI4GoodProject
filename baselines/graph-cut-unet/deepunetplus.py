# code modified from: https://github.com/TKouyama/DeepUnet_Keras/blob/master/unet_deep_modified_se.py

from keras.models import Model
from keras.layers import Input, Dense, Multiply
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv2DTranspose, UpSampling2D
from keras.layers.merge import concatenate, add, average
from keras.layers import LeakyReLU, BatchNormalization, Activation, Dropout, MaxPooling2D, GlobalAveragePooling2D
from keras import regularizers


class DeepUNetPlus(object):
    def __init__(self, input_channel_count, output_channel_count, first_layer_filter_count):
        self.INPUT_IMAGE_SIZE = 256
        self.CONCATENATE_AXIS = -1

        inputs = Input((self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, input_channel_count))

        filter_count = first_layer_filter_count  # 32
        enc1, res_enc1 = self._add_encoding_layer(filter_count, inputs, True)  # 256 => 128
        enc2, res_enc2 = self._add_encoding_layer(filter_count, enc1, True)  # 128 =>  64
        enc3, res_enc3 = self._add_encoding_layer(filter_count * 1, enc2, True)  # 64 =>  32
        enc4, res_enc4 = self._add_encoding_layer(filter_count * 2, enc3, True)  # 32 =>  16
        enc5, res_enc5 = self._add_encoding_layer(filter_count * 4, enc4, True)  # 16 =>   8
        enc6, res_enc6 = self._add_encoding_layer(filter_count * 8, enc5, True)  # 8 =>   4
        enc7, res_enc7 = self._add_encoding_layer(filter_count * 16, enc6, False)  # 4 =>   4
        enc8, res_enc8 = self._add_encoding_layer(filter_count * 32, enc7, False)  # 4 =>   4

        dec1 = self._add_decoding_layer(filter_count * 32, True, enc8, res_enc8, False)  # 4 => 4
        dec2 = self._add_decoding_layer(filter_count * 16, True, dec1, res_enc7, True)  # 4 => 8
        dec3 = self._add_decoding_layer(filter_count * 8, True, dec2, res_enc6, True)  # 8 => 16
        dec4 = self._add_decoding_layer(filter_count * 4, True, dec3, res_enc5, True)  # 16 => 32
        dec5 = self._add_decoding_layer(filter_count * 2, True, dec4, res_enc4, True)  # 32 => 64
        dec6 = self._add_decoding_layer(filter_count * 1, True, dec5, res_enc3, True)  # 64 => 128
        dec7 = self._add_decoding_layer(filter_count, True, dec6, res_enc2, True)  # 128 => 256

        dec8 = concatenate([dec7, res_enc1], axis=self.CONCATENATE_AXIS)
        dec8 = Conv2D(filter_count, 3, strides=1, padding="same", kernel_initializer='he_uniform')(dec8)
        dec8 = BatchNormalization()(dec8)
        dec8 = Activation(activation='relu')(dec8)

        dec8 = Conv2D(output_channel_count, 1, strides=1, padding="same")(dec8)
        dec8 = Activation(activation='sigmoid')(dec8)

        self.UNET = Model(inputs=inputs, outputs=dec8)

    def _add_encoding_layer(self, filter_count, sequence, ds):
        res_sequence = sequence

        res_sequence = BatchNormalization()(res_sequence)
        res_sequence = Activation(activation='relu')(res_sequence)
        res_sequence = Conv2D(filter_count, 3, strides=1, padding="same", kernel_initializer='he_uniform')(res_sequence)

        res_sequence = BatchNormalization()(res_sequence)
        res_sequence = Activation(activation='relu')(res_sequence)
        res_sequence = Conv2D(filter_count, 3, strides=1, padding="same", kernel_initializer='he_uniform')(res_sequence)

        res_sequence = self._se_block(res_sequence, filter_count)

        shortcut_sequence = sequence
        shortcut_sequence = Conv2D(filter_count, 1, strides=1, padding="same")(shortcut_sequence)

        add_sequence = add([res_sequence, shortcut_sequence])
        add_sequence = Activation(activation='relu')(add_sequence)

        if ds:
            # Downsampling with stride
            new_sequence = Conv2D(filter_count, 2, strides=2, padding="same")(add_sequence)
        else:
            new_sequence = Conv2D(filter_count, 1, strides=1, padding="same")(add_sequence)

        return new_sequence, add_sequence

    def _add_decoding_layer(self, filter_count, add_drop_layer, sequence, res_enc, us):
        res_sequence = sequence
        res_sequence = concatenate([res_sequence, res_enc], axis=self.CONCATENATE_AXIS)

        res_sequence = BatchNormalization()(res_sequence)
        res_sequence = Activation(activation='relu')(res_sequence)
        res_sequence = Conv2D(int(filter_count * 2), 3, strides=1, padding="same", kernel_initializer='he_uniform')(res_sequence)


        res_sequence = BatchNormalization()(res_sequence)
        res_sequence = Activation(activation='relu')(res_sequence)
        res_sequence = Conv2D(filter_count, 3, strides=1, padding="same", kernel_initializer='he_uniform')(res_sequence)


        res_sequence = self._se_block(res_sequence, filter_count)


        shortcut_sequence = sequence

        shortcut_sequence = Conv2D(filter_count, 1, strides=1, padding="same")(shortcut_sequence)

        add_sequence = add([res_sequence, shortcut_sequence])
        add_sequence = Activation(activation='relu')(add_sequence)

        if add_drop_layer:
            add_sequence = Dropout(0.2)(add_sequence)

        if us:
            new_sequence = Conv2DTranspose(filter_count, 2, strides=2, padding="same", kernel_initializer='he_uniform')(
                add_sequence)

        else:
            new_sequence = Conv2D(filter_count, 1, strides=1, padding="same")(add_sequence)

        return new_sequence

    def get_model(self):
        return self.UNET

    def _se_block(self, input, channels, r=8):
        x = GlobalAveragePooling2D()(input)
        x = Dense(channels // r, activation="relu", kernel_initializer='Ones')(x)
        x = Dense(channels, activation="sigmoid", kernel_initializer='Ones')(x)

        return Multiply()([input, x])
