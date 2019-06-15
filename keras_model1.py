########################################
##### SUBMISSION SYSTEM NO 1 ###########
########################################

from keras.layers import Bidirectional, Conv2D, MaxPooling2D, Input, Concatenate, SeparableConv2D
from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
import keras
keras.backend.set_image_data_format('channels_first')
from IPython import embed

def get_model(data_in, data_out, dropout_rate, nb_cnn2d_filt, pool_size,
                                rnn_size, fnn_size, weights):
    inp = Input(shape=(data_in[-3], data_in[-2], data_in[-1]))
    
    spec_mag_cnn = inp
    spec_mag_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3),
                              padding='same', kernel_initializer='uniform')(spec_mag_cnn)
    spec_mag_cnn = BatchNormalization()(spec_mag_cnn)
    spec_mag_cnn = Activation('relu')(spec_mag_cnn)
    spec_mag_cnn = MaxPooling2D(pool_size=(1, pool_size[0]))(spec_mag_cnn)
    spec_mag_cnn = Dropout(dropout_rate)(spec_mag_cnn)
    
    
    doa = spec_mag_cnn
    sed = spec_mag_cnn
    
    for i, convCnt in enumerate(pool_size):
        doa = Conv2D(filters=nb_cnn2d_filt//2, kernel_size=(3, 3),
                              padding='same', kernel_initializer='uniform')(doa)
        doa = BatchNormalization()(doa)
        doa = Activation('relu')(doa)
        doa = MaxPooling2D(pool_size=(1, pool_size[i+1]))(doa)
        doa = Dropout(dropout_rate)(doa)
        if i == 1:
            break
        
    doa = Permute((2, 1, 3))(doa)
    
    for i, convCnt in enumerate(pool_size):
        sed = Conv2D(filters=nb_cnn2d_filt//2, kernel_size=(3, 3),
                              padding='same', kernel_initializer='uniform')(sed)
        sed = BatchNormalization()(sed)
        sed = Activation('relu')(sed)
        sed = MaxPooling2D(pool_size=(1, pool_size[i+1]))(sed)
        sed = Dropout(dropout_rate)(sed)
        if i == 1:
            break
        
    sed = Permute((2, 1, 3))(sed)


    doa = Reshape((data_in[-2], -1))(doa)
    sed = Reshape((data_in[-2], -1))(sed)

    for nb_rnn_filt in rnn_size:
        doa = Bidirectional(
            GRU(nb_rnn_filt//2, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,
                return_sequences=True, kernel_initializer='uniform'),
            merge_mode='mul'
        )(doa)
            
    for nb_fnn_filt in fnn_size:
        doa = TimeDistributed(Dense(nb_fnn_filt, kernel_initializer='uniform'))(doa)
        doa = Dropout(dropout_rate)(doa)

    doa = TimeDistributed(Dense(data_out[1][-1], kernel_initializer='uniform'))(doa)
    doa = Activation('linear', name='doa_out')(doa)


    for nb_rnn_filt in rnn_size:
        sed = Bidirectional(
            GRU(nb_rnn_filt//2, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,
                return_sequences=True, kernel_initializer='uniform'),
            merge_mode='mul'
        )(sed)
            
    for nb_fnn_filt in fnn_size:
        sed = TimeDistributed(Dense(nb_fnn_filt, kernel_initializer='uniform'))(sed)
        sed = Dropout(dropout_rate)(sed)
    sed = TimeDistributed(Dense(data_out[0][-1], kernel_initializer='uniform'))(sed)
    sed = Activation('sigmoid', name='sed_out')(sed)

    model = Model(inputs=inp, outputs=[sed, doa])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'logcosh'], loss_weights=weights)

    model.summary()
    return model
