########################################
##### SUBMISSION SYSTEM NO 3 ###########
########################################

from keras.layers import Bidirectional, Conv2D, MaxPooling2D, Input, Concatenate, Lambda
from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
from keras.layers.recurrent import GRU, LSTM
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
import keras
keras.backend.set_image_data_format('channels_first')
from IPython import embed


def get_model(data_in, data_out, dropout_rate, nb_cnn2d_filt, pool_size,
                                rnn_size, fnn_size, weights):
    # model definition
    ma1 = Input(shape=(data_in[-2], data_in[-1]))
    ma2 = Input(shape=(data_in[-2], data_in[-1]))
    ma3 = Input(shape=(data_in[-2], data_in[-1]))
    ma4 = Input(shape=(data_in[-2], data_in[-1]))
    
    phas1 = Input(shape=(data_in[-2], data_in[-1]))
    phas2 = Input(shape=(data_in[-2], data_in[-1]))
    phas3 = Input(shape=(data_in[-2], data_in[-1]))
    phas4 = Input(shape=(data_in[-2], data_in[-1]))
    
    mag1 = Reshape((1, data_in[-2], data_in[-1]))(ma1)
    mag2 = Reshape((1, data_in[-2], data_in[-1]))(ma2)
    mag3 = Reshape((1, data_in[-2], data_in[-1]))(ma3)
    mag4 = Reshape((1, data_in[-2], data_in[-1]))(ma4)
    
    phase1 = Reshape((1, data_in[-2], data_in[-1]))(phas1)
    phase2 = Reshape((1, data_in[-2], data_in[-1]))(phas2)
    phase3 = Reshape((1, data_in[-2], data_in[-1]))(phas3)
    phase4 = Reshape((1, data_in[-2], data_in[-1]))(phas4)
    # CNN
    #spec_mag_cnn = Lambda(split_tensor_mag, output_shape=(4, 128, 1024))(spec_start)
    
    # MAG ================================================
    spec_mag_cnn1 = mag1
    spec_mag_cnn1 = Conv2D(filters=nb_cnn2d_filt//8, kernel_size=(3, 3),
                              padding='same', kernel_initializer='uniform')(spec_mag_cnn1)
    spec_mag_cnn1 = BatchNormalization()(spec_mag_cnn1)
    spec_mag_cnn1 = Activation('relu')(spec_mag_cnn1)
    spec_mag_cnn1 = MaxPooling2D(pool_size=(1, pool_size[0]))(spec_mag_cnn1)
    spec_mag_cnn1 = Dropout(dropout_rate)(spec_mag_cnn1)
    
    spec_mag_cnn2 = mag2
    spec_mag_cnn2 = Conv2D(filters=nb_cnn2d_filt//8, kernel_size=(3, 3),
                              padding='same', kernel_initializer='uniform')(spec_mag_cnn2)
    spec_mag_cnn2 = BatchNormalization()(spec_mag_cnn2)
    spec_mag_cnn2 = Activation('relu')(spec_mag_cnn2)
    spec_mag_cnn2 = MaxPooling2D(pool_size=(1, pool_size[0]))(spec_mag_cnn2)
    spec_mag_cnn2 = Dropout(dropout_rate)(spec_mag_cnn2)
    
    spec_mag_cnn3 = mag3
    spec_mag_cnn3 = Conv2D(filters=nb_cnn2d_filt//8, kernel_size=(3, 3),
                              padding='same', kernel_initializer='uniform')(spec_mag_cnn3)
    spec_mag_cnn3 = BatchNormalization()(spec_mag_cnn3)
    spec_mag_cnn3 = Activation('relu')(spec_mag_cnn3)
    spec_mag_cnn3 = MaxPooling2D(pool_size=(1, pool_size[0]))(spec_mag_cnn3)
    spec_mag_cnn3 = Dropout(dropout_rate)(spec_mag_cnn3)
    
    spec_mag_cnn4 = mag4
    spec_mag_cnn4 = Conv2D(filters=nb_cnn2d_filt//8, kernel_size=(3, 3),
                              padding='same', kernel_initializer='uniform')(spec_mag_cnn4)
    spec_mag_cnn4 = BatchNormalization()(spec_mag_cnn4)
    spec_mag_cnn4 = Activation('relu')(spec_mag_cnn4)
    spec_mag_cnn4 = MaxPooling2D(pool_size=(1, pool_size[0]))(spec_mag_cnn4)
    spec_mag_cnn4 = Dropout(dropout_rate)(spec_mag_cnn4)

   # PHASE ======================================
    spec_phase_cnn1 = phase1
    spec_phase_cnn1 = Conv2D(filters=nb_cnn2d_filt//8, kernel_size=(3, 3),
                              padding='same', kernel_initializer='uniform')(spec_phase_cnn1)
    spec_phase_cnn1 = BatchNormalization()(spec_phase_cnn1)
    spec_phase_cnn1 = Activation('relu')(spec_phase_cnn1)
    spec_phase_cnn1 = MaxPooling2D(pool_size=(1, pool_size[0]))(spec_phase_cnn1)
    spec_phase_cnn1 = Dropout(dropout_rate)(spec_phase_cnn1)
    
    spec_phase_cnn2 = phase2
    spec_phase_cnn2 = Conv2D(filters=nb_cnn2d_filt//8, kernel_size=(3, 3),
                              padding='same', kernel_initializer='uniform')(spec_phase_cnn2)
    spec_phase_cnn2 = BatchNormalization()(spec_phase_cnn2)
    spec_phase_cnn2 = Activation('relu')(spec_phase_cnn2)
    spec_phase_cnn2 = MaxPooling2D(pool_size=(1, pool_size[0]))(spec_phase_cnn2)
    spec_phase_cnn2 = Dropout(dropout_rate)(spec_phase_cnn2)
    
    spec_phase_cnn3 = phase3
    spec_phase_cnn3 = Conv2D(filters=nb_cnn2d_filt//8, kernel_size=(3, 3),
                              padding='same', kernel_initializer='uniform')(spec_phase_cnn3)
    spec_phase_cnn3 = BatchNormalization()(spec_phase_cnn3)
    spec_phase_cnn3 = Activation('relu')(spec_phase_cnn3)
    spec_phase_cnn3 = MaxPooling2D(pool_size=(1, pool_size[0]))(spec_phase_cnn3)
    spec_phase_cnn3 = Dropout(dropout_rate)(spec_phase_cnn3)
    
    spec_phase_cnn4 = phase4
    spec_phase_cnn4 = Conv2D(filters=nb_cnn2d_filt//8, kernel_size=(3, 3),
                              padding='same', kernel_initializer='uniform')(spec_phase_cnn4)
    spec_phase_cnn4 = BatchNormalization()(spec_phase_cnn4)
    spec_phase_cnn4 = Activation('relu')(spec_phase_cnn4)
    spec_phase_cnn4 = MaxPooling2D(pool_size=(1, pool_size[0]))(spec_phase_cnn4)
    spec_phase_cnn4 = Dropout(dropout_rate)(spec_phase_cnn4)
    
    # MAG 2 step =======================================
    mag12 = Concatenate(axis=1)([spec_mag_cnn1, spec_mag_cnn2])
    mag34 = Concatenate(axis=1)([spec_mag_cnn3, spec_mag_cnn4])
    
    spec_mag_cnn12 = mag12
    spec_mag_cnn12 = Conv2D(filters=nb_cnn2d_filt//4, kernel_size=(3, 3),
                              padding='same', kernel_initializer='uniform')(spec_mag_cnn12)
    spec_mag_cnn12 = BatchNormalization()(spec_mag_cnn12)
    spec_mag_cnn12 = Activation('relu')(spec_mag_cnn12)
    spec_mag_cnn12 = MaxPooling2D(pool_size=(1, pool_size[1]))(spec_mag_cnn12)
    spec_mag_cnn12 = Dropout(dropout_rate)(spec_mag_cnn12)

    spec_mag_cnn34 = mag34
    spec_mag_cnn34 = Conv2D(filters=nb_cnn2d_filt//4, kernel_size=(3, 3),
                              padding='same', kernel_initializer='uniform')(spec_mag_cnn34)
    spec_mag_cnn34 = BatchNormalization()(spec_mag_cnn34)
    spec_mag_cnn34 = Activation('relu')(spec_mag_cnn34)
    spec_mag_cnn34 = MaxPooling2D(pool_size=(1, pool_size[1]))(spec_mag_cnn34)
    spec_mag_cnn34 = Dropout(dropout_rate)(spec_mag_cnn34)
    
    # PHASE 2 step =====================================
    phase12 = Concatenate(axis=1)([spec_phase_cnn1, spec_phase_cnn2])
    phase34 = Concatenate(axis=1)([spec_phase_cnn3, spec_phase_cnn4])
    
    spec_phase_cnn12 = phase12
    spec_phase_cnn12 = Conv2D(filters=nb_cnn2d_filt//4, kernel_size=(3, 3),
                              padding='same', kernel_initializer='uniform')(spec_phase_cnn12)
    spec_phase_cnn12 = BatchNormalization()(spec_phase_cnn12)
    spec_phase_cnn12 = Activation('relu')(spec_phase_cnn12)
    spec_phase_cnn12 = MaxPooling2D(pool_size=(1, pool_size[1]))(spec_phase_cnn12)
    spec_phase_cnn12 = Dropout(dropout_rate)(spec_phase_cnn12)

    spec_phase_cnn34 = phase34
    spec_phase_cnn34 = Conv2D(filters=nb_cnn2d_filt//4, kernel_size=(3, 3),
                              padding='same', kernel_initializer='uniform')(spec_phase_cnn34)
    spec_phase_cnn34 = BatchNormalization()(spec_phase_cnn34)
    spec_phase_cnn34 = Activation('relu')(spec_phase_cnn34)
    spec_phase_cnn34 = MaxPooling2D(pool_size=(1, pool_size[1]))(spec_phase_cnn34)
    spec_phase_cnn34 = Dropout(dropout_rate)(spec_phase_cnn34)
    
    # MAG 3 step =====================
    mag = Concatenate(axis=1)([spec_mag_cnn12, spec_mag_cnn34])

    spec_mag_cnn = mag
    spec_mag_cnn = Conv2D(filters=nb_cnn2d_filt//2, kernel_size=(3, 3),
                              padding='same', kernel_initializer='uniform')(spec_mag_cnn)
    spec_mag_cnn = BatchNormalization()(spec_mag_cnn)
    spec_mag_cnn = Activation('relu')(spec_mag_cnn)
    spec_mag_cnn = MaxPooling2D(pool_size=(1, pool_size[2]))(spec_mag_cnn)
    spec_mag_cnn = Dropout(dropout_rate)(spec_mag_cnn)
    
    # PHASE 3 step =====================    
    phase = Concatenate(axis=1)([spec_phase_cnn12, spec_phase_cnn34])

    spec_phase_cnn = phase
    spec_phase_cnn = Conv2D(filters=nb_cnn2d_filt//2, kernel_size=(3, 3),
                              padding='same', kernel_initializer='uniform')(spec_phase_cnn)
    spec_phase_cnn = BatchNormalization()(spec_phase_cnn)
    spec_phase_cnn = Activation('relu')(spec_phase_cnn)
    spec_phase_cnn = MaxPooling2D(pool_size=(1, pool_size[2]))(spec_phase_cnn)
    spec_phase_cnn = Dropout(dropout_rate)(spec_phase_cnn)
    
    # FUSION ================
    
    concat_feat = Concatenate(axis=1)([spec_mag_cnn, spec_phase_cnn])
    
    concat_feat = Permute((2, 1, 3))(concat_feat)


    # RNN
    spec_rnn = Reshape((data_in[-2], -1))(concat_feat)

    doa = spec_rnn

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

    # FC - SED
    sed = spec_rnn

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

    model = Model(inputs=[ma1, ma2, ma3, ma4, phas1, phas2, phas3, phas4], outputs=[sed, doa])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'logcosh'], loss_weights=weights)

    model.summary()
    return model
