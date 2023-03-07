#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# add Dropout to use with Active Learning

import os
import numpy as np
import tensorflow as tf
from keras.optimizers import *
from keras.callbacks import *
from tensorflow.keras.callbacks import *

import Score
from loss_functions import *
import logging

# ======================
# GLOBALS
# ======================

logger = logging.getLogger(__file__)

s = Semantic_loss_functions()

def dense_block(x, in_planes):
    org = x
    # print(x.shape)
    c1 = tf.keras.layers.Conv2D(in_planes, (1, 1))(x)
    b1 = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(c1)
    x = tf.keras.layers.ReLU()(b1)

    c2 = tf.keras.layers.Conv2D(int(in_planes/4), (3, 3), padding='same')(x)
    b2 = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(c2)
    x = tf.keras.layers.ReLU()(b2)
    d1 = x
    # print(x.shape)
    x = tf.keras.layers.concatenate([org, d1], axis=3)
    c3 = tf.keras.layers.Conv2D(in_planes, (1, 1))(x)
    b3 = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(c3)
    x = tf.keras.layers.ReLU()(b3)
    c4 = tf.keras.layers.Conv2D(int(in_planes/4), (3, 3), padding='same')(x)
    b4 = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(c4)
    x = tf.keras.layers.ReLU()(b4)
    d2= x

    x = tf.keras.layers.concatenate([org, d1, d2], axis=3)
    c5 = tf.keras.layers.Conv2D(in_planes, (1, 1))(x)
    b5 = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(c5)
    x = tf.keras.layers.ReLU()(b5)
    c6 = tf.keras.layers.Conv2D(int(in_planes/4), (3, 3), padding='same')(x)
    b6 = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(c6)
    x = tf.keras.layers.ReLU()(b6)
    d3= x

    x = tf.keras.layers.concatenate([org, d1, d2, d3], axis=3)
    c7 = tf.keras.layers.Conv2D(in_planes, (1, 1))(x)
    b7 = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(c7)
    x = tf.keras.layers.ReLU()(b7)
    c8 = tf.keras.layers.Conv2D(int(in_planes/4), (3, 3), padding='same')(x)
    b8 = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(c8)
    x = tf.keras.layers.ReLU()(b8)
    d4= x

    x = tf.keras.layers.concatenate([d1, d2, d3, d4], axis=3)
    x = tf.keras.layers.add([org, x])

    return x


class KiUnet(object):

    def __init__(self, channels, features_root=8,
                 filter_size=3, pool_size=2, nb_classes=2, dense=False):
        """
        Initialization of U-Net network
        @method "__init__"
        @param {integer} channels Number of features in input image
        @param {integer} features_root Number of filters
        @param {integer} filter_size Size of filters
        @param {integer} pool_size Size of pooling
        """
        self.channels = channels
        self.nb_classes = nb_classes
        # self.layers = layers
        self.features_root = features_root
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.dense = dense

    def instanciation(self, tile_size):
        """
        Instancitation of U-Net network
        @method "instanciation"
        @param {integer} tile_size Size of treatment tile
        @return List of raised errors
        """
        errors = []

        try:
            if not isinstance(self.channels, int):
                raise TypeError
        except TypeError as e:
            errors += ["CHANNELS_NOT_INTEGER"]
            logger.error(e)
            return errors

        try:
            if self.channels < 0:
                raise ValueError
        except ValueError as e:
            errors += ["CHANNELS_NEGATIVE"]
            logger.error(e)
            return errors

        try:
            if not isinstance(self.features_root, int):
                raise TypeError
        except TypeError as e:
            errors += ["FEATURESROOT_NOT_INTEGER"]
            logger.error(e)
            return errors

        try:
            if self.features_root < 0:
                raise ValueError
        except ValueError as e:
            errors += ["FEATURESROOT_NEGATIVE"]
            logger.error(e)
            return errors
        try:
            if not isinstance(self.nb_classes, int):
                raise TypeError
        except TypeError as e:
            errors += ["NB_CLASSES_NOT_INTEGER"]
            logger.error(e)
            return errors

        try:
            if self.nb_classes < 0:
                raise ValueError
        except ValueError as e:
            errors += ["NB_CLASSES_NEGATIVE"]
            logger.error(e)
            return errors

        try:
            if not isinstance(self.filter_size, int):
                raise TypeError
        except TypeError as e:
            errors += ["FILTERSIZE_NOT_INTEGER"]
            logger.error(e)
            return errors

        try:
            if self.filter_size < 0:
                raise ValueError
        except ValueError as e:
            errors += ["FILTERSIZE_NEGATIVE"]
            logger.error(e)
            return errors

        try:
            if not isinstance(self.pool_size, int):
                raise TypeError
        except TypeError as e:
            errors += ["POOLSIZE_NOT_INTEGER"]
            logger.error(e)
            return errors

        try:
            if self.pool_size < 0:
                raise ValueError
        except ValueError as e:
            errors += ["POOLSIZE_NEGATIVE"]
            logger.error(e)
            return errors

        try:
            # Build the model
            inputs = tf.keras.layers.Input((tile_size, tile_size, self.channels))
            s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

            #filter_size=3
            #features_root=8
            #pool_size=2

            # Contraction path

            ### Encoder

            #U-Net branch block 1
            encoder1 = tf.keras.layers.Conv2D(self.features_root * 2,
                                        (self.filter_size, self.filter_size),
                                        kernel_initializer='he_uniform',
                                        padding='same')(s)
            if self.dense==True:
                encoder1 = dense_block(encoder1, 16)
            encoder1 = tf.keras.layers.MaxPooling2D((self.pool_size, self.pool_size))(encoder1)
            en1_bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(encoder1)
            out = tf.keras.layers.ReLU()(en1_bn)

            #Ki-Net branch block 1
            encoderf1 = tf.keras.layers.Conv2D(self.features_root * 2,
                                        (self.filter_size, self.filter_size),
                                        kernel_initializer='he_uniform',
                                        padding='same')(s)
            if self.dense==True:
                encoderf1 = dense_block(encoderf1, 16)
            encoderf1 = tf.keras.layers.UpSampling2D(size=(self.pool_size, self.pool_size), interpolation='bilinear')(encoderf1)
            enf1_bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(encoderf1)
            out1 = tf.keras.layers.ReLU()(enf1_bn)

            #Cross residual feature block (CRFB) block 1
            tmp = out
            intere1_1 = tf.keras.layers.Conv2D(self.features_root * 2,
                                        (self.filter_size, self.filter_size),
                                        kernel_initializer='he_uniform',
                                        padding='same')(out1)
            inte1_1bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(intere1_1)
            inte1_1bn = tf.keras.layers.ReLU()(inte1_1bn)
            inte1_1bn = tf.keras.layers.MaxPooling2D((self.pool_size*2 , self.pool_size*2))(inte1_1bn)
            out = tf.keras.layers.add([out, inte1_1bn])

            intere1_2 = tf.keras.layers.Conv2D(self.features_root * 2,
                                        (self.filter_size, self.filter_size),
                                        kernel_initializer='he_uniform',
                                        padding='same')(tmp)
            inte1_2bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(intere1_2)
            inte1_2bn = tf.keras.layers.ReLU()(inte1_2bn)
            inte1_2bn = tf.keras.layers.UpSampling2D(size=(self.pool_size*2, self.pool_size*2), interpolation='bilinear')(inte1_2bn)
            out1 = tf.keras.layers.add([out1, inte1_2bn])
        
            u1 = out  #skip conn
            o1 = out1  #skip conn


            #U-Net branch block 2
            encoder2 = tf.keras.layers.Conv2D(self.features_root * 4,
                                        (self.filter_size, self.filter_size),
                                        kernel_initializer='he_uniform',
                                        padding='same')(out)
            if self.dense==True:
                encoder2 = dense_block(encoder2, 32)
            encoder2 = tf.keras.layers.MaxPooling2D((self.pool_size, self.pool_size))(encoder2)
            en2_bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(encoder2)
            out = tf.keras.layers.ReLU()(en2_bn)

            #Ki-Net branch block 2
            encoderf2 = tf.keras.layers.Conv2D(self.features_root * 4,
                                        (self.filter_size, self.filter_size),
                                        kernel_initializer='he_uniform',
                                        padding='same')(out1)
            if self.dense==True:
                encoderf2 = dense_block(encoderf2, 32)
            encoderf2 = tf.keras.layers.UpSampling2D(size=(self.pool_size, self.pool_size), interpolation='bilinear')(encoderf2)
            enf2_bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(encoderf2)
            out1 = tf.keras.layers.ReLU()(enf2_bn)

            #Cross residual feature block (CRFB) block 2
            tmp = out
            intere2_1 = tf.keras.layers.Conv2D(self.features_root * 4,
                                        (self.filter_size, self.filter_size),
                                        kernel_initializer='he_uniform',
                                        padding='same')(out1)
            inte2_1bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(intere2_1)
            inte2_1bn = tf.keras.layers.ReLU()(inte2_1bn)
            inte2_1bn = tf.keras.layers.MaxPooling2D((self.pool_size*8, self.pool_size*8))(inte2_1bn)
            out = tf.keras.layers.add([out, inte2_1bn])

            intere2_2 = tf.keras.layers.Conv2D(self.features_root * 4,
                                        (self.filter_size, self.filter_size),
                                        kernel_initializer='he_uniform',
                                        padding='same')(tmp)
            inte2_2bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(intere2_2)
            inte2_2bn = tf.keras.layers.ReLU()(inte2_2bn)
            inte2_2bn = tf.keras.layers.UpSampling2D(size=(self.pool_size*8, self.pool_size*8), interpolation='bilinear')(inte2_2bn)
            out1 = tf.keras.layers.add([out1, inte2_2bn])
        
            u2 = out  #skip conn
            o2 = out1  #skip conn

            #U-Net branch block 3
            encoder3 = tf.keras.layers.Conv2D(self.features_root * 8,
                                        (self.filter_size, self.filter_size),
                                        kernel_initializer='he_uniform',
                                        padding='same')(out)
            if self.dense==True:
                encoder3 = dense_block(encoder3, 64)
            encoder3 = tf.keras.layers.MaxPooling2D((self.pool_size, self.pool_size))(encoder3)
            en3_bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(encoder3)
            out = tf.keras.layers.ReLU()(en3_bn)

            #Ki-Net branch block 3
            encoderf3 = tf.keras.layers.Conv2D(self.features_root * 8,
                                        (self.filter_size, self.filter_size),
                                        kernel_initializer='he_uniform',
                                        padding='same')(out1)
            if self.dense==True:
                encoderf3 = dense_block(encoderf3, 64)
            encoderf3 = tf.keras.layers.UpSampling2D(size=(self.pool_size, self.pool_size), interpolation='bilinear')(encoderf3)
            enf3_bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(encoderf3)
            out1 = tf.keras.layers.ReLU()(enf3_bn)

            #Cross residual feature block (CRFB) block 3
            tmp = out
            intere3_1 = tf.keras.layers.Conv2D(self.features_root * 8,
                                        (self.filter_size, self.filter_size),
                                        kernel_initializer='he_uniform',
                                        padding='same')(out1)
            inte3_1bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(intere3_1)
            inte3_1bn = tf.keras.layers.ReLU()(inte3_1bn)
            inte3_1bn = tf.keras.layers.MaxPooling2D((self.pool_size*32, self.pool_size*32))(inte3_1bn)
            out = tf.keras.layers.add([out, inte3_1bn])

            intere3_2 = tf.keras.layers.Conv2D(self.features_root * 8,
                                        (self.filter_size, self.filter_size),
                                        kernel_initializer='he_uniform',
                                        padding='same')(tmp)
            inte3_2bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(intere3_2)
            inte3_2bn = tf.keras.layers.ReLU()(inte3_2bn)
            inte3_2bn = tf.keras.layers.UpSampling2D(size=(self.pool_size*32, self.pool_size*32), interpolation='bilinear')(inte3_2bn)
            out1 = tf.keras.layers.add([out1, inte3_2bn])
        
            ### End of encoder block

            ### Start Decoder

            #U-Net branch block 4
            decoder1 = tf.keras.layers.Conv2D(self.features_root * 4,
                                        (self.filter_size, self.filter_size),
                                        kernel_initializer='he_uniform',
                                        padding='same')(out)
            if self.dense==True:
                decoder1 = dense_block(decoder1, 32)
            decoder1 = tf.keras.layers.UpSampling2D(size=(self.pool_size, self.pool_size), interpolation='bilinear')(decoder1)
            de1_bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(decoder1)
            out = tf.keras.layers.ReLU()(de1_bn)

            #Ki-Net branch block 4
            decoderf1 = tf.keras.layers.Conv2D(self.features_root * 4,
                                        (self.filter_size, self.filter_size),
                                        kernel_initializer='he_uniform',
                                        padding='same')(out1)
            if self.dense==True:
                decoderf1 = dense_block(decoderf1, 32)
            decoderf1 = tf.keras.layers.MaxPooling2D((self.pool_size, self.pool_size))(decoderf1)
            def1_bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(decoderf1)
            out1 = tf.keras.layers.ReLU()(def1_bn)

            #Cross residual feature block (CRFB) block 4
            tmp = out
            interd1_1 = tf.keras.layers.Conv2D(self.features_root * 4,
                                        (self.filter_size, self.filter_size),
                                        kernel_initializer='he_uniform',
                                        padding='same')(out1)
            intd1_1bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(interd1_1)
            intd1_1bn = tf.keras.layers.ReLU()(intd1_1bn)
            intd1_1bn = tf.keras.layers.MaxPooling2D((self.pool_size*8 , self.pool_size*8))(intd1_1bn)
            out = tf.keras.layers.add([out, intd1_1bn])

            intere1_2 = tf.keras.layers.Conv2D(self.features_root * 4,
                                        (self.filter_size, self.filter_size),
                                        kernel_initializer='he_uniform',
                                        padding='same')(tmp)
            inte1_2bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(intere1_2)
            inte1_2bn = tf.keras.layers.ReLU()(inte1_2bn)
            inte1_2bn = tf.keras.layers.UpSampling2D(size=(self.pool_size*8, self.pool_size*8), interpolation='bilinear')(inte1_2bn)
            out1 = tf.keras.layers.add([out1, inte1_2bn])

            out = tf.keras.layers.add([out, u2])  #skip conn
            out1 = tf.keras.layers.add([out1, o2])  #skip conn

            #U-Net branch block 5
            decoder2 = tf.keras.layers.Conv2D(self.features_root * 2,
                                        (self.filter_size, self.filter_size),
                                        kernel_initializer='he_uniform',
                                        padding='same')(out)
            if self.dense==True:
                decoder2 = dense_block(decoder2, 16)
            decoder2 = tf.keras.layers.UpSampling2D(size=(self.pool_size, self.pool_size), interpolation='bilinear')(decoder2)
            de2_bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(decoder2)
            out = tf.keras.layers.ReLU()(de2_bn)

            #Ki-Net branch block 5
            decoderf2 = tf.keras.layers.Conv2D(self.features_root * 2,
                                        (self.filter_size, self.filter_size),
                                        kernel_initializer='he_uniform',
                                        padding='same')(out1)
            if self.dense==True:
                decoderf2 = dense_block(decoderf2, 16)
            decoderf2 = tf.keras.layers.MaxPooling2D((self.pool_size, self.pool_size))(decoderf2)
            def2_bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(decoderf2)
            out1 = tf.keras.layers.ReLU()(def2_bn)

            #Cross residual feature block (CRFB) block 5
            tmp = out
            interd2_1 = tf.keras.layers.Conv2D(self.features_root * 2,
                                        (self.filter_size, self.filter_size),
                                        kernel_initializer='he_uniform',
                                        padding='same')(out1)
            intd2_1bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(interd2_1)
            intd2_1bn = tf.keras.layers.ReLU()(intd2_1bn)
            intd2_1bn = tf.keras.layers.MaxPooling2D((self.pool_size*2 , self.pool_size*2))(intd2_1bn)
            out = tf.keras.layers.add([out, intd2_1bn])

            intere2_2 = tf.keras.layers.Conv2D(self.features_root * 2,
                                        (self.filter_size, self.filter_size),
                                        kernel_initializer='he_uniform',
                                        padding='same')(tmp)
            inte2_2bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(intere2_2)
            inte2_2bn = tf.keras.layers.ReLU()(inte2_2bn)
            inte2_2bn = tf.keras.layers.UpSampling2D(size=(self.pool_size*2, self.pool_size*2), interpolation='bilinear')(inte2_2bn)
            out1 = tf.keras.layers.add([out1, inte2_2bn])

            out = tf.keras.layers.add([out, u1])  #skip conn
            out1 = tf.keras.layers.add([out1, o1])  #skip conn

            #U-Net branch block 6
            decoder3 = tf.keras.layers.Conv2D(self.features_root,
                                        (self.filter_size, self.filter_size),
                                        kernel_initializer='he_uniform',
                                        padding='same')(out)
            if self.dense==True:
                decoder3 = dense_block(decoder3, 8)
            decoder3 = tf.keras.layers.UpSampling2D(size=(self.pool_size, self.pool_size), interpolation='bilinear')(decoder3)
            de3_bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(decoder3)
            out = tf.keras.layers.ReLU()(de3_bn)

            #Ki-Net branch block 6
            decoderf3 = tf.keras.layers.Conv2D(self.features_root,
                                        (self.filter_size, self.filter_size),
                                        kernel_initializer='he_uniform',
                                        padding='same')(out1)
            if self.dense==True:
                decoderf3 = dense_block(decoderf3, 8)
            decoderf3 = tf.keras.layers.MaxPooling2D((self.pool_size, self.pool_size))(decoderf3)
            def3_bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(decoderf3)
            out1 = tf.keras.layers.ReLU()(def3_bn)


            ### Fusion
            out = tf.keras.layers.add([out, out1])  

            # out = tf.keras.layers.Conv2D(2, (1, 1))(out)

            # outputs = tf.keras.layers.ReLU()(out) 
            outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(out)


            # if self.nb_classes == 2:
            #     outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(out)
            #     print(outputs.shape)
            # else:
            #     outputs = tf.keras.layers.Conv2D(self.nb_classes, (1, 1), activation='softmax')(out)
        except Exception as e:
            errors.append("INSTANCIATION_ERROR")
            logger.error(e)

        self.model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        return errors

    def train(self, checkpoint_path, cost, X_train, Y_train, X_val, Y_val,
              validation_percentage=0.2, epochs=50, batch_size=16):
        """
        Training of U-Net network
        @method "train"
        @param {string} checkpoint_path Model path
        @param {string} cost Type of cost function
        @param {numpy array} X_train Array of raw images
        @param {numpy array} Y_train Array of seg images
        @param {float} validation_percentage Base percentage for testing
        @param {integer} epochs Number of epochs
        @param {integer} batch_size Size of batch
        @return Trained model
        @return Best F-Score
        """
        # cost=tf.keras.optimizers.Adam(learning_rate=0.0001)
        if self.nb_classes == 2:
            self.model.compile(optimizer=cost,
                               loss=s.focal_tversky,
                               metrics=[Score.dice_coef])
        else:
            self.model.compile(optimizer=cost,
                               loss="categorical_crossentropy",
                               metrics=["categorical_accuracy",
                                        Score.dice_coef])

        self.model.summary()

        # Modelcheckpoint
        checkpoint_dir = os.path.dirname(checkpoint_path)

        learning_rate_reduction = ReduceLROnPlateau(monitor='val_dice_coef',
                                            patience=10,
                                            verbose=1,
                                            factor=0.7,
                                            min_lr=0.0000001)

        if self.nb_classes == 2:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             save_best_only=True,
                                                             monitor="val_dice_coef",
                                                             mode="max",
                                                             verbose=1)
        else:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             save_best_only=True,
                                                             monitor="val_categorical_accuracy",
                                                             mode="max",
                                                             verbose=1)
        results = self.model.fit(X_train,
                                 Y_train,
                                 validation_data=(X_val,Y_val),
                                 batch_size=batch_size,
                                 epochs=epochs, callbacks=[cp_callback, learning_rate_reduction])

        history = History()
        if self.nb_classes == 2:
            Metric = np.max(results.history['val_dice_coef'])
        else:
            Metric = np.max(results.history['val_categorical_accuracy'])

        return self.model, Metric

    def predict(self, checkpoint_path, sample2treat):
        """
        Prediction function of U-Net network
        @method "predict"
        @param {string} checkpoint_path Model path
        @param {numpy array} sample2treat Image to predict
        @return Prediction
        """
        self.model.load_weights(checkpoint_path)
        pred = self.model.predict(sample2treat, verbose=1)

        return pred
