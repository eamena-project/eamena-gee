#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
from keras.callbacks import History

import Score
from loss_functions import *

import logging

# ======================
# GLOBALS
# ======================

logger = logging.getLogger(__file__)

s = Semantic_loss_functions()

class Unet(object):

    def __init__(self, channels, features_root=16,
                 filter_size=3, pool_size=2, nb_classes=2):
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

    def instanciation(self, tile_size, training=None):
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

            # Contraction path
            c1 = tf.keras.layers.Conv2D(self.features_root,
                                        (self.filter_size, self.filter_size),
                                        activation='relu',
                                        kernel_initializer='he_normal',
                                        padding='same')(s)
            c1 = tf.keras.layers.Dropout(0.1)(c1, training=training)
            c1 = tf.keras.layers.Conv2D(self.features_root,
                                        (self.filter_size, self.filter_size),
                                        activation='relu',
                                        kernel_initializer='he_normal',
                                        padding='same')(c1)
            p1 = tf.keras.layers.MaxPooling2D((self.pool_size, self.pool_size))(c1)

            c2 = tf.keras.layers.Conv2D(self.features_root * 2,
                                        (self.filter_size, self.filter_size),
                                        activation='relu',
                                        kernel_initializer='he_normal',
                                        padding='same')(p1)
            c2 = tf.keras.layers.Dropout(0.1)(c2, training=training)
            c2 = tf.keras.layers.Conv2D(self.features_root * 2,
                                        (self.filter_size, self.filter_size),
                                        activation='relu',
                                        kernel_initializer='he_normal',
                                        padding='same')(c2)
            p2 = tf.keras.layers.MaxPooling2D((self.pool_size, self.pool_size))(c2)

            c3 = tf.keras.layers.Conv2D(self.features_root * 4,
                                        (self.filter_size, self.filter_size),
                                        activation='relu',
                                        kernel_initializer='he_normal',
                                        padding='same')(p2)
            c3 = tf.keras.layers.Dropout(0.2)(c3, training=training)
            c3 = tf.keras.layers.Conv2D(self.features_root * 4,
                                        (self.filter_size, self.filter_size),
                                        activation='relu',
                                        kernel_initializer='he_normal',
                                        padding='same')(c3)
            p3 = tf.keras.layers.MaxPooling2D((self.pool_size, self.pool_size))(c3)

            c4 = tf.keras.layers.Conv2D(self.features_root * 8,
                                        (self.filter_size, self.filter_size),
                                        activation='relu',
                                        kernel_initializer='he_normal',
                                        padding='same')(p3)
            c4 = tf.keras.layers.Dropout(0.2)(c4, training=training)
            c4 = tf.keras.layers.Conv2D(self.features_root * 8,
                                        (self.filter_size, self.filter_size),
                                        activation='relu',
                                        kernel_initializer='he_normal',
                                        padding='same')(c4)
            p4 = tf.keras.layers.MaxPooling2D((self.pool_size, self.pool_size))(c4)

            c5 = tf.keras.layers.Conv2D(self.features_root * 16,
                                        (self.filter_size, self.filter_size),
                                        activation='relu',
                                        kernel_initializer='he_normal',
                                        padding='same')(p4)
            c5 = tf.keras.layers.Dropout(0.3)(c5, training=training)
            c5 = tf.keras.layers.Conv2D(self.features_root * 16,
                                        (self.filter_size, self.filter_size),
                                        activation='relu',
                                        kernel_initializer='he_normal',
                                        padding='same')(c5)

            # Expansive path
            u6 = tf.keras.layers.Conv2DTranspose(self.features_root * 8,
                                                (self.pool_size, self.pool_size),
                                                strides=(self.pool_size, self.pool_size),
                                                padding='same')(c5)
            u6 = tf.keras.layers.concatenate([u6, c4])
            c6 = tf.keras.layers.Conv2D(self.features_root * 8,
                                        (self.filter_size, self.filter_size),
                                        activation='relu',
                                        kernel_initializer='he_normal',
                                        padding='same')(u6)
            c6 = tf.keras.layers.Dropout(0.2)(c6, training=training)
            c6 = tf.keras.layers.Conv2D(self.features_root * 8,
                                        (self.filter_size, self.filter_size),
                                        activation='relu',
                                        kernel_initializer='he_normal',
                                        padding='same')(c6)

            u7 = tf.keras.layers.Conv2DTranspose(self.features_root * 4,
                                                (self.pool_size, self.pool_size),
                                                strides=(self.pool_size, self.pool_size),
                                                padding='same')(c6)
            u7 = tf.keras.layers.concatenate([u7, c3])
            c7 = tf.keras.layers.Conv2D(self.features_root * 4,
                                        (self.filter_size, self.filter_size),
                                        activation='relu',
                                        kernel_initializer='he_normal',
                                        padding='same')(u7)
            c7 = tf.keras.layers.Dropout(0.2)(c7, training=training)
            c7 = tf.keras.layers.Conv2D(self.features_root * 4,
                                        (self.filter_size, self.filter_size),
                                        activation='relu',
                                        kernel_initializer='he_normal',
                                        padding='same')(c7)

            u8 = tf.keras.layers.Conv2DTranspose(self.features_root * 2,
                                                (self.pool_size, self.pool_size),
                                                strides=(self.pool_size, self.pool_size),
                                                padding='same')(c7)
            u8 = tf.keras.layers.concatenate([u8, c2])
            c8 = tf.keras.layers.Conv2D(self.features_root * 2,
                                        (self.filter_size, self.filter_size),
                                        activation='relu',
                                        kernel_initializer='he_normal',
                                        padding='same')(u8)
            c8 = tf.keras.layers.Dropout(0.1)(c8, training=training)
            c8 = tf.keras.layers.Conv2D(self.features_root * 2,
                                        (self.filter_size, self.filter_size),
                                        activation='relu',
                                        kernel_initializer='he_normal',
                                        padding='same')(c8)

            u9 = tf.keras.layers.Conv2DTranspose(self.features_root,
                                                (self.pool_size, self.pool_size),
                                                strides=(self.pool_size, self.pool_size),
                                                padding='same')(c8)
            u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
            c9 = tf.keras.layers.Conv2D(self.features_root,
                                        (self.filter_size, self.filter_size),
                                        activation='relu',
                                        kernel_initializer='he_normal',
                                        padding='same')(u9)
            c9 = tf.keras.layers.Dropout(0.1)(c9, training=training)
            c9 = tf.keras.layers.Conv2D(self.features_root,
                                        (self.filter_size, self.filter_size),
                                        activation='relu',
                                        kernel_initializer='he_normal',
                                        padding='same')(c9)

            if self.nb_classes == 2:
                outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
            else:
                outputs = tf.keras.layers.Conv2D(self.nb_classes, (1, 1), activation='softmax')(c9)
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
        # cost=tf.keras.optimizers.Adam(lr=0.0001)
        if self.nb_classes == 2:
            self.model.compile(optimizer=cost,
                               loss='binary_crossentropy',
                            #    loss=s.bce_dice_loss,
                               metrics=[Score.dice_coef])
        else:
            self.model.compile(optimizer=cost,
                               loss="categorical_crossentropy",
                               metrics=["categorical_accuracy",
                                        Score.dice_coef])

        # self.model.summary()

        # Modelcheckpoint
        checkpoint_dir = os.path.dirname(checkpoint_path)

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
                                 epochs=epochs, callbacks=[cp_callback])

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
        # self.model.load_weights(checkpoint_path).expect_partial()
        pred = self.model.predict(sample2treat, verbose=0)

        return pred
