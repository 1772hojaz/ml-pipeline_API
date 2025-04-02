#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight


class LungDiseaseModel:
    def __init__(self):
        self.model = None

    def build_model(self):
        base_model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False  # Freeze pretrained layers
        
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)  # Sigmoid for binary classification
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, train_generator, val_generator, class_weight_dict):
        checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)

        history = self.model.fit(train_generator, validation_data=val_generator, epochs=20,
                                 callbacks=[early_stop, checkpoint, reduce_lr], class_weight=class_weight_dict)
        return history

    def compute_class_weights(self, Y):
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(Y), y=Y)
        class_weight_dict = dict(enumerate(class_weights))
        return class_weight_dict

    def evaluate_model(self, test_generator):
        y_pred = (self.model.predict(test_generator) > 0.5).astype(int).flatten()
        y_true = test_generator.classes
        return y_true, y_pred

