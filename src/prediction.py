#!/usr/bin/env python3

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix


class ModelPredictor:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def make_predictions(self, test_generator):
        y_pred = (self.model.predict(test_generator) > 0.5).astype(int).flatten()
        y_true = test_generator.classes
        return y_true, y_pred

    def display_results(self, y_true, y_pred):
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred, target_names=['Normal', 'Atelectasis']))

