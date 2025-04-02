#!/usr/bin/env python3
from preprocessing import ImagePreprocessor
from model import LungDiseaseModel
from prediction import ModelPredictor
from config import DATASET_PATH, BINARY_CLASSES, MODEL_PATH


def main():
    # Data preprocessing
    preprocessor = ImagePreprocessor(DATASET_PATH, BINARY_CLASSES)
    df = preprocessor.load_data()
    train_df, val_df, test_df = preprocessor.split_data(df)

    image_generator = preprocessor.get_image_generator()
    train_generator = preprocessor.create_data_generator(train_df, image_generator)
    val_generator = preprocessor.create_data_generator(val_df, image_generator)
    test_generator = preprocessor.create_data_generator(test_df, image_generator)

    # Model building and training
    model = LungDiseaseModel()
    model.model = model.build_model()

    class_weight_dict = model.compute_class_weights(train_df['Label'])
    model.train_model(train_generator, val_generator, class_weight_dict)

    # Prediction and evaluation
    predictor = ModelPredictor(MODEL_PATH)
    y_true, y_pred = predictor.make_predictions(test_generator)
    predictor.display_results(y_true, y_pred)


if __name__ == "__main__":
    main()

