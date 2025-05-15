import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from config import KDD_DATASET_PATH, RANDOM_SEED
from data_processor import DataProcessor
from model_evaluator import ModelEvaluator
from synthetic_generator import SyntheticDataGenerator


def main() -> None:
    """Main function to execute the pipeline."""
    try:
        print(
            "\n===== Synthetic Tabular Data Generation with Scikit-Learn and SDV =====\n"
        )

        # 1. Load and preprocess data
        data_processor = DataProcessor(file_path=KDD_DATASET_PATH)
        data = data_processor.load_data()
        X, y = data_processor.preprocess_data()

        # Add target back to data for synthetic generation
        data_with_target = X.copy()
        data_with_target["class"] = data_processor.label_encoder.inverse_transform(y)
        label_names = list(data_processor.label_encoder.classes_)

        # 2. Split data into training and testing sets
        X_train, X_test, y_train, y_test = data_processor.get_train_test_split(
            test_size=0.3
        )

        # Prepare data with target for synthetic generation
        train_data_with_target = X_train.copy()
        train_data_with_target["class"] = (
            data_processor.label_encoder.inverse_transform(y_train)
        )

        # 3. Generate synthetic data
        data_generator = SyntheticDataGenerator(train_data_with_target)

        # Generate synthetic data with different methods
        gaussian_copula_data = data_generator.generate_gaussian_copula()
        ctgan_data = data_generator.generate_ctgan(epochs=50)
        copulagan_data = data_generator.generate_copula_gan(epochs=50)
        tvae_data = data_generator.generate_tvae(epochs=50)

        # 4. Prepare datasets for model evaluation
        # Calculate class weights
        classes = np.unique(y)
        class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
        print(f"\nClass weights: {class_weights}")

        # Extract X and y from synthetic data
        gaussian_copula_X = gaussian_copula_data.drop("class", axis=1)
        gaussian_copula_y = data_processor.label_encoder.transform(
            gaussian_copula_data["class"]
        )

        ctgan_X = ctgan_data.drop("class", axis=1)
        ctgan_y = data_processor.label_encoder.transform(ctgan_data["class"])

        copulagan_X = copulagan_data.drop("class", axis=1)
        copulagan_y = data_processor.label_encoder.transform(copulagan_data["class"])

        tvae_X = tvae_data.drop("class", axis=1)
        tvae_y = data_processor.label_encoder.transform(tvae_data["class"])

        # Combine original and synthetic data
        X_train_with_gaussian = pd.concat([X_train, gaussian_copula_X], axis=0)
        y_train_with_gaussian = np.concatenate([y_train, gaussian_copula_y])

        X_train_with_ctgan = pd.concat([X_train, ctgan_X], axis=0)
        y_train_with_ctgan = np.concatenate([y_train, ctgan_y])

        X_train_with_copulagan = pd.concat([X_train, copulagan_X], axis=0)
        y_train_with_copulagan = np.concatenate([y_train, copulagan_y])

        X_train_with_tvae = pd.concat([X_train, tvae_X], axis=0)
        y_train_with_tvae = np.concatenate([y_train, tvae_y])

        # 5. Evaluate models
        evaluator = ModelEvaluator()

        # Evaluate with original data only
        evaluator.train_and_evaluate(
            X_train,
            y_train,
            X_test,
            y_test,
            label_names,
            "Original Data Only",
            class_weights,
        )

        # Evaluate with original + Gaussian Copula data
        evaluator.train_and_evaluate(
            X_train_with_gaussian,
            y_train_with_gaussian,
            X_test,
            y_test,
            label_names,
            "Original + GaussianCopula",
            class_weights,
        )

        # Evaluate with original + CTGAN data
        evaluator.train_and_evaluate(
            X_train_with_ctgan,
            y_train_with_ctgan,
            X_test,
            y_test,
            label_names,
            "Original + CTGAN",
            class_weights,
        )

        evaluator.train_and_evaluate(
            X_train_with_copulagan,
            y_train_with_copulagan,
            X_test,
            y_test,
            label_names,
            "Original + CopulaGAN",
            class_weights,
        )

        evaluator.train_and_evaluate(
            X_train_with_tvae,
            y_train_with_tvae,
            X_test,
            y_test,
            label_names,
            "Original + TVAE",
            class_weights,
        )

        # 6. Create summary table and plots
        evaluator.create_summary_table()
        evaluator.create_summary_plots()

        print("\n===== Pipeline completed successfully =====")

    except Exception as e:
        print(f"\nError in main pipeline: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
