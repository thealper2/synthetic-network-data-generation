import time
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from config import RANDOM_SEED


class ModelEvaluator:
    """Class to evaluate machine learning models on different datasets."""

    def __init__(self):
        """Initialize the ModelEvaluator."""
        self.models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1
            ),
            "Naive Bayes": GaussianNB(),
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1
            ),
        }
        self.results = {}

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        label_names: np.ndarray,
        experiment_name: str,
        class_weights: Optional[Dict] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Train and evaluate models on the given datasets.

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            experiment_name: Name of the experiment (e.g., 'Original', 'GaussianCopula')
            class_weights: Optional dictionary of class weights

        Returns:
            Dictionary with evaluation metrics for each model
        """
        print(f"\n{'=' * 50}")
        print(f"Evaluating models for experiment: {experiment_name}")
        print(f"{'=' * 50}")

        experiment_results = {}

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # For each model
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            start_time = time.time()

            try:
                # Set class weights if provided
                if class_weights is not None and hasattr(model, "class_weight"):
                    model.set_params(class_weight="balanced")

                # Train the model
                model.fit(X_train_scaled, y_train)

                # Make predictions
                y_pred = model.predict(X_test_scaled)

                # Calculate metrics
                metrics = {
                    "F1 Score (macro)": f1_score(y_test, y_pred, average="macro"),
                    "Precision (macro)": precision_score(
                        y_test, y_pred, average="macro"
                    ),
                    "Recall (macro)": recall_score(y_test, y_pred, average="macro"),
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Confusion Matrix": confusion_matrix(y_test, y_pred),
                }

                elapsed_time = time.time() - start_time
                metrics["Training Time (s)"] = elapsed_time

                # Store results
                experiment_results[model_name] = metrics

                print(f"{model_name} trained in {elapsed_time:.2f} seconds")
                print(f"F1 Score (macro): {metrics['F1 Score (macro)']:.4f}")
                print(f"Precision (macro): {metrics['Precision (macro)']:.4f}")
                print(f"Recall (macro): {metrics['Recall (macro)']:.4f}")
                print(f"Accuracy: {metrics['Accuracy']:.4f}")
                print(f"Confusion Matrix:\n{metrics['Confusion Matrix']}")

                # Plot confusion matrix
                self._plot_confusion_matrix(
                    metrics["Confusion Matrix"],
                    label_names,
                    model_name,
                    experiment_name,
                )

            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                experiment_results[model_name] = {
                    "F1 Score (macro)": float("nan"),
                    "Precision (macro)": float("nan"),
                    "Recall (macro)": float("nan"),
                    "Accuracy": float("nan"),
                    "Confusion Matrix": None,
                    "Training Time (s)": float("nan"),
                    "Error": str(e),
                }

        # Store experiment results
        self.results[experiment_name] = experiment_results
        return experiment_results

    def _plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        model_name: str,
        experiment_name: str,
    ) -> None:
        """
        Plot confusion matrix for a model.

        Args:
            cm: Confusion matrix
            class_names: Names of the classes
            model_name: Name of the model
            experiment_name: Name of the experiment
        """
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, values_format="d")
        plt.title(f"{model_name} - {experiment_name}")
        plt.tight_layout()
        plt.xticks(rotation=90)
        plt.savefig(
            f"confusion_matrix_{model_name.lower().replace(' ', '_')}_{experiment_name.lower().replace(' ', '_')}.png"
        )
        plt.close()

    def create_summary_table(self) -> pd.DataFrame:
        """
        Create a summary table of all model evaluations.

        Returns:
            DataFrame containing the evaluation metrics for all models and experiments
        """
        print("\nCreating summary table...")

        # Initialize lists to store the data
        rows = []

        # Iterate through all experiments and models
        for experiment_name, experiment_results in self.results.items():
            for model_name, metrics in experiment_results.items():
                row = {
                    "Experiment": experiment_name,
                    "Model": model_name,
                    "F1 Score (macro)": metrics.get("F1 Score (macro)", float("nan")),
                    "Precision (macro)": metrics.get("Precision (macro)", float("nan")),
                    "Recall (macro)": metrics.get("Recall (macro)", float("nan")),
                    "Accuracy": metrics.get("Accuracy", float("nan")),
                    "Training Time (s)": metrics.get("Training Time (s)", float("nan")),
                }
                rows.append(row)

        # Create DataFrame
        summary_df = pd.DataFrame(rows)

        # Sort by experiment and model name
        summary_df = summary_df.sort_values(["Experiment", "Model"])

        print("\nResults Summary:")
        print(summary_df)

        # Save summary table to CSV
        summary_df.to_csv("model_results_summary.csv", index=False)
        print("Summary table saved to 'model_results_summary.csv'")

        return summary_df

    def create_summary_plots(self) -> None:
        """Create summary plots comparing all models and experiments."""
        print("\nCreating summary plots...")

        # Get the summary table
        summary_df = self.create_summary_table()

        # Plot metrics comparison
        metrics = [
            "F1 Score (macro)",
            "Precision (macro)",
            "Recall (macro)",
            "Accuracy",
        ]

        # Plot each metric
        for metric in metrics:
            plt.figure(figsize=(14, 8))

            # Reshape data for grouped bar plot
            pivot_df = summary_df.pivot(
                index="Model", columns="Experiment", values=metric
            )

            # Plot grouped bar chart
            ax = pivot_df.plot(kind="bar", figsize=(14, 8))
            plt.title(f"Comparison of {metric} Across Experiments")
            plt.ylabel(metric)
            plt.xlabel("Model")
            plt.legend(title="Experiment")
            plt.grid(axis="y", linestyle="--", alpha=0.7)

            # Add value labels on top of bars
            for container in ax.containers:
                ax.bar_label(container, fmt="%.3f", fontsize=8)

            plt.tight_layout()
            plt.savefig(f"comparison_{metric.lower().replace(' ', '_')}.png")
            plt.close()

        # Plot training time comparison
        plt.figure(figsize=(14, 8))
        pivot_df = summary_df.pivot(
            index="Model", columns="Experiment", values="Training Time (s)"
        )
        ax = pivot_df.plot(kind="bar", figsize=(14, 8))
        plt.title("Comparison of Training Time Across Experiments")
        plt.ylabel("Training Time (seconds)")
        plt.xlabel("Model")
        plt.legend(title="Experiment")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Add value labels on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8)

        plt.tight_layout()
        plt.savefig("comparison_training_time.png")
        plt.close()

        print("Summary plots created successfully")
