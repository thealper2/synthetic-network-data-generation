import time
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from sdv.evaluation.single_table import evaluate_quality
from sdv.metadata import Metadata
from sdv.single_table import (
    CopulaGANSynthesizer,
    CTGANSynthesizer,
    GaussianCopulaSynthesizer,
    TVAESynthesizer,
)


class SyntheticDataGenerator:
    """Class to generate synthetic data using different methods from the SDV library."""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the SyntheticDataGenerator.

        Args:
            data: Original DataFrame with both features and target
            discrete_columns: List of column names that are discrete/categorical
        """
        self.data = data
        self.metadata = Metadata.detect_from_dataframe(data)
        self.metadata.update_columns(
            column_names=[
                "protocol_type",
                "land",
                "wrong_fragment",
                "urgent",
                "num_failed_logins",
                "logged_in",
                "num_compromised",
                "root_shell",
                "su_attempted",
                "num_root",
                "num_file_creations",
                "num_shells",
                "num_access_files",
                "num_outbound_cmds",
                "is_host_login",
                "is_guest_login",
            ],
            sdtype="numerical",
            table_name="table",
            # computer_representation='Float'
        )

    def generate_gaussian_copula(self) -> pd.DataFrame:
        """
        Generate synthetic data using GaussianCopula.

        Returns:
            DataFrame with synthetic data

        Raises:
            Exception: If error occurs during synthetic data generation
        """
        try:
            print("\nGenerating synthetic data using GaussianCopula...")
            start_time = time.time()

            # Initialize and fit GaussianCopula model
            model = GaussianCopulaSynthesizer(
                metadata=self.metadata,
            )

            model.fit(self.data)

            # Generate synthetic data of the same size as the original
            synthetic_data = model.sample(num_rows=len(self.data))

            # Ensure target column has the same data type
            if "class" in synthetic_data.columns:
                synthetic_data["class"] = synthetic_data["class"].astype(
                    self.data["class"].dtype
                )

            elapsed_time = time.time() - start_time
            print(f"GaussianCopula generation completed in {elapsed_time:.2f} seconds")
            print(f"Generated synthetic data shape: {synthetic_data.shape}")

            # Evaluate the quality of synthetic data
            self._evaluate_synthetic_data(synthetic_data, "GaussianCopula")

            return synthetic_data

        except Exception as e:
            raise Exception(
                f"Error generating synthetic data with GaussianCopula: {str(e)}"
            )

    def generate_ctgan(self, epochs: int = 100) -> pd.DataFrame:
        """
        Generate synthetic data using CTGAN.

        Args:
            epochs: Number of training epochs for CTGAN

        Returns:
            DataFrame with synthetic data

        Raises:
            Exception: If error occurs during synthetic data generation
        """
        try:
            print("\nGenerating synthetic data using CTGAN...")
            start_time = time.time()

            # Initialize and fit CTGAN model
            model = CTGANSynthesizer(
                epochs=epochs,
                metadata=self.metadata,
                verbose=True,
            )

            model.fit(self.data)

            # Generate synthetic data of the same size as the original
            synthetic_data = model.sample(num_rows=len(self.data))

            # Ensure target column has the same data type
            if "class" in synthetic_data.columns:
                synthetic_data["class"] = synthetic_data["class"].astype(
                    self.data["class"].dtype
                )

            elapsed_time = time.time() - start_time
            print(f"CTGAN generation completed in {elapsed_time:.2f} seconds")
            print(f"Generated synthetic data shape: {synthetic_data.shape}")

            # Evaluate the quality of synthetic data
            self._evaluate_synthetic_data(synthetic_data, "CTGAN")

            return synthetic_data

        except Exception as e:
            raise Exception(f"Error generating synthetic data with CTGAN: {str(e)}")

    def generate_copula_gan(self, epochs: int = 100) -> pd.DataFrame:
        """
        Generate synthetic data using Copula GAN.

        Args:
            epochs: Number of training epochs for Copula GAN

        Returns:
            Dataframe with synthetic data

        Raises:
            Exception: If error occurs during synthetic data generation
        """
        try:
            print("\nGenerating synthetic data using CopulaGAN...")
            start_time = time.time()

            # Initialize and fit CopulaGAN model
            model = CopulaGANSynthesizer(
                epochs=epochs, metadata=self.metadata, verbose=True
            )

            model.fit(self.data)

            # Generate synthetic data of the same size as the original
            synthetic_data = model.sample(num_rows=len(self.data))

            # Ensure target column has the same data type
            if "class" in synthetic_data.columns:
                synthetic_data["class"] = synthetic_data["class"].astype(
                    self.data["class"].dtype
                )

            elapsed_time = time.time() - start_time
            print(f"CopulaGAN generation completed in {elapsed_time:.2f} seconds")
            print(f"Generated synthetic data shape: {synthetic_data.shape}")

            # Evaluate the quality of synthetic data
            self._evaluate_synthetic_data(synthetic_data, "CopulaGAN")

            return synthetic_data

        except Exception as e:
            raise Exception(f"Error generating synthetic data with CopulaGAN: {str(e)}")

    def generate_tvae(self, epochs: int = 100) -> pd.DataFrame:
        """
        Generate synthetic data using TVAE.

        Returns:
            DataFrame with synthetic data

        Raises:
            Exception: If error occurs during synthetic data generation
        """
        try:
            print("\nGenerating synthetic data using TVAE...")
            start_time = time.time()

            # Initialize and fit TVAE mode
            model = TVAESynthesizer(epochs=epochs, metadata=self.metadata, verbose=True)

            model.fit(self.data)

            # Generate synthetic data of the same size as the original
            synthetic_data = model.sample(num_rows=len(self.data))

            # Ensure target column has the same data type
            if "class" in synthetic_data.columns:
                synthetic_data["class"] = synthetic_data["class"].astype(
                    self.data["class"].dtype
                )

            elapsed_time = time.time() - start_time
            print(f"TVAE generation completed in {elapsed_time:.2f} seconds")
            print(f"Generated synthetic data shape: {synthetic_data.shape}")

            # Evaluate the quality of synthetic data
            self._evaluate_synthetic_data(synthetic_data, "TVAE")

            return synthetic_data

        except Exception as e:
            raise Exception(f"Error generating synthetic data with CopulaGAN: {str(e)}")
        ß

    def _evaluate_synthetic_data(
        self, synthetic_data: pd.DataFrame, method_name: str
    ) -> None:
        """
        Evaluate the quality of synthetic data.

        Args:
            synthetic_data: Generated synthetic data
            method_name: Name of the method used to generate the data
        """
        try:
            print(f"\nEvaluating quality of {method_name} synthetic data...")

            # Compare data distributions
            quality_report = evaluate_quality(
                real_data=self.data,
                synthetic_data=synthetic_data,
                metadata=self.metadata,
            )

            print(f"{method_name} Quality Score: {quality_report.get_score():.4f}")

            # Class distribution comparison
            if "class" in self.data.columns and "class" in synthetic_data.columns:
                real_class_dist = self.data["class"].value_counts(normalize=True)
                synth_class_dist = synthetic_data["class"].value_counts(normalize=True)

                print("\nClass Distribution (Original vs Synthetic):")
                comparison_df = pd.DataFrame(
                    {
                        "Original": real_class_dist,
                        f"Synthetic ({method_name})": synth_class_dist,
                    }
                )
                print(comparison_df)

                # Plot class distribution comparison
                plt.figure(figsize=(12, 6))
                comparison_df.plot(kind="bar")
                plt.title(
                    f"Class Distribution: Original vs {method_name} Synthetic Data"
                )
                plt.ylabel("Proportion")
                plt.tight_layout()
                plt.savefig(f"class_distribution_{method_name.lower()}.png")
                plt.close()

        except Exception as e:
            print(f"Warning: Error during synthetic data evaluation: {str(e)}")
