from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import KDD_COLUMN_NAMES, RANDOM_SEED


class DataProcessor:
    """Class to handle loading and preprocessing of the KDD Cup '99 dataset."""

    def __init__(self, file_path: str):
        """
        Initialize the DataProcessor with the file path to the KDD Cup '99 dataset.

        Args:
            file_path: Path to the KDD Cup '99 dataset CSV file
        """
        self.file_path = file_path
        self.data = None
        self.X = None
        self.y = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.categorical_features = None
        self.numerical_features = None

    def load_data(self) -> pd.DataFrame:
        """
        Load the KDD Cup '99 dataset.

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If the dataset file is not found
            Exception: For any other error during data loading
        """
        try:
            # KDD Cup '99 column names
            column_names = [
                "duration",
                "protocol_type",
                "service",
                "flag",
                "src_bytes",
                "dst_bytes",
                "land",
                "wrong_fragment",
                "urgent",
                "hot",
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
                "count",
                "srv_count",
                "serror_rate",
                "srv_serror_rate",
                "rerror_rate",
                "srv_rerror_rate",
                "same_srv_rate",
                "diff_srv_rate",
                "srv_diff_host_rate",
                "dst_host_count",
                "dst_host_srv_count",
                "dst_host_same_srv_rate",
                "dst_host_diff_srv_rate",
                "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate",
                "dst_host_serror_rate",
                "dst_host_srv_serror_rate",
                "dst_host_rerror_rate",
                "dst_host_srv_rerror_rate",
                "class",
            ]

            print(f"Loading KDD Cup '99 dataset from {self.file_path}...")
            # Load only a subset of data for demonstration purposes
            # Remove the nrows parameter for full data processing
            self.data = pd.read_csv(self.file_path, header=None, names=column_names)
            value_counts = self.data["class"].value_counts()
            rare_classes = value_counts[value_counts == 1].index.tolist()

            for class_name in rare_classes:
                sample = self.data[self.data["class"]].sample(1, replace=True)
                self.data = pd.concat([self.data, sample], ignore_index=True)

            print(f"Dataset loaded with shape: {self.data.shape}")
            return self.data

        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at {self.file_path}")
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")

    def preprocess_data(
        self, sample_size: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess the KDD Cup '99 dataset.

        Args:
            sample_size: Optional size to sample from the dataset for faster processing

        Returns:
            Tuple containing features (X) and target (y)

        Raises:
            ValueError: If the data hasn't been loaded yet
        """
        if self.data is None:
            raise ValueError("Data must be loaded before preprocessing")

        print("Preprocessing data...")

        # Sample data if specified
        if sample_size is not None and sample_size < len(self.data):
            self.data = self.data.sample(sample_size, random_state=RANDOM_SEED)
            print(f"Sampled {sample_size} records for faster processing")

        # Identify categorical and numerical features
        self.feature_names = self.data.columns.tolist()[
            :-1
        ]  # All columns except the target
        self.categorical_features = self.data.select_dtypes(
            include=["object", "category", "boolean"]
        ).columns.tolist()
        self.numerical_features = [
            col for col in self.feature_names if col not in self.categorical_features
        ]

        print(f"Categorical features: {self.categorical_features}")
        print(f"Number of numerical features: {len(self.numerical_features)}")

        # Encode the target class
        self.y = self.label_encoder.fit_transform(self.data["class"])
        print(f"Target classes: {self.label_encoder.classes_}")

        # Get feature data
        self.X = self.data.drop("class", axis=1)
        self.categorical_features.remove("class")

        # Handle categorical features
        for col in self.categorical_features:
            le = LabelEncoder()
            self.X[col] = le.fit_transform(self.data[col])

        print(f"Preprocessed data shape: {self.X.shape}")
        return self.X, self.y

    def get_train_test_split(self, test_size: float = 0.2) -> Tuple:
        if self.X is None or self.y is None:
            raise ValueError("Data must be preprocessed before splitting.")

        return train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=RANDOM_SEED,
            stratify=self.y,
        )
