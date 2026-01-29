"""Dataset loader utilities for the Customer Feedback task.

This module provides a small utility class to extract a zip archive
containing CSV files and expose train/test pandas DataFrame objects.

The implementation uses pathlib for path handling.

Example:
    loader = DatasetLoader('archive.zip', 'dataset')
    train = loader.train_df
    test = loader.test_df

Attributes:
    None
"""

import zipfile
from pathlib import Path

import pandas as pd


class DatasetLoader:
    """Load dataset CSVs from a zip archive into pandas DataFrames.

    The loader will extract the provided archive into ``extract_dir`` on
    initialisation (if the directory does not already exist) and then load
    two CSV files into :class:`pandas.DataFrame` attributes.

    Args:
        archive_path: Path to the zip archive containing the dataset CSVs.
        extract_dir: Directory where the archive will be extracted.

    Attributes:
        train_df: DataFrame for the training set (or ``None`` if loading
            failed).
        test_df: DataFrame for the test set (or ``None`` if loading
            failed).

    Raises:
        FileNotFoundError: If ``archive_path`` does not exist or required
            CSV files are missing after extraction.
    """

    def __init__(self, archive_path: str, extract_dir: str) -> None:
        self._archive_path = Path(archive_path)
        self._extract_dir = Path(extract_dir)
        self.train_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None

        self._extract_dataset()
        self._create_dataframes()

    def _extract_dataset(self) -> None:
        """Extract the zip archive into the extraction directory.

        If ``extract_dir`` already exists the method does nothing. If the
        archive does not exist a :class:`FileNotFoundError` is raised.
        """
        if not self._archive_path.is_file():
            raise FileNotFoundError(f"Archive not found: {self._archive_path}")

        if not self._extract_dir.exists():
            print(f"Extracting {self._archive_path.name}...")
            self._extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(self._archive_path, "r") as zip_file:
                zip_file.extractall(self._extract_dir)
            print(f"Dataset extracted: {self._extract_dir.resolve()}")
        else:
            print(f"Dataset already extracted: {self._extract_dir.resolve()}")

    def _create_dataframes(self) -> None:
        """Load CSV files from the extraction directory into DataFrames.

        Expects the files ``AI_Portfolio_Task1_Test.csv`` and
        ``AI_Portfolio_Task1_Train.csv`` to be present in ``extract_dir``.
        If a file is missing a :class:`FileNotFoundError` is raised.
        """
        test_csv = self._extract_dir / "AI_Portfolio_Task1_Test.csv"
        train_csv = self._extract_dir / "AI_Portfolio_Task1_Train.csv"

        if not test_csv.is_file():
            raise FileNotFoundError(f"Test CSV not found: {test_csv}")

        if not train_csv.is_file():
            raise FileNotFoundError(f"Train CSV not found: {train_csv}")

        self.test_df = pd.read_csv(test_csv)
        self.train_df = pd.read_csv(train_csv)