"""
helpers.py

General utility functions used throughout the project.
"""

from pathlib import Path


def create_directory(directory: Path) -> None:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    directory : Path
        Directory to create.
    """

    directory.mkdir(parents=True, exist_ok=True)