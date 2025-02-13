from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable
from pathlib import Path

import numpy as np
import pandas as pd
from aeon.datasets import load_regression
from scipy.io import arff


def load_mosquitosound(
        extract_path_mosquitosound: Path,
        force_refresh = False,
    ):
    """
    Loads and preprocesses the MostquitoSound erff files into
    numpy arrays, and caches to disk for faster retrieval.
    """
    # Define cache directory and paths
    cache_dir = extract_path_mosquitosound / "cache"
    cache_dir.mkdir(exist_ok=True)
    cache_files = {
        "TRAIN": {
            "X": cache_dir / "MosquitoSound_TRAIN_np_X.npy",
            "y": cache_dir / "MosquitoSound_TRAIN_np_y.npy"
        },
        "TEST": {
            "X": cache_dir / "MosquitoSound_TEST_np_X.npy",
            "y": cache_dir / "MosquitoSound_TEST_np_y.npy"
        }
    }
    
    # Check if all cache files exist and load if not forced to refresh
    all_cached = all(
        cache_files[split][arr].exists() 
        for split in ["TRAIN", "TEST"] 
        for arr in ["X", "y"]
    )
    if all_cached and not force_refresh:
        return (
            np.load(cache_files["TRAIN"]["X"]),
            np.load(cache_files["TRAIN"]["y"]),
            np.load(cache_files["TEST"]["X"]),
            np.load(cache_files["TEST"]["y"])
        )
    
    # Process raw data and cache as numpy arrays
    results = {}
    for postfix in ["TRAIN", "TEST"]:
        # Load ARFF file
        data, _meta = arff.loadarff(extract_path_mosquitosound / f"MosquitoSound_{postfix}.arff")
        X = []
        y = []
        for entry in data:
            entry = entry.tolist()
            X += [entry[:-1]]
            y += [entry[-1]]
        X = np.array(X)[:, None, :]
        
        # Convert labels to integers
        if postfix == "TRAIN":
            unique_classes, y = np.unique(y, return_inverse=True)
        else:
            # Use same label mapping as training set
            y = np.searchsorted(unique_classes, y)
        
        # Save results
        np.save(cache_files[postfix]["X"], X)
        np.save(cache_files[postfix]["y"], y)
        results[postfix] = {'X': X, 'y': y}
    
    return (
        results["TRAIN"]['X'], 
        results["TRAIN"]['y'], 
        results["TEST"]['X'], 
        results["TEST"]['y']
    )
    


def get_aeon_dataset(
        dataset_name: str, 
        extract_path: Path,
        regression_or_classification: Literal["regression", "classification"],
    ):
    """
    Loads a dataset from the UCR/UEA archive using the aeon library.
    Time series array shape is (N, D, T), dtype float64.
    Targets are float64 for regression and int64 labels in [0, ..., C-1].
    Data is not preprocessed.

    Returns:
        Tuple: 4-tuple of the form (X_train, y_train, X_test, y_test)
    """
    if dataset_name == "MosquitoSound":
        return load_mosquitosound(extract_path / "MosquitoSound")
    else:
        X_train, y_train = load_regression(dataset_name, split="train", extract_path=extract_path)
        X_test, y_test = load_regression(dataset_name, split="test", extract_path=extract_path)
        
        # make correct dtypes
        if regression_or_classification == "classification":
            unique_classes, y_train = np.unique(y_train, return_inverse=True)
            y_test = np.searchsorted(unique_classes, y_test)
            y_train = y_train.astype(np.int64)
            y_test = y_test.astype(np.int64)
        elif regression_or_classification != "regression":
            raise RuntimeError(f"invalid argument for regression_or_classification: '{regression_or_classification}'")
        
        return X_train, y_train, X_test, y_test
    
    
    
def get_dataset_metadata_df(
        dataset_names: List[str],
        data_dir: Path, 
        regression_or_classification: Literal["regression", "classification"],
        force_refresh: bool = False,
        include_n_classes: bool = False
    ) -> pd.DataFrame:
    """
    Get metadata DataFrame, either from cache or by generating it.
    """
    cache_path = data_dir / "dataset_metadata.csv"
    
    # Return cached data if it exists and refresh not forced
    if cache_path.exists() and not force_refresh:
        return pd.read_csv(cache_path, index_col=0)
    
    # Generate metadata
    metadata_dict = {}
    for dataset in dataset_names:
        print(f"Processing {dataset}...")
        try:
            X_train, y_train, X_test, y_test = get_aeon_dataset(dataset, data_dir, regression_or_classification)
            metadata_dict[dataset] = {
                'n_train': len(X_train),
                'n_test': len(X_test),
                'length': X_train.shape[2],
                'dim': X_train.shape[1],
            }
            if include_n_classes:
                metadata_dict[dataset]['n_classes'] = len(np.unique(y_train))
                
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")
            continue
    
    # Create and save DataFrame
    df = pd.DataFrame.from_dict(metadata_dict, orient='index')
    df.to_csv(cache_path)
    return df













# import requests
# from bs4 import BeautifulSoup
# import os

# def download_files(
#         base_url: str, 
#         target_dirs: List[str], 
#         root_files: List[str] = [], 
#         save_path: str = "downloaded_files"
#     ):
#     """
#     Downloads files from a website with a nested directory structure.

#     Args:
#         base_url: The base URL of the website.
#         target_dirs: A list of directories to download files from.
#         root_files: A list of files to download from the root directory.
#         save_path: The local path to save the downloaded files.
#     """

#     response = requests.get(base_url)
#     response.raise_for_status()  # Raise an exception for bad status codes

#     soup = BeautifulSoup(response.content, 'html.parser')

#     # Create a directory to store downloaded files
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)

#     # Download files from the root directory
#     for file_name in root_files:
#         file_url = base_url + file_name
#         download_file(file_url, save_path)

#     # Find all links to directories
#     for link in soup.find_all('a', href=True):
#         dir_name = link['href']
#         if dir_name.endswith('/') and dir_name[:-1] in target_dirs:
#             dir_url = base_url + dir_name
#             download_files_in_dir(dir_url, os.path.join(save_path, dir_name[:-1]))

# def download_files_in_dir(dir_url, save_path):
#     """
#     Downloads all CSV files within a specific directory.

#     Args:
#         dir_url: The URL of the directory.
#         save_path: The local path to save the downloaded files.
#     """

#     response = requests.get(dir_url)
#     response.raise_for_status()

#     soup = BeautifulSoup(response.content, 'html.parser')

#     if not os.path.exists(save_path):
#         os.makedirs(save_path)

#     for link in soup.find_all('a', href=True):
#         file_name = link['href']
#         if file_name.endswith('.csv'):
#             file_url = dir_url + file_name
#             download_file(file_url, save_path)

# def download_file(file_url, save_path):
#     """
#     Downloads a single file.

#     Args:
#         file_url: The URL of the file to download.
#         save_path: The local path to save the downloaded file.
#     """

#     try:
#         response = requests.get(file_url, stream=True)
#         response.raise_for_status()

#         file_name = os.path.basename(file_url)
#         local_file_path = os.path.join(save_path, file_name)

#         with open(local_file_path, 'wb') as f:
#             for chunk in response.iter_content(chunk_size=8192):
#                 f.write(chunk)

#         print(f"Downloaded: {file_name}")
#     except requests.exceptions.RequestException as e:
#         print(f"Error downloading {file_url}: {e}")

# # download_files(
# #     base_url = "https://timeseriesclassification.com/results/ReferenceResults/regression/", 
# #     target_dirs = ["fittime", "mae", "mape", "mse", "predicttime", "r2", "rmse"], 
# #     root_files = ["estimators.txt"],
# #     save_path = "data/bench_regression_TSER/regression"
# #     )