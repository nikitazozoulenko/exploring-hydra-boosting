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
        X = np.array(X, dtype=np.float32)[:, None, :]
        
        # Convert labels to integers
        if postfix == "TRAIN":
            unique_classes, y = np.unique(y, return_inverse=True)
        else:
            # Use same label mapping as training set
            y = np.searchsorted(unique_classes, y)
        y = y.astype(np.int32)
        
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
    Time series array shape is (N, D, T), dtype float32.
    Targets are float32 for regression and int32 labels in [0, ..., C-1].
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
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        if regression_or_classification == "classification":
            unique_classes, y_train = np.unique(y_train, return_inverse=True)
            y_test = np.searchsorted(unique_classes, y_test)
            y_train = y_train.astype(np.int32)
            y_test = y_test.astype(np.int32)
        elif regression_or_classification == "regression":
            y_train = y_train.astype(np.float32)
            y_test = y_test.astype(np.float32)
        else:
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