import os
import math
import numpy as np
import pandas as pd
import joblib

from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset


class FlowImgDataset(Dataset):
    def __init__(self, X_img: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X_img).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_cicids_parquet(data_dir: str) -> pd.DataFrame:
    dfs = []

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = os.listdir(data_dir)
    parquet_files = [f for f in files if f.endswith(".parquet")]

    if len(parquet_files) == 0:
        raise FileNotFoundError(f"No parquet files found in: {data_dir}")

    for f in parquet_files:
        path = os.path.join(data_dir, f)
        print(f"Loading: {f}")
        df = pd.read_parquet(path)
        df["source_file"] = f
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    data.columns = [c.strip() for c in data.columns]

    print("Raw shape:", data.shape)
    if "Label" not in data.columns:
        raise KeyError("'Label' column not found in dataset.")

    print("Top labels:\n", data["Label"].value_counts().head(15))
    return data


def preprocess_dataframe(data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    data = data.copy()

    data["label_binary"] = (
        data["Label"].astype(str).str.upper() != "BENIGN"
    ).astype(int)

    data = data.replace([np.inf, -np.inf], np.nan)

    drop_cols = [c for c in ["Label", "label_binary", "source_file"] if c in data.columns]
    X = data.drop(columns=drop_cols)

    X = X.select_dtypes(include=[np.number])
    X = X.dropna()

    y = data.loc[X.index, "label_binary"].astype(int).values

    print("Numeric X shape:", X.shape)
    print("y shape:", y.shape)

    return X, y


def get_side_and_pad(num_features: int) -> Tuple[int, int]:
    side = int(math.ceil(math.sqrt(num_features)))
    pad = side * side - num_features
    return side, pad


def batch_vec_to_img(Xm: np.ndarray, side: int, pad: int) -> np.ndarray:
    if pad > 0:
        Xm = np.pad(Xm, ((0, 0), (0, pad)), mode="constant")

    Xm = Xm.reshape(Xm.shape[0], side, side)
    Xm = Xm[:, None, :, :]   # (N, 1, H, W)
    return Xm.astype(np.float32)


def prepare_datasets(
    data_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
    scaler_save_path: Optional[str] = None
):
    data = load_cicids_parquet(data_dir)
    X, y = preprocess_dataframe(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    if scaler_save_path is not None:
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        joblib.dump(scaler, scaler_save_path)
        print(f"Scaler saved to: {scaler_save_path}")

    num_features = X_train.shape[1]
    side, pad = get_side_and_pad(num_features)

    print(f"Features: {num_features} -> pseudo-image: (1, {side}, {side}), pad={pad}")

    X_train_img = batch_vec_to_img(X_train, side, pad)
    X_test_img = batch_vec_to_img(X_test, side, pad)

    train_ds = FlowImgDataset(X_train_img, y_train)
    test_ds = FlowImgDataset(X_test_img, y_test)

    meta = {
        "num_features": num_features,
        "side": side,
        "pad": pad,
        "train_size": len(train_ds),
        "test_size": len(test_ds)
    }

    return train_ds, test_ds, meta, scaler
