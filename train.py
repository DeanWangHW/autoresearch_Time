"""
Single-file PatchTST supervised training and autoresearch loop.

The goal is to keep the whole editable surface in one file, similar to
`~/Code/autoresearch/train.py`, while targeting ETTh forecasting with MSE as the
selection metric. The default task is ETTh1 with horizon 96.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from sklearn.preprocessing import StandardScaler
from torch import Tensor, optim
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = REPO_ROOT / "patchtst" / "dataset"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs" / "patchtst"

MPLCONFIGDIR = REPO_ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))


# -----------------------------------------------------------------------------
# Editable hyperparameters
# -----------------------------------------------------------------------------

# Data
MODEL = "PatchTST"
MODEL_ID = "ETTh1_336_96"
DATA = "ETTh1"
ROOT_PATH = str(DEFAULT_DATA_ROOT)
TRAIN_DATA_PATH = "ETTh1_train.csv"
BLIND_TEST_DATA_PATH = "ETTh1_blind_test.csv"
DATA_PATH = TRAIN_DATA_PATH
FEATURES = "M"
TARGET = "OT"
FREQ = "h"

# Forecasting target
SEQ_LEN = 336
LABEL_LEN = 48
PRED_LEN = 96

# PatchTST architecture
PATCH_LEN = 16
STRIDE = 8
D_MODEL = 16
N_HEADS = 4
E_LAYERS = 3
D_FF = 128
DROPOUT = 0.3
FC_DROPOUT = 0.3
HEAD_DROPOUT = 0.0
EMBED = "timeF"
ACTIVATION = "gelu"
PADDING_PATCH = "end"
REVIN = 1
AFFINE = 0
SUBTRACT_LAST = 0
DECOMPOSITION = 0
KERNEL_SIZE = 25
INDIVIDUAL = 0

# Training
RANDOM_SEED = 2021
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
TRAIN_TIME_BUDGET = 300.0
WARMUP_RATIO = 0.0
FINAL_LR_RATIO = 1.0
DEVICE = "auto"
USE_AMP = False
NUM_WORKERS = 0
MAX_TRAIN_STEPS = None
MAX_EVAL_STEPS = None
TRAIN_VAL_RATIO = 0.8

# Runtime
OUTPUT_ROOT = str(DEFAULT_OUTPUT_ROOT)
SAVE_PREDICTIONS = True


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def detect_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if requested == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {requested}")



def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    safe_true = np.where(np.abs(true) < 1e-6, 1e-6, true)
    mae = float(np.mean(np.abs(pred - true)))
    mse = float(np.mean((pred - true) ** 2))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((pred - true) / safe_true)))
    mspe = float(np.mean(np.square((pred - true) / safe_true)))
    denom = float(np.sqrt(np.sum((true - true.mean()) ** 2)))
    rse = float(np.sqrt(np.sum((true - pred) ** 2)) / max(denom, 1e-12))
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "mspe": mspe,
        "rse": rse,
    }


# -----------------------------------------------------------------------------
# Time features and datasets
# -----------------------------------------------------------------------------


class TimeFeature:
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        raise NotImplementedError


class SecondOfMinute(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5



def time_features_from_frequency_str(freq_str: str) -> list[TimeFeature]:
    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Second: [SecondOfMinute, MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
    }
    offset = to_offset(freq_str)
    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [feature() for feature in feature_classes]
    raise RuntimeError(f"Unsupported frequency: {freq_str}")



def time_features(dates: pd.DatetimeIndex, freq: str = "h") -> np.ndarray:
    return np.vstack([feature(dates) for feature in time_features_from_frequency_str(freq)])



class BaseETTDataset(Dataset):
    def __init__(
        self,
        root_path: str,
        flag: str,
        size: list[int],
        features: str,
        data_path: str,
        target: str,
        scale: bool,
        timeenc: int,
        freq: str,
        split_mode: str = "standard",
        train_val_ratio: float = 0.8,
    ) -> None:
        assert flag in {"train", "val", "test"}
        self.seq_len, self.label_len, self.pred_len = size
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.source_path = str(Path(self.root_path) / self.data_path)
        self.split_mode = split_mode
        self.train_val_ratio = train_val_ratio
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]
        self.scaler = StandardScaler()
        self.data_x: np.ndarray
        self.data_y: np.ndarray
        self.data_stamp: np.ndarray
        self._read_data()

    def _read_csv(self) -> pd.DataFrame:
        return pd.read_csv(Path(self.root_path) / self.data_path)

    def _borders(self, df_raw: pd.DataFrame) -> tuple[list[int], list[int]]:
        raise NotImplementedError

    def _train_val_file_borders(self, df_raw: pd.DataFrame) -> tuple[list[int], list[int]]:
        num_rows = len(df_raw)
        min_train_rows = self.seq_len + self.pred_len + 1
        min_val_rows = self.pred_len + 1
        split_index = int(num_rows * self.train_val_ratio)
        split_index = max(split_index, min_train_rows)
        split_index = min(split_index, num_rows - min_val_rows)
        if split_index <= self.seq_len or split_index >= num_rows:
            raise ValueError(
                f"Training split is too small for seq_len={self.seq_len}, pred_len={self.pred_len}, rows={num_rows}"
            )
        border1s = [0, max(split_index - self.seq_len, 0), max(split_index - self.seq_len, 0)]
        border2s = [split_index, num_rows, num_rows]
        return border1s, border2s

    def _full_file_borders(self, df_raw: pd.DataFrame) -> tuple[list[int], list[int]]:
        num_rows = len(df_raw)
        if num_rows <= self.seq_len + self.pred_len:
            raise ValueError(
                f"Blind test file is too small for seq_len={self.seq_len}, pred_len={self.pred_len}, rows={num_rows}"
            )
        return [0, 0, 0], [num_rows, num_rows, num_rows]

    def _read_data(self) -> None:
        df_raw = self._read_csv()
        border1s, border2s = self._borders(df_raw)
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in {"M", "MS"}:
            df_data = df_raw[df_raw.columns[1:]]
        else:
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2].copy()
        df_stamp["date"] = pd.to_datetime(df_stamp["date"])
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour)
            if self.freq == "t":
                df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute // 15)
            data_stamp = df_stamp.drop(columns=["date"]).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq=self.freq).transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self) -> int:
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data)


class DatasetETTHour(BaseETTDataset):
    def _borders(self, df_raw: pd.DataFrame) -> tuple[list[int], list[int]]:
        if self.split_mode == "train_only":
            return self._train_val_file_borders(df_raw)
        if self.split_mode == "blind_test":
            return self._full_file_borders(df_raw)
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        return border1s, border2s


class DatasetETTMinute(BaseETTDataset):
    def _borders(self, df_raw: pd.DataFrame) -> tuple[list[int], list[int]]:
        if self.split_mode == "train_only":
            return self._train_val_file_borders(df_raw)
        if self.split_mode == "blind_test":
            return self._full_file_borders(df_raw)
        border1s = [
            0,
            12 * 30 * 24 * 4 - self.seq_len,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
        ]
        return border1s, border2s


class DatasetCustom(BaseETTDataset):
    def _borders(self, df_raw: pd.DataFrame) -> tuple[list[int], list[int]]:
        if self.split_mode == "train_only":
            return self._train_val_file_borders(df_raw)
        if self.split_mode == "blind_test":
            return self._full_file_borders(df_raw)
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove("date")
        df_raw = df_raw[["date"] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_val = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df_raw)]
        return border1s, border2s


DATASET_MAP = {
    "ETTh1": DatasetETTHour,
    "ETTh2": DatasetETTHour,
    "ETTm1": DatasetETTMinute,
    "ETTm2": DatasetETTMinute,
    "custom": DatasetCustom,
}



def data_provider(config: SimpleNamespace | Any, flag: str) -> tuple[Dataset, DataLoader]:
    data_cls = DATASET_MAP[config.data]
    timeenc = 1 if config.embed == "timeF" else 0
    data_path = config.train_data_path
    split_mode = "train_only"
    if flag == "test":
        shuffle = False
        drop_last = True
        data_path = config.blind_test_data_path
        split_mode = "blind_test"
    else:
        shuffle = True
        drop_last = True

    dataset_kwargs = dict(
        root_path=config.root_path,
        data_path=data_path,
        flag=flag,
        size=[config.seq_len, config.label_len, config.pred_len],
        features=config.features,
        target=config.target,
        scale=True,
        timeenc=timeenc,
        freq=config.freq,
    )
    signature = inspect.signature(data_cls)
    if "split_mode" in signature.parameters:
        dataset_kwargs["split_mode"] = split_mode
    if "train_val_ratio" in signature.parameters:
        dataset_kwargs["train_val_ratio"] = config.train_val_ratio
    dataset = data_cls(**dataset_kwargs)
    print(flag, len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        drop_last=drop_last,
    )
    return dataset, loader


# -----------------------------------------------------------------------------
# PatchTST model
# -----------------------------------------------------------------------------


class Transpose(nn.Module):
    def __init__(self, *dims: int, contiguous: bool = False) -> None:
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(*self.dims)
        return x.contiguous() if self.contiguous else x



def get_activation_fn(activation: str | nn.Module) -> nn.Module:
    if callable(activation):
        return activation()
    if activation.lower() == "relu":
        return nn.ReLU()
    if activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {activation}")



class MovingAvg(nn.Module):
    def __init__(self, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        moving_mean = self.moving_avg(x)
        return x - moving_mean, moving_mean



def positional_encoding(pe: str | None, learn_pe: bool, q_len: int, d_model: int) -> nn.Parameter:
    if pe is None:
        weight = torch.empty((q_len, d_model))
        nn.init.uniform_(weight, -0.02, 0.02)
        learn_pe = False
    elif pe == "zero":
        weight = torch.empty((q_len, 1))
        nn.init.uniform_(weight, -0.02, 0.02)
    elif pe == "zeros":
        weight = torch.empty((q_len, d_model))
        nn.init.uniform_(weight, -0.02, 0.02)
    elif pe in {"normal", "gauss"}:
        weight = torch.zeros((q_len, 1))
        torch.nn.init.normal_(weight, mean=0.0, std=0.1)
    elif pe == "uniform":
        weight = torch.zeros((q_len, 1))
        nn.init.uniform_(weight, a=0.0, b=0.1)
    elif pe == "sincos":
        position = torch.arange(0, q_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        weight = torch.zeros(q_len, d_model)
        weight[:, 0::2] = torch.sin(position * div_term)
        weight[:, 1::2] = torch.cos(position * div_term)
        weight = weight - weight.mean()
        weight = weight / (weight.std() * 10)
    else:
        raise ValueError(f"Unsupported positional encoding: {pe}")
    return nn.Parameter(weight, requires_grad=learn_pe)



class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True, subtract_last: bool = False) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x: Tensor, mode: str) -> Tensor:
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError

    def _get_statistics(self, x: Tensor) -> None:
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x: Tensor) -> Tensor:
        x = x - (self.last if self.subtract_last else self.mean)
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: Tensor) -> Tensor:
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + (self.last if self.subtract_last else self.mean)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_dropout: float = 0.0, res_attention: bool = False) -> None:
        super().__init__()
        head_dim = d_model // n_heads
        self.scale = head_dim ** -0.5
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        attn_scores = torch.matmul(q, k) * self.scale
        if prev is not None:
            attn_scores = attn_scores + prev
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask
        if key_padding_mask is not None:
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        if self.res_attention:
            return output, attn_weights, attn_scores
        return output, attn_weights


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        res_attention: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.w_q = nn.Linear(d_model, d_k * n_heads)
        self.w_k = nn.Linear(d_model, d_k * n_heads)
        self.w_v = nn.Linear(d_model, d_v * n_heads)
        self.sdp_attn = ScaledDotProductAttention(d_model, n_heads, attn_dropout, res_attention)
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))
        self.res_attention = res_attention

    def forward(
        self,
        q: Tensor,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        bs = q.size(0)
        if k is None:
            k = q
        if v is None:
            v = q

        q_proj = self.w_q(q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_proj = self.w_k(k).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_proj = self.w_v(v).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)

        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(
                q_proj,
                k_proj,
                v_proj,
                prev=prev,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            output, attn_weights = self.sdp_attn(
                q_proj,
                k_proj,
                v_proj,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)
        if self.res_attention:
            return output, attn_weights, attn_scores
        return output, attn_weights


class TSTEncoderLayer(nn.Module):
    def __init__(
        self,
        q_len: int,
        d_model: int,
        n_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        store_attn: bool = False,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        activation: str = "gelu",
        res_attention: bool = False,
        pre_norm: bool = False,
    ) -> None:
        del q_len
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(
            d_model,
            n_heads,
            d_k,
            d_v,
            res_attention=res_attention,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
        )
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)
            self.norm_ffn = nn.LayerNorm(d_model)
        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(
        self,
        src: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if self.pre_norm:
            src = self.norm_attn(src)
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask, attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)

        if self.pre_norm:
            src = self.norm_ffn(src)
        src2 = self.ff(src)
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        return src


class TSTEncoder(nn.Module):
    def __init__(
        self,
        q_len: int,
        d_model: int,
        n_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        activation: str = "gelu",
        res_attention: bool = False,
        n_layers: int = 1,
        pre_norm: bool = False,
        store_attn: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TSTEncoderLayer(
                    q_len,
                    d_model,
                    n_heads,
                    d_k=d_k,
                    d_v=d_v,
                    d_ff=d_ff,
                    norm=norm,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    activation=activation,
                    res_attention=res_attention,
                    pre_norm=pre_norm,
                    store_attn=store_attn,
                )
                for _ in range(n_layers)
            ]
        )
        self.res_attention = res_attention

    def forward(
        self,
        src: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = src
        scores = None
        if self.res_attention:
            for layer in self.layers:
                output, scores = layer(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            for layer in self.layers:
                output = layer(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return output


class TSTiEncoder(nn.Module):
    def __init__(
        self,
        c_in: int,
        patch_num: int,
        patch_len: int,
        max_seq_len: int = 1024,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 16,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        act: str = "gelu",
        store_attn: bool = False,
        key_padding_mask: str = "auto",
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        del c_in, max_seq_len, key_padding_mask, padding_var, attn_mask, verbose, kwargs
        super().__init__()
        self.patch_num = patch_num
        self.patch_len = patch_len
        q_len = patch_num
        self.w_p = nn.Linear(patch_len, d_model)
        self.w_pos = positional_encoding(pe, learn_pe, q_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder = TSTEncoder(
            q_len,
            d_model,
            n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            activation=act,
            res_attention=res_attention,
            n_layers=n_layers,
            pre_norm=pre_norm,
            store_attn=store_attn,
        )

    def forward(self, x: Tensor) -> Tensor:
        n_vars = x.shape[1]
        x = x.permute(0, 1, 3, 2)
        x = self.w_p(x)
        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        u = self.dropout(u + self.w_pos)
        z = self.encoder(u)
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))
        return z.permute(0, 1, 3, 2)


class FlattenHead(nn.Module):
    def __init__(self, individual: bool, n_vars: int, nf: int, target_window: int, head_dropout: float = 0.0) -> None:
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars
        if self.individual:
            self.flattens = nn.ModuleList([nn.Flatten(start_dim=-2) for _ in range(n_vars)])
            self.linears = nn.ModuleList([nn.Linear(nf, target_window) for _ in range(n_vars)])
            self.dropouts = nn.ModuleList([nn.Dropout(head_dropout) for _ in range(n_vars)])
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: Tensor) -> Tensor:
        if self.individual:
            outputs = []
            for idx in range(self.n_vars):
                z = self.flattens[idx](x[:, idx, :, :])
                z = self.linears[idx](z)
                z = self.dropouts[idx](z)
                outputs.append(z)
            return torch.stack(outputs, dim=1)
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)


class PatchTSTBackbone(nn.Module):
    def __init__(
        self,
        c_in: int,
        context_window: int,
        target_window: int,
        patch_len: int,
        stride: int,
        max_seq_len: int = 1024,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 16,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        act: str = "gelu",
        key_padding_mask: str = "auto",
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        fc_dropout: float = 0.0,
        head_dropout: float = 0.0,
        padding_patch: Optional[str] = None,
        pretrain_head: bool = False,
        head_type: str = "flatten",
        individual: bool = False,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == "end":
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1
        self.backbone = TSTiEncoder(
            c_in,
            patch_num=patch_num,
            patch_len=patch_len,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=act,
            key_padding_mask=key_padding_mask,
            padding_var=padding_var,
            attn_mask=attn_mask,
            res_attention=res_attention,
            pre_norm=pre_norm,
            store_attn=store_attn,
            pe=pe,
            learn_pe=learn_pe,
            verbose=verbose,
            **kwargs,
        )
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        if self.pretrain_head:
            self.head = nn.Sequential(nn.Dropout(fc_dropout), nn.Conv1d(self.head_nf, c_in, 1))
        elif head_type == "flatten":
            self.head = FlattenHead(individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

    def forward(self, z: Tensor) -> Tensor:
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "norm")
            z = z.permute(0, 2, 1)
        if self.padding_patch == "end":
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = z.permute(0, 1, 3, 2)
        z = self.backbone(z)
        z = self.head(z)
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "denorm")
            z = z.permute(0, 2, 1)
        return z


class PatchTSTModel(nn.Module):
    def __init__(self, config: SimpleNamespace | Any) -> None:
        super().__init__()
        c_in = config.enc_in
        context_window = config.seq_len
        target_window = config.pred_len
        common_kwargs = dict(
            c_in=c_in,
            context_window=context_window,
            target_window=target_window,
            patch_len=config.patch_len,
            stride=config.stride,
            n_layers=config.e_layers,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            fc_dropout=config.fc_dropout,
            head_dropout=config.head_dropout,
            padding_patch=config.padding_patch,
            individual=bool(config.individual),
            revin=bool(config.revin),
            affine=bool(config.affine),
            subtract_last=bool(config.subtract_last),
        )
        self.decomposition = bool(config.decomposition)
        if self.decomposition:
            self.decomp_module = SeriesDecomp(config.kernel_size)
            self.model_trend = PatchTSTBackbone(**common_kwargs)
            self.model_res = PatchTSTBackbone(**common_kwargs)
        else:
            self.model = PatchTSTBackbone(**common_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init = res_init.permute(0, 2, 1)
            trend_init = trend_init.permute(0, 2, 1)
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            return x.permute(0, 2, 1)
        x = x.permute(0, 2, 1)
        x = self.model(x)
        return x.permute(0, 2, 1)


# -----------------------------------------------------------------------------
# Direct training entrypoint
# -----------------------------------------------------------------------------


def build_setting(config: SimpleNamespace) -> str:
    return (
        f"{config.model_id}_{config.model}_{config.data}_ft{config.features}_sl{config.seq_len}"
        f"_ll{config.label_len}_pl{config.pred_len}_dm{config.d_model}_nh{config.n_heads}"
        f"_el{config.e_layers}_df{config.d_ff}_eb{config.embed}"
    )


def build_config(overrides: dict[str, Any] | None = None) -> SimpleNamespace:
    config = {
        "random_seed": RANDOM_SEED,
        "model": MODEL,
        "model_id": MODEL_ID,
        "data": DATA,
        "root_path": str(ROOT_PATH),
        "data_path": DATA_PATH,
        "train_data_path": TRAIN_DATA_PATH,
        "blind_test_data_path": BLIND_TEST_DATA_PATH,
        "features": FEATURES,
        "target": TARGET,
        "freq": FREQ,
        "seq_len": SEQ_LEN,
        "label_len": LABEL_LEN,
        "pred_len": PRED_LEN,
        "patch_len": PATCH_LEN,
        "stride": STRIDE,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "e_layers": E_LAYERS,
        "d_ff": D_FF,
        "dropout": DROPOUT,
        "fc_dropout": FC_DROPOUT,
        "head_dropout": HEAD_DROPOUT,
        "embed": EMBED,
        "activation": ACTIVATION,
        "padding_patch": PADDING_PATCH,
        "revin": REVIN,
        "affine": AFFINE,
        "subtract_last": SUBTRACT_LAST,
        "decomposition": DECOMPOSITION,
        "kernel_size": KERNEL_SIZE,
        "individual": INDIVIDUAL,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "train_time_budget": float(TRAIN_TIME_BUDGET),
        "warmup_ratio": float(WARMUP_RATIO),
        "final_lr_ratio": float(FINAL_LR_RATIO),
        "device": DEVICE,
        "use_amp": USE_AMP,
        "num_workers": NUM_WORKERS,
        "max_train_steps": MAX_TRAIN_STEPS,
        "max_eval_steps": MAX_EVAL_STEPS,
        "train_val_ratio": float(TRAIN_VAL_RATIO),
        "output_root": str(OUTPUT_ROOT),
        "save_predictions": SAVE_PREDICTIONS,
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
    }
    if overrides:
        unknown = sorted(set(overrides) - set(config))
        if unknown:
            raise KeyError(f"Unknown config override(s): {unknown}")
        config.update(overrides)
    if overrides and "data_path" in overrides and "train_data_path" not in overrides:
        config["train_data_path"] = config["data_path"]
    config["data_path"] = config["train_data_path"]

    if config["model"] != "PatchTST":
        raise ValueError("This entrypoint only supports PatchTST.")
    if int(config["pred_len"]) != 96:
        raise ValueError(f"This project is fixed to horizon 96, got pred_len={config['pred_len']}")
    if config["features"] == "S":
        config["enc_in"] = config["dec_in"] = config["c_out"] = 1
    config["output_root"] = str(config["output_root"])
    config["checkpoints"] = str(Path(config["output_root"]) / "checkpoints")
    config["results_dir"] = str(Path(config["output_root"]) / "results")
    namespace = SimpleNamespace(**config)
    namespace.setting = build_setting(namespace)
    return namespace


def config_to_dict(config: SimpleNamespace) -> dict[str, Any]:
    return {key: value for key, value in vars(config).items()}


def format_plaintext_summary(summary: dict[str, Any]) -> str:
    lines = [
        "---",
        f"val_mse:          {summary['val']['mse']:.6f}",
        f"blind_test_mse:   {summary['test']['mse']:.6f}",
        f"training_seconds: {summary['train']['training_seconds']:.1f}",
        f"total_seconds:    {summary['total_seconds']:.1f}",
        f"num_steps:        {summary['train']['num_steps']}",
        f"num_epochs:       {summary['train']['num_epochs']}",
        f"stop_reason:      {summary['train']['stop_reason']}",
        f"blind_test_file:  {summary['test']['source_path']}",
    ]
    if "summary_path" in summary:
        lines.append(f"summary_path:     {summary['summary_path']}")
    return "\n".join(lines)


def make_smoke_overrides(
    output_root: str | None = None,
    train_time_budget: float = 0.01,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "model_id": f"{DATA}_smoke96",
        "seq_len": 96,
        "label_len": 48,
        "pred_len": 96,
        "patch_len": 8,
        "stride": 4,
        "d_model": 8,
        "n_heads": 2,
        "e_layers": 1,
        "d_ff": 32,
        "dropout": 0.1,
        "fc_dropout": 0.1,
        "batch_size": 8,
        "train_time_budget": float(train_time_budget),
        "num_workers": 0,
        "max_eval_steps": 1,
        "save_predictions": False,
    }
    if output_root is not None:
        overrides["output_root"] = output_root
    return overrides


def autocast_context(device: torch.device, use_amp: bool):
    if use_amp and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def create_grad_scaler(device: torch.device, use_amp: bool):
    if device.type == "cuda":
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return None


def prepare_batch(batch: tuple[Tensor, ...], device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
    return (
        batch_x.float().to(device),
        batch_y.float().to(device),
        batch_x_mark.float().to(device),
        batch_y_mark.float().to(device),
    )


def run_model(
    model: PatchTSTModel,
    config: SimpleNamespace,
    batch_x: Tensor,
    batch_y: Tensor,
    batch_x_mark: Tensor,
    batch_y_mark: Tensor,
) -> tuple[Tensor, Tensor]:
    del batch_x_mark, batch_y_mark
    outputs = model(batch_x)
    f_dim = -1 if config.features == "MS" else 0
    outputs = outputs[:, -config.pred_len :, f_dim:]
    target = batch_y[:, -config.pred_len :, f_dim:]
    return outputs, target


def update_learning_rate(optimizer: optim.Optimizer, base_lr: float, progress: float, warmup_ratio: float, final_lr_ratio: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    if warmup_ratio > 0.0 and progress < warmup_ratio:
        lr = base_lr * max(progress / warmup_ratio, 1e-3)
    else:
        decay_progress = 0.0 if warmup_ratio >= 1.0 else (progress - warmup_ratio) / max(1.0 - warmup_ratio, 1e-12)
        decay_progress = min(max(decay_progress, 0.0), 1.0)
        lr = base_lr * (1.0 - (1.0 - final_lr_ratio) * decay_progress)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def evaluate_split(model: PatchTSTModel, config: SimpleNamespace, device: torch.device, split: str) -> dict[str, float]:
    dataset, loader = data_provider(config, split)
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for step_idx, batch in enumerate(loader):
            if config.max_eval_steps is not None and step_idx >= config.max_eval_steps:
                break
            batch_x, batch_y, batch_x_mark, batch_y_mark = prepare_batch(batch, device)
            with autocast_context(device, config.use_amp):
                outputs, target = run_model(model, config, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(outputs.detach().cpu().numpy())
            trues.append(target.detach().cpu().numpy())
    model.train()
    if not preds:
        raise RuntimeError(f"No batches produced for split={split}")

    pred_arr = np.concatenate(preds, axis=0)
    true_arr = np.concatenate(trues, axis=0)
    metrics = compute_metrics(pred_arr, true_arr)
    metrics["source_path"] = getattr(dataset, "source_path", "")
    metrics["num_windows"] = float(len(dataset))
    out_dir = Path(config.results_dir) / config.setting / split
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    if config.save_predictions:
        np.save(out_dir / "pred.npy", pred_arr)
        np.save(out_dir / "true.npy", true_arr)
    return metrics


def run_experiment(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = build_config(overrides)
    device = detect_device(config.device)
    use_amp = bool(config.use_amp and device.type == "cuda")

    set_seed(config.random_seed)
    torch.set_float32_matmul_precision("high")

    model = PatchTSTModel(config).float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    scaler = create_grad_scaler(device, use_amp)

    _, train_loader = data_provider(config, "train")
    checkpoint_path = Path(config.checkpoints) / config.setting / "checkpoint.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    train_start = time.perf_counter()
    history: list[dict[str, float]] = []
    num_steps = 0
    num_epochs = 0
    stop_reason: str | None = None
    current_lr = config.learning_rate

    while True:
        num_epochs += 1
        epoch_losses: list[float] = []
        for batch in train_loader:
            progress = 1.0 if config.train_time_budget <= 0 else min(
                (time.perf_counter() - train_start) / config.train_time_budget,
                1.0,
            )
            current_lr = update_learning_rate(
                optimizer,
                config.learning_rate,
                progress,
                config.warmup_ratio,
                config.final_lr_ratio,
            )
            optimizer.zero_grad(set_to_none=True)
            batch_x, batch_y, batch_x_mark, batch_y_mark = prepare_batch(batch, device)
            with autocast_context(device, use_amp):
                outputs, target = run_model(model, config, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(outputs, target)
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            num_steps += 1
            epoch_losses.append(float(loss.detach().cpu().item()))
            elapsed = time.perf_counter() - train_start
            if config.max_train_steps is not None and num_steps >= config.max_train_steps:
                stop_reason = "max_train_steps"
                break
            if num_steps >= 1 and elapsed >= config.train_time_budget:
                stop_reason = "time_budget"
                break

        elapsed = time.perf_counter() - train_start
        history.append(
            {
                "epoch": float(num_epochs),
                "train_loss": float(np.mean(epoch_losses)) if epoch_losses else math.nan,
                "elapsed_seconds": elapsed,
                "num_steps": float(num_steps),
                "learning_rate": current_lr,
            }
        )
        print(
            f"Epoch {num_epochs} | steps={num_steps} train={history[-1]['train_loss']:.6f} "
            f"elapsed={elapsed:.2f}s lr={current_lr:.6g}"
        )
        if stop_reason is not None or not epoch_losses:
            break

    training_seconds = time.perf_counter() - train_start
    torch.save(model.state_dict(), checkpoint_path)

    val_metrics = evaluate_split(model, config, device, "val")
    test_metrics = evaluate_split(model, config, device, "test")
    total_seconds = time.perf_counter() - train_start

    summary = {
        "device": str(device),
        "setting": config.setting,
        "config": config_to_dict(config),
        "train": {
            "history": history,
            "checkpoint_path": str(checkpoint_path),
            "num_steps": num_steps,
            "num_epochs": num_epochs,
            "stop_reason": stop_reason,
            "training_seconds": training_seconds,
            "time_budget_seconds": config.train_time_budget,
        },
        "val": val_metrics,
        "test": test_metrics,
        "total_seconds": total_seconds,
    }
    summary_dir = Path(config.results_dir) / config.setting
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "run_summary.txt"
    summary["summary_path"] = str(summary_path)
    plaintext_summary = format_plaintext_summary(summary)
    summary_path.write_text(plaintext_summary + "\n", encoding="utf-8")
    print(plaintext_summary)
    print(json.dumps(summary, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-file PatchTST training")
    parser.add_argument("--smoke", action="store_true", help="Run a minimal horizon-96 smoke configuration")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    overrides = make_smoke_overrides() if args.smoke else None
    return run_experiment(overrides=overrides)


if __name__ == "__main__":
    main()
