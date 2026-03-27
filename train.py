"""
Single-file PatchTST supervised training and autoresearch loop.

The goal is to keep the whole editable surface in one file, similar to
`~/Code/autoresearch/train.py`, while targeting ETTh forecasting with MSE as the
selection metric. The default task is ETTh1 with horizon 96.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
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
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = REPO_ROOT / "patchtst" / "dataset"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs" / "patchtst"

MPLCONFIGDIR = REPO_ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))


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



def apply_config_overrides(config: "PatchTSTConfig", overrides: dict[str, Any] | None = None) -> "PatchTSTConfig":
    if not overrides:
        return config
    payload = config.to_dict()
    for key, value in overrides.items():
        if key not in payload:
            raise KeyError(f"Unknown config override: {key}")
        payload[key] = value
    return PatchTSTConfig(**payload)



def adjust_learning_rate(
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler._LRScheduler | lr_scheduler.OneCycleLR,
    epoch: int,
    config: "PatchTSTConfig",
    printout: bool = True,
) -> None:
    if config.lradj == "type1":
        lr_adjust = {epoch: config.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif config.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif config.lradj == "type3":
        lr_adjust = {
            epoch: config.learning_rate
            if epoch < 3
            else config.learning_rate * (0.9 ** ((epoch - 3) // 1))
        }
    elif config.lradj == "constant":
        lr_adjust = {epoch: config.learning_rate}
    elif config.lradj == "3":
        lr_adjust = {epoch: config.learning_rate if epoch < 10 else config.learning_rate * 0.1}
    elif config.lradj == "4":
        lr_adjust = {epoch: config.learning_rate if epoch < 15 else config.learning_rate * 0.1}
    elif config.lradj == "5":
        lr_adjust = {epoch: config.learning_rate if epoch < 25 else config.learning_rate * 0.1}
    elif config.lradj == "6":
        lr_adjust = {epoch: config.learning_rate if epoch < 5 else config.learning_rate * 0.1}
    elif config.lradj == "TST":
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    else:
        raise ValueError(f"Unsupported lradj mode: {config.lradj}")

    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        if printout:
            print(f"Updating learning rate to {lr}")



class EarlyStopping:
    def __init__(self, patience: int = 7, delta: float = 0.0) -> None:
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False
        self.val_loss_min = math.inf

    def __call__(self, val_loss: float, model: nn.Module, checkpoint_path: Path) -> None:
        score = -val_loss
        if self.best_score is None or score >= self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model, checkpoint_path)
            self.counter = 0
            return
        self.counter += 1
        print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
        if self.counter >= self.patience:
            self.early_stop = True

    def save_checkpoint(self, val_loss: float, model: nn.Module, checkpoint_path: Path) -> None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        self.val_loss_min = val_loss



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
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        return border1s, border2s


class DatasetETTMinute(BaseETTDataset):
    def _borders(self, df_raw: pd.DataFrame) -> tuple[list[int], list[int]]:
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



def data_provider(config: "PatchTSTConfig", flag: str) -> tuple[Dataset, DataLoader]:
    data_cls = DATASET_MAP[config.data]
    timeenc = 1 if config.embed == "timeF" else 0
    if flag == "test":
        shuffle = False
        drop_last = True
    else:
        shuffle = True
        drop_last = True

    dataset = data_cls(
        root_path=config.root_path,
        data_path=config.data_path,
        flag=flag,
        size=[config.seq_len, config.label_len, config.pred_len],
        features=config.features,
        target=config.target,
        scale=True,
        timeenc=timeenc,
        freq=config.freq,
    )
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
    def __init__(self, config: "PatchTSTConfig") -> None:
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
# Training and research configs
# -----------------------------------------------------------------------------


@dataclass
class PatchTSTConfig:
    random_seed: int = 2021
    model_id: str = "ETTh1_336_96"
    model: str = "PatchTST"
    data: str = "ETTh1"
    root_path: str = str(DEFAULT_DATA_ROOT)
    data_path: str = "ETTh1.csv"
    features: str = "M"
    target: str = "OT"
    freq: str = "h"

    seq_len: int = 336
    label_len: int = 48
    pred_len: int = 96

    fc_dropout: float = 0.3
    head_dropout: float = 0.0
    patch_len: int = 16
    stride: int = 8
    padding_patch: str = "end"
    revin: int = 1
    affine: int = 0
    subtract_last: int = 0
    decomposition: int = 0
    kernel_size: int = 25
    individual: int = 0

    embed_type: int = 0
    enc_in: int = 7
    dec_in: int = 7
    c_out: int = 7
    d_model: int = 16
    n_heads: int = 4
    e_layers: int = 3
    d_layers: int = 1
    d_ff: int = 128
    moving_avg: int = 25
    factor: int = 1
    distil: bool = True
    dropout: float = 0.3
    embed: str = "timeF"
    activation: str = "gelu"
    output_attention: bool = False

    num_workers: int = 0
    train_epochs: int = 20
    batch_size: int = 128
    patience: int = 5
    learning_rate: float = 1e-4
    des: str = "Exp"
    loss: str = "mse"
    lradj: str = "type3"
    pct_start: float = 0.3
    use_amp: bool = False

    device: str = "auto"
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    checkpoints: str = ""
    results_dir: str = ""
    test_results_dir: str = ""
    results_file: str = ""
    save_predictions: bool = True

    max_train_steps_per_epoch: int | None = None
    max_eval_steps: int | None = None

    def __post_init__(self) -> None:
        output_root = Path(self.output_root)
        if not self.checkpoints:
            self.checkpoints = str(output_root / "checkpoints")
        if not self.results_dir:
            self.results_dir = str(output_root / "results")
        if not self.test_results_dir:
            self.test_results_dir = str(output_root / "test_results")
        if not self.results_file:
            self.results_file = str(output_root / "research_results.tsv")
        if self.features == "S":
            self.enc_in = self.dec_in = self.c_out = 1
        elif self.data.startswith("ETT") and self.enc_in == 1:
            self.enc_in = self.dec_in = self.c_out = 7
        if self.model != "PatchTST":
            raise ValueError("This single-file entrypoint only supports PatchTST.")
        if self.pred_len != 96 and "smoke" not in self.model_id.lower():
            # The project focus is fixed on horizon 96; keep this explicit in normal runs.
            print(f"Warning: current run uses pred_len={self.pred_len}, not the default focus of 96.")

    def setting(self) -> str:
        return (
            f"{self.model_id}_{self.model}_{self.data}_ft{self.features}_sl{self.seq_len}"
            f"_ll{self.label_len}_pl{self.pred_len}_dm{self.d_model}_nh{self.n_heads}"
            f"_el{self.e_layers}_dl{self.d_layers}_df{self.d_ff}_fc{self.factor}"
            f"_eb{self.embed}_dt{self.distil}_{self.des}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def make_smoke_config(config: PatchTSTConfig) -> PatchTSTConfig:
    return replace(
        config,
        model_id=f"{config.data}_smoke96",
        seq_len=96,
        label_len=48,
        pred_len=96,
        d_model=8,
        n_heads=2,
        e_layers=1,
        d_layers=1,
        d_ff=32,
        batch_size=8,
        train_epochs=1,
        patience=1,
        num_workers=0,
        max_train_steps_per_epoch=1,
        max_eval_steps=1,
    )


@dataclass
class TrialResult:
    name: str
    config: dict[str, Any]
    metrics: dict[str, float]
    test_metrics: dict[str, float]
    summary: dict[str, Any]
    description: str = ""


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------


class PatchTSTTrainer:
    def __init__(self, config: PatchTSTConfig) -> None:
        self.config = config
        self.device = detect_device(config.device)
        self.use_amp = config.use_amp and self.device.type == "cuda"
        set_seed(config.random_seed)
        self.model = PatchTSTModel(config).float().to(self.device)

    def _select_optimizer(self) -> optim.Optimizer:
        return optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def _select_criterion(self) -> nn.Module:
        if self.config.loss != "mse":
            raise ValueError(f"Unsupported loss: {self.config.loss}")
        return nn.MSELoss()

    def _prepare_batch(self, batch: tuple[Tensor, ...]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        return (
            batch_x.float().to(self.device),
            batch_y.float().to(self.device),
            batch_x_mark.float().to(self.device),
            batch_y_mark.float().to(self.device),
        )

    def _autocast_context(self):
        if self.use_amp:
            return torch.cuda.amp.autocast()
        return nullcontext()

    def _limit_steps(self, step_idx: int, max_steps: int | None) -> bool:
        return max_steps is not None and step_idx >= max_steps

    def _run_model(self, batch_x: Tensor, batch_y: Tensor, batch_x_mark: Tensor, batch_y_mark: Tensor) -> tuple[Tensor, Tensor]:
        del batch_x_mark, batch_y_mark
        outputs = self.model(batch_x)
        f_dim = -1 if self.config.features == "MS" else 0
        outputs = outputs[:, -self.config.pred_len :, f_dim:]
        target = batch_y[:, -self.config.pred_len :, f_dim:]
        return outputs, target

    def evaluate_loss(self, loader: DataLoader, criterion: nn.Module, max_steps: int | None = None) -> float:
        losses: list[float] = []
        self.model.eval()
        with torch.no_grad():
            for step_idx, batch in enumerate(loader):
                if self._limit_steps(step_idx, max_steps):
                    break
                batch_x, batch_y, batch_x_mark, batch_y_mark = self._prepare_batch(batch)
                with self._autocast_context():
                    outputs, target = self._run_model(batch_x, batch_y, batch_x_mark, batch_y_mark)
                    loss = criterion(outputs, target)
                losses.append(float(loss.detach().cpu().item()))
        self.model.train()
        return float(np.mean(losses)) if losses else math.nan

    def train(self) -> dict[str, Any]:
        _, train_loader = data_provider(self.config, "train")
        _, val_loader = data_provider(self.config, "val")
        _, test_loader = data_provider(self.config, "test")

        checkpoint_path = Path(self.config.checkpoints) / self.config.setting() / "checkpoint.pth"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        effective_steps = len(train_loader)
        if self.config.max_train_steps_per_epoch is not None:
            effective_steps = min(effective_steps, self.config.max_train_steps_per_epoch)
        effective_steps = max(effective_steps, 1)

        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            steps_per_epoch=effective_steps,
            pct_start=self.config.pct_start,
            epochs=self.config.train_epochs,
            max_lr=self.config.learning_rate,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        early_stopping = EarlyStopping(patience=self.config.patience)

        history: list[dict[str, float]] = []
        for epoch in range(self.config.train_epochs):
            losses: list[float] = []
            self.model.train()
            for step_idx, batch in enumerate(train_loader):
                if self._limit_steps(step_idx, self.config.max_train_steps_per_epoch):
                    break
                optimizer.zero_grad(set_to_none=True)
                batch_x, batch_y, batch_x_mark, batch_y_mark = self._prepare_batch(batch)
                with self._autocast_context():
                    outputs, target = self._run_model(batch_x, batch_y, batch_x_mark, batch_y_mark)
                    loss = criterion(outputs, target)
                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                if self.config.lradj == "TST":
                    scheduler.step()
                losses.append(float(loss.detach().cpu().item()))

            train_loss = float(np.mean(losses)) if losses else math.nan
            val_loss = self.evaluate_loss(val_loader, criterion, self.config.max_eval_steps)
            test_loss = self.evaluate_loss(test_loader, criterion, self.config.max_eval_steps)
            history.append({
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
            })
            print(
                f"Epoch {epoch + 1}/{self.config.train_epochs} | "
                f"train={train_loss:.6f} val={val_loss:.6f} test={test_loss:.6f}"
            )
            early_stopping(val_loss, self.model, checkpoint_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.config.lradj != "TST":
                adjust_learning_rate(optimizer, scheduler, epoch + 1, self.config, printout=False)

        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        return {"history": history, "checkpoint_path": str(checkpoint_path)}

    def evaluate_split(self, split: str) -> dict[str, float]:
        _, loader = data_provider(self.config, split)
        preds: list[np.ndarray] = []
        trues: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for step_idx, batch in enumerate(loader):
                if self._limit_steps(step_idx, self.config.max_eval_steps):
                    break
                batch_x, batch_y, batch_x_mark, batch_y_mark = self._prepare_batch(batch)
                with self._autocast_context():
                    outputs, target = self._run_model(batch_x, batch_y, batch_x_mark, batch_y_mark)
                preds.append(outputs.detach().cpu().numpy())
                trues.append(target.detach().cpu().numpy())
        if not preds:
            raise RuntimeError(f"No batches produced for split={split}")
        pred_arr = np.concatenate(preds, axis=0)
        true_arr = np.concatenate(trues, axis=0)
        metrics = compute_metrics(pred_arr, true_arr)
        out_dir = Path(self.config.results_dir) / self.config.setting() / split
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        if self.config.save_predictions:
            np.save(out_dir / "pred.npy", pred_arr)
            np.save(out_dir / "true.npy", true_arr)
        return metrics

    def run(self) -> dict[str, Any]:
        train_summary = self.train()
        val_metrics = self.evaluate_split("val")
        test_metrics = self.evaluate_split("test")
        summary = {
            "device": str(self.device),
            "setting": self.config.setting(),
            "train": train_summary,
            "val": val_metrics,
            "test": test_metrics,
        }
        print(json.dumps(summary, indent=2))
        return summary


# -----------------------------------------------------------------------------
# Autoresearch-style experiment loop
# -----------------------------------------------------------------------------


class PatchTSTResearcher:
    def __init__(self, base_config: PatchTSTConfig, selection_split: str = "val") -> None:
        if selection_split not in {"val", "test"}:
            raise ValueError("selection_split must be 'val' or 'test'")
        self.base_config = base_config
        self.selection_split = selection_split
        self.results_file = Path(base_config.results_file)
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_results_header()

    def _ensure_results_header(self) -> None:
        if self.results_file.exists():
            return
        with open(self.results_file, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["trial", "selection_mse", "test_mse", "status", "description"])

    def default_candidates(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "lower_dropout",
                "overrides": {"dropout": 0.1, "fc_dropout": 0.1},
                "description": "reduce backbone and head dropout",
            },
            {
                "name": "wider_model",
                "overrides": {"d_model": 24, "n_heads": 4, "d_ff": 96},
                "description": "increase model width and feedforward capacity",
            },
            {
                "name": "larger_patch",
                "overrides": {"patch_len": 24, "stride": 12},
                "description": "use larger temporal patches",
            },
            {
                "name": "no_revin",
                "overrides": {"revin": 0},
                "description": "disable RevIN normalization",
            },
        ]

    def _trial_config(self, name: str, overrides: dict[str, Any] | None = None) -> PatchTSTConfig:
        config = apply_config_overrides(self.base_config, overrides)
        return replace(
            config,
            model_id=f"{self.base_config.data}_{name}",
            des=name,
        )

    def run_trial(
        self,
        name: str,
        overrides: dict[str, Any] | None = None,
        description: str = "",
    ) -> TrialResult:
        config = self._trial_config(name, overrides)
        trainer = PatchTSTTrainer(config)
        summary = trainer.run()
        selection_metrics = summary[self.selection_split]
        result = TrialResult(
            name=name,
            config=config.to_dict(),
            metrics=selection_metrics,
            test_metrics=summary["test"],
            summary=summary,
            description=description or name,
        )
        summary_path = Path(config.output_root) / "summaries" / f"{name}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return result

    def compare_trials(self, baseline: TrialResult, candidate: TrialResult) -> dict[str, Any]:
        baseline_mse = baseline.metrics["mse"]
        candidate_mse = candidate.metrics["mse"]
        status = "keep" if candidate_mse < baseline_mse else "discard"
        return {
            "baseline": baseline.name,
            "candidate": candidate.name,
            "status": status,
            "baseline_mse": baseline_mse,
            "candidate_mse": candidate_mse,
            "delta_mse": candidate_mse - baseline_mse,
        }

    def log_trial(self, trial: TrialResult, status: str, description: str) -> None:
        with open(self.results_file, "a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow([
                trial.name,
                f"{trial.metrics['mse']:.6f}",
                f"{trial.test_metrics['mse']:.6f}",
                status,
                description,
            ])

    def run_research(self, max_trials: int = 3) -> dict[str, Any]:
        baseline = self.run_trial("baseline", description="baseline")
        self.log_trial(baseline, "keep", "baseline")
        best = baseline
        decisions: list[dict[str, Any]] = []
        for candidate_spec in self.default_candidates()[:max_trials]:
            candidate = self.run_trial(
                candidate_spec["name"],
                overrides=candidate_spec["overrides"],
                description=candidate_spec["description"],
            )
            decision = self.compare_trials(best, candidate)
            self.log_trial(candidate, decision["status"], candidate_spec["description"])
            decisions.append(decision)
            if decision["status"] == "keep":
                best = candidate
        report = {
            "selection_split": self.selection_split,
            "results_file": str(self.results_file),
            "best_trial": best.name,
            "best_selection_mse": best.metrics["mse"],
            "best_test_mse": best.test_metrics["mse"],
            "decisions": decisions,
        }
        print(json.dumps(report, indent=2))
        return report


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-file PatchTST training and research")
    parser.add_argument("--smoke", action="store_true", help="Run a minimal horizon-96 smoke configuration")
    parser.add_argument("--research", action="store_true", help="Run an autoresearch-style trial loop")
    parser.add_argument("--research-trials", type=int, default=3, help="Number of candidate trials to run")
    parser.add_argument("--selection-split", choices=["val", "test"], default="val")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--data", default="ETTh1")
    parser.add_argument("--data-path", default="ETTh1.csv")
    parser.add_argument("--root-path", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--features", default="M", choices=["M", "S", "MS"])
    parser.add_argument("--seq-len", type=int, default=336)
    parser.add_argument("--label-len", type=int, default=48)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-steps-per-epoch", type=int)
    parser.add_argument("--max-eval-steps", type=int)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model-id", default="ETTh1_336_96")
    return parser



def config_from_args(args: argparse.Namespace) -> PatchTSTConfig:
    config = PatchTSTConfig(
        model_id=args.model_id,
        data=args.data,
        data_path=args.data_path,
        root_path=args.root_path,
        features=args.features,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        max_train_steps_per_epoch=args.max_train_steps_per_epoch,
        max_eval_steps=args.max_eval_steps,
        device=args.device,
        output_root=args.output_root,
    )
    if args.smoke:
        config = make_smoke_config(config)
    return config



def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    print(json.dumps(config.to_dict(), indent=2))
    if args.research:
        researcher = PatchTSTResearcher(config, selection_split=args.selection_split)
        return researcher.run_research(max_trials=args.research_trials)
    trainer = PatchTSTTrainer(config)
    return trainer.run()


if __name__ == "__main__":
    main()
