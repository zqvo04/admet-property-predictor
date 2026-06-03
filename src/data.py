"""데이터 로딩 / 엔드포인트 레지스트리 / split / 스케일링.

설계 원칙(Tox21 교훈):
  - 데이터는 TDC/MoleculeNet에서 받되 **특징화는 RDKit**(src.featurizers)로 직접.
  - 회귀/분류를 task-type 메타데이터로 통합 (loss/metric 자동 분기).
  - 회귀 타깃은 z-score 정규화 (스케일러 저장 + 평가 시 원단위 역변환).
  - SMILES 유효성 검사 + 실패 분자 스킵. 단일/멀티태스크 모드.
  - TDC benchmark group 표준 scaffold split 사용, 없으면 RDKit scaffold split 폴백.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .featurizers import is_valid_smiles

# ---------------------------------------------------------------------------
# 1. 엔드포인트 레지스트리
#    type   : 'reg' | 'cls'
#    source : 'tdc' | 'moleculenet'
#    tdc_name      : TDC benchmark_group(admet_group) leaderboard 명칭
#    metric : TDC 공식 지표 (evaluate.py / 리더보드와 일치)
#    molnet : (선택) deepchem.molnet 교차검증 로더 이름
# ---------------------------------------------------------------------------
ENDPOINTS: Dict[str, dict] = {
    # ---------- 회귀 ----------
    "Caco2": {"type": "reg", "source": "tdc", "tdc_name": "Caco2_Wang",
              "metric": "mae", "group": "Absorption"},
    "Solubility": {"type": "reg", "source": "tdc", "tdc_name": "Solubility_AqSolDB",
                   "metric": "mae", "group": "Absorption", "molnet": "delaney"},
    "Lipophilicity": {"type": "reg", "source": "tdc", "tdc_name": "Lipophilicity_AstraZeneca",
                      "metric": "mae", "group": "Absorption", "molnet": "lipo"},
    "PPBR": {"type": "reg", "source": "tdc", "tdc_name": "PPBR_AZ",
             "metric": "mae", "group": "Distribution"},
    "VDss": {"type": "reg", "source": "tdc", "tdc_name": "VDss_Lombardo",
             "metric": "spearman", "group": "Distribution"},
    "Half_Life": {"type": "reg", "source": "tdc", "tdc_name": "Half_Life_Obach",
                  "metric": "spearman", "group": "Excretion"},
    "Clearance_Hepatocyte": {"type": "reg", "source": "tdc",
                             "tdc_name": "Clearance_Hepatocyte_AZ",
                             "metric": "spearman", "group": "Excretion"},
    "LD50": {"type": "reg", "source": "tdc", "tdc_name": "LD50_Zhu",
             "metric": "mae", "group": "Toxicity"},
    # ---------- 분류 ----------
    "BBBP": {"type": "cls", "source": "tdc", "tdc_name": "BBB_Martins",
             "metric": "roc-auc", "group": "Distribution", "molnet": "bbbp"},
    "Bioavailability": {"type": "cls", "source": "tdc", "tdc_name": "Bioavailability_Ma",
                        "metric": "roc-auc", "group": "Absorption"},
    "CYP2D6_Inhibition": {"type": "cls", "source": "tdc", "tdc_name": "CYP2D6_Veith",
                          "metric": "pr-auc", "group": "Metabolism"},
    "CYP3A4_Inhibition": {"type": "cls", "source": "tdc", "tdc_name": "CYP3A4_Veith",
                          "metric": "pr-auc", "group": "Metabolism"},
    "hERG": {"type": "cls", "source": "tdc", "tdc_name": "hERG",
             "metric": "roc-auc", "group": "Toxicity"},
}

REG_ENDPOINTS = [k for k, v in ENDPOINTS.items() if v["type"] == "reg"]
CLS_ENDPOINTS = [k for k, v in ENDPOINTS.items() if v["type"] == "cls"]


def endpoint_info(name: str) -> dict:
    if name not in ENDPOINTS:
        raise KeyError(f"unknown endpoint '{name}'. available: {list(ENDPOINTS)}")
    return ENDPOINTS[name]


# ---------------------------------------------------------------------------
# 2. z-score 스케일러 (회귀 타깃)
# ---------------------------------------------------------------------------
@dataclass
class StandardScaler1D:
    """회귀 타깃 z-score 정규화. fit/transform/inverse_transform + 저장/복원."""
    mean_: float = 0.0
    std_: float = 1.0

    def fit(self, y: np.ndarray) -> "StandardScaler1D":
        y = np.asarray(y, dtype=np.float64)
        y = y[~np.isnan(y)]
        self.mean_ = float(y.mean()) if y.size else 0.0
        s = float(y.std()) if y.size else 1.0
        self.std_ = s if s > 1e-8 else 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=np.float64) - self.mean_) / self.std_

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=np.float64) * self.std_ + self.mean_

    def to_dict(self) -> dict:
        return {"mean_": self.mean_, "std_": self.std_}

    @classmethod
    def from_dict(cls, d: dict) -> "StandardScaler1D":
        return cls(mean_=d["mean_"], std_=d["std_"])


# ---------------------------------------------------------------------------
# 3. 데이터 컨테이너
# ---------------------------------------------------------------------------
@dataclass
class ADMETData:
    """단일 엔드포인트 데이터 + split + 스케일러."""
    name: str
    task_type: str                       # 'reg' | 'cls'
    metric: str
    source: str
    train: pd.DataFrame = field(default_factory=pd.DataFrame)  # cols: Drug, Y(, Y_scaled)
    valid: pd.DataFrame = field(default_factory=pd.DataFrame)
    test: pd.DataFrame = field(default_factory=pd.DataFrame)
    scaler: Optional[StandardScaler1D] = None

    def smiles(self, split: str) -> List[str]:
        return getattr(self, split)["Drug"].tolist()

    def targets(self, split: str, scaled: bool = False) -> np.ndarray:
        df = getattr(self, split)
        col = "Y_scaled" if (scaled and "Y_scaled" in df) else "Y"
        return df[col].to_numpy(dtype=np.float32)

    def __repr__(self) -> str:
        return (f"ADMETData({self.name}, {self.task_type}, metric={self.metric}, "
                f"train={len(self.train)}, valid={len(self.valid)}, test={len(self.test)})")


# ---------------------------------------------------------------------------
# 4. SMILES 정제
# ---------------------------------------------------------------------------
def _clean(df: pd.DataFrame, smiles_col: str = "Drug", y_col: str = "Y") -> pd.DataFrame:
    df = df[[smiles_col, y_col]].copy()
    df.columns = ["Drug", "Y"]
    df = df.dropna(subset=["Drug", "Y"])
    before = len(df)
    df = df[df["Drug"].map(is_valid_smiles)].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        warnings.warn(f"{dropped} invalid SMILES dropped")
    return df


# ---------------------------------------------------------------------------
# 5. RDKit scaffold split (TDC 미사용/오프라인 폴백)
# ---------------------------------------------------------------------------
def scaffold_split(
    df: pd.DataFrame, frac=(0.8, 0.1, 0.1), seed: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bemis-Murcko scaffold 기준 결정론적 split (큰 scaffold group부터 train에 배치)."""
    from collections import defaultdict
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffolds: Dict[str, List[int]] = defaultdict(list)
    for idx, smi in enumerate(df["Drug"].tolist()):
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False)
        except Exception:
            sc = ""
        scaffolds[sc].append(idx)

    sets = sorted(scaffolds.values(), key=lambda s: (len(s), s[0]), reverse=True)
    n = len(df)
    n_train, n_valid = int(frac[0] * n), int(frac[1] * n)
    train_idx, valid_idx, test_idx = [], [], []
    for grp in sets:
        if len(train_idx) + len(grp) <= n_train:
            train_idx += grp
        elif len(valid_idx) + len(grp) <= n_valid:
            valid_idx += grp
        else:
            test_idx += grp
    return (df.iloc[train_idx].reset_index(drop=True),
            df.iloc[valid_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True))


# ---------------------------------------------------------------------------
# 6. TDC 로딩 (benchmark group 표준 split)
# ---------------------------------------------------------------------------
def load_tdc_endpoint(
    name: str, seed: int = 1, tdc_path: str = "data/", scale_reg: bool = True
) -> ADMETData:
    """TDC ADMET benchmark group에서 단일 엔드포인트 로딩.

    표준 test split 사용 + seed별 train/valid split(공식 제출 프로토콜).
    회귀 타깃은 train 통계로 z-score 정규화(Y_scaled 컬럼 추가).
    """
    info = endpoint_info(name)
    from tdc.benchmark_group import admet_group

    group = admet_group(path=tdc_path)
    bm = group.get(info["tdc_name"])
    train_val, test = bm["train_val"], bm["test"]
    train, valid = group.get_train_valid_split(
        benchmark=info["tdc_name"], split_type="default", seed=seed
    )

    data = ADMETData(name=name, task_type=info["type"], metric=info["metric"],
                     source="tdc")
    data.train = _clean(train)
    data.valid = _clean(valid)
    data.test = _clean(test)

    if info["type"] == "reg" and scale_reg:
        data.scaler = StandardScaler1D().fit(data.train["Y"].to_numpy())
        for split in ("train", "valid", "test"):
            df = getattr(data, split)
            df["Y_scaled"] = data.scaler.transform(df["Y"].to_numpy())
    return data


# ---------------------------------------------------------------------------
# 7. MoleculeNet 로딩 (교차검증/보강) — RDKit로만 featurize
# ---------------------------------------------------------------------------
_MOLNET_CSV = {
    # deepchem 미설치 시 직접 CSV URL로 폴백
    "delaney": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
    "lipo": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
    "bbbp": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
}
_MOLNET_COLS = {
    "delaney": ("smiles", "measured log solubility in mols per litre", "reg"),
    "lipo": ("smiles", "exp", "reg"),
    "bbbp": ("smiles", "p_np", "cls"),
}


def load_moleculenet(name: str, seed: int = 0, scale_reg: bool = True) -> ADMETData:
    """MoleculeNet 엔드포인트(delaney/lipo/bbbp)를 CSV로 로딩 후 scaffold split.

    TDC와 값이 다를 수 있으므로 source='moleculenet'로 명확히 라벨링.
    """
    info = ENDPOINTS.get(_molnet_to_endpoint(name), {})
    smi_col, y_col, ttype = _MOLNET_COLS[name]
    df = pd.read_csv(_MOLNET_CSV[name])
    df = df.rename(columns={smi_col: "Drug", y_col: "Y"})
    df = _clean(df)

    tr, va, te = scaffold_split(df, seed=seed)
    data = ADMETData(name=f"{name}@moleculenet", task_type=ttype,
                     metric=info.get("metric", "mae" if ttype == "reg" else "roc-auc"),
                     source="moleculenet")
    data.train, data.valid, data.test = tr, va, te
    if ttype == "reg" and scale_reg:
        data.scaler = StandardScaler1D().fit(tr["Y"].to_numpy())
        for split in ("train", "valid", "test"):
            d = getattr(data, split)
            d["Y_scaled"] = data.scaler.transform(d["Y"].to_numpy())
    return data


def _molnet_to_endpoint(molnet_name: str) -> str:
    for ep, info in ENDPOINTS.items():
        if info.get("molnet") == molnet_name:
            return ep
    return molnet_name


# ---------------------------------------------------------------------------
# 8. 멀티태스크 결합 (공통 SMILES 기준 task 행렬 + task-type 마스크)
# ---------------------------------------------------------------------------
@dataclass
class MultiTaskData:
    task_names: List[str]
    task_types: List[str]                 # 'reg'|'cls' per task
    smiles: List[str]
    Y: np.ndarray                         # (N, T) NaN=결측
    mask: np.ndarray                      # (N, T) 관측여부
    scalers: Dict[str, StandardScaler1D] = field(default_factory=dict)

    @property
    def type_vector(self) -> np.ndarray:
        """task별 0=reg, 1=cls."""
        return np.array([0 if t == "reg" else 1 for t in self.task_types], dtype=np.int64)


def build_multitask(datasets: Dict[str, ADMETData], split: str = "train") -> MultiTaskData:
    """여러 ADMETData를 SMILES union 기준 (N,T) 타깃/마스크 행렬로 결합.

    회귀 task는 각 dataset의 scaler로 정규화된 값(Y_scaled)을 사용.
    """
    task_names = list(datasets.keys())
    task_types = [datasets[n].task_type for n in task_names]

    smi_index: Dict[str, int] = {}
    rows: List[Dict[str, float]] = []
    for ti, name in enumerate(task_names):
        ds = datasets[name]
        df = getattr(ds, split)
        ycol = "Y_scaled" if (ds.task_type == "reg" and "Y_scaled" in df) else "Y"
        for smi, y in zip(df["Drug"], df[ycol]):
            if smi not in smi_index:
                smi_index[smi] = len(rows)
                rows.append({})
            rows[smi_index[smi]][name] = float(y)

    smiles = list(smi_index.keys())
    N, T = len(smiles), len(task_names)
    Y = np.full((N, T), np.nan, dtype=np.float32)
    mask = np.zeros((N, T), dtype=np.float32)
    for r, smi in enumerate(smiles):
        for ti, name in enumerate(task_names):
            if name in rows[r]:
                Y[r, ti] = rows[r][name]
                mask[r, ti] = 1.0

    scalers = {n: datasets[n].scaler for n in task_names if datasets[n].scaler is not None}
    return MultiTaskData(task_names, task_types, smiles, Y, mask, scalers)
