"""RDKit 기반 특징화 — DeepChem featurizer의 numpy 비호환 문제를 회피.

제공:
  - ecfp_featurize(smiles, r=2, nbits=2048) -> dense np.ndarray
  - mol_to_graph(smiles) -> torch_geometric.data.Data  (lazy PyG import)
  - smiles_to_mol / is_valid_smiles : 유효성 검사 유틸
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")  # RDKit 파싱 경고 억제


# ---------------------------------------------------------------------------
# SMILES 유효성
# ---------------------------------------------------------------------------
def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    return mol


def is_valid_smiles(smiles: str) -> bool:
    return smiles_to_mol(smiles) is not None


# ---------------------------------------------------------------------------
# ECFP (Morgan) — dense bit vector
# ---------------------------------------------------------------------------
def _morgan_bits(mol: Chem.Mol, r: int, nbits: int) -> np.ndarray:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=r, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.float32)
    # ConvertToNumpyArray가 numpy 버전에 따라 깨질 수 있어 비트 인덱스로 직접 채움
    for bit in fp.GetOnBits():
        arr[bit] = 1.0
    return arr


def ecfp_featurize(
    smiles: Iterable[str], r: int = 2, nbits: int = 2048, return_mask: bool = False
):
    """SMILES 리스트 -> (N, nbits) dense float32 행렬.

    유효하지 않은 SMILES는 0벡터로 채우고 mask=False. return_mask=True면 (X, mask) 반환.
    """
    smiles = list(smiles)
    X = np.zeros((len(smiles), nbits), dtype=np.float32)
    mask = np.zeros((len(smiles),), dtype=bool)
    for i, smi in enumerate(smiles):
        mol = smiles_to_mol(smi)
        if mol is None:
            continue
        X[i] = _morgan_bits(mol, r, nbits)
        mask[i] = True
    if return_mask:
        return X, mask
    return X


# ---------------------------------------------------------------------------
# Atom / bond graph features (PyG)
# ---------------------------------------------------------------------------
# 원자 특징: 원자번호 one-hot(주요 원소) + degree + formal charge + 방향성 + 수소수 + 혼성
_ATOM_LIST = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # B C N O F P S Cl Br I
_HYBRIDIZATION = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]


def _onehot(value, choices) -> List[float]:
    vec = [0.0] * (len(choices) + 1)
    if value in choices:
        vec[choices.index(value)] = 1.0
    else:
        vec[-1] = 1.0  # unknown 버킷
    return vec


def _atom_features(atom: Chem.Atom) -> List[float]:
    feats: List[float] = []
    feats += _onehot(atom.GetAtomicNum(), _ATOM_LIST)
    feats += _onehot(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5])
    feats += _onehot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
    feats += _onehot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    feats += _onehot(atom.GetHybridization(), _HYBRIDIZATION)
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)
    feats.append(atom.GetMass() * 0.01)
    return feats


# 원자 특징 차원 (모델 입력 차원 결정에 사용)
ATOM_FEATURE_DIM = len(_atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))


def _bond_features(bond: Chem.Bond) -> List[float]:
    bt = bond.GetBondType()
    return [
        float(bt == Chem.rdchem.BondType.SINGLE),
        float(bt == Chem.rdchem.BondType.DOUBLE),
        float(bt == Chem.rdchem.BondType.TRIPLE),
        float(bt == Chem.rdchem.BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


BOND_FEATURE_DIM = 6


def mol_to_graph(smiles: str, y=None):
    """SMILES -> PyG Data. self-loop 처리 포함. 유효하지 않으면 None."""
    import torch
    from torch_geometric.data import Data

    mol = smiles_to_mol(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x = torch.tensor([_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr += [bf, bf]

    # self-loop: 고립 원자(엣지 없음) 대비 + 메시지 패싱 안정화
    n = mol.GetNumAtoms()
    for i in range(n):
        edge_index.append([i, i])
        edge_attr.append([0.0] * BOND_FEATURE_DIM)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y is not None:
        data.y = torch.tensor(np.atleast_2d(np.asarray(y, dtype=np.float32)))
    data.smiles = smiles
    return data


def smiles_list_to_graphs(smiles: Iterable[str], y=None):
    """여러 SMILES -> (graphs, valid_mask). y는 (N,) 또는 (N,T) 가능."""
    smiles = list(smiles)
    y = None if y is None else np.asarray(y, dtype=np.float32)
    graphs, mask = [], []
    for i, smi in enumerate(smiles):
        yi = None if y is None else (y[i] if y.ndim > 1 else [y[i]])
        g = mol_to_graph(smi, yi)
        if g is None:
            mask.append(False)
            continue
        mask.append(True)
        graphs.append(g)
    return graphs, np.asarray(mask, dtype=bool)
