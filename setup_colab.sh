#!/usr/bin/env bash
# =============================================================================
# setup_colab.sh — ADMET-Property-Predictor 자가치유형 Colab 셋업
# 사용법(Colab 셀):
#   !git clone https://github.com/zqvo04/admet-property-predictor.git || \
#       (cd admet-property-predictor && git pull)
#   %cd admet-property-predictor
#   !bash setup_colab.sh
# torch 2.x / py3.11~3.12 / CUDA 충돌 회피 검증 순서.
# =============================================================================
set -e

echo "================================================================"
echo " ADMET setup: detecting environment"
echo "================================================================"
python - <<'PY'
import sys
print("python:", sys.version.split()[0])
try:
    import torch
    print("torch:", torch.__version__, "| cuda:", torch.version.cuda, "| gpu:", torch.cuda.is_available())
except Exception as e:
    print("torch: NOT INSTALLED (", e, ")")
PY

echo "----------------------------------------------------------------"
echo " 1/4  cheminformatics + ML core (rdkit, sklearn, scipy, pandas)"
echo "----------------------------------------------------------------"
pip install -q "rdkit>=2023.9.1" "scikit-learn>=1.3" "scipy>=1.11" "pandas>=2.0" tqdm seaborn

echo "----------------------------------------------------------------"
echo " 2/4  PyTDC (benchmark group + official metrics)"
echo "----------------------------------------------------------------"
# PyTDC는 옛 numpy 핀을 끌고 오므로 numpy를 다시 안 깨도록 마지막에 정리
pip install -q PyTDC

echo "----------------------------------------------------------------"
echo " 3/4  PyTorch Geometric (torch 버전에 맞춰 wheel index 자동 선택)"
echo "----------------------------------------------------------------"
python - <<'PY'
import subprocess, sys
try:
    import torch
    tv = torch.__version__.split('+')[0]
    cu = ('cu' + torch.version.cuda.replace('.', '')) if torch.cuda.is_available() and torch.version.cuda else 'cpu'
    idx = f"https://data.pyg.org/whl/torch-{tv}+{cu}.html"
    print("PyG wheel index:", idx)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "torch-scatter", "torch-sparse", "-f", idx])
except Exception as e:
    print("optional PyG companion wheels skipped:", e)
PY
pip install -q "torch-geometric>=2.4"

echo "----------------------------------------------------------------"
echo " 4/4  numpy 호환성 최종 정리 (TDC가 깬 경우 복구)"
echo "----------------------------------------------------------------"
pip install -q "numpy>=1.26" "matplotlib>=3.7"

echo "================================================================"
echo " verification"
echo "================================================================"
python - <<'PY'
mods = ["numpy", "pandas", "rdkit", "sklearn", "scipy", "torch", "torch_geometric", "tdc", "matplotlib", "seaborn"]
for m in mods:
    try:
        mod = __import__(m)
        print(f"  ok  {m:18s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  XX  {m:18s} {e}")
print("\n✅ setup_colab.sh 완료")
PY
