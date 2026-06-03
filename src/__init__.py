"""ADMET-Property-Predictor — multitask regression + classification suite.

신약 발굴 ML 시리즈 2번째 조각 (Tox21 → **ADMET** → 분자생성 → 결합예측).
Tox21 레포와 동일한 모듈 구조: data / featurizers / models / losses / train / evaluate.
"""

__version__ = "0.1.0"

from . import data, featurizers  # noqa: F401

__all__ = ["data", "featurizers", "__version__"]
