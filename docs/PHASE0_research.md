# PHASE 0 — 참고 자료 / TDC ADMET 사용법 / 리더보드 제출

## 1. 참고 레포 3개

### (1) Therapeutics Data Commons (TDC) — `mims-harvard/TDC`
- URL: https://github.com/mims-harvard/TDC
- 참고 섹션: `tdc.benchmark_group.admet_group`, `tdc.single_pred.ADME / Tox`,
  공식 docs (https://tdcommons.ai/benchmark/admet_group/overview/)
- 활용 포인트:
  - **ADMET Benchmark Group**: 22개 엔드포인트의 표준 scaffold split + 공식 지표를
    한 API로 제공. `group.get(name)` → train_val/test, `group.evaluate(pred)` → 공식 점수.
  - 5-seed 평균±표준편차 제출 프로토콜을 그대로 따라 리더보드와 직접 비교.
  - 우리 `src/data.py`의 엔드포인트 레지스트리 + split 로딩이 이걸 래핑.

### (2) DeepChem — `deepchem/deepchem`
- URL: https://github.com/deepchem/deepchem
- 참고 섹션: `deepchem.molnet` (load_delaney=ESOL, load_lipo, load_bbbp),
  `deepchem.feat` (featurizer 설계 참고만)
- 활용 포인트:
  - MoleculeNet 동일 엔드포인트(ESOL/Lipophilicity/BBBP) 교차검증 데이터 확보.
  - **교훈**: DeepChem featurizer가 최신 numpy에서 깨졌으므로 **로딩만 참고**,
    특징화는 우리가 RDKit로 직접 구현 (`src/featurizers.py`).
  - MoleculeNet split 관행(scaffold split)과 데이터 정의를 비교 기준으로 사용.

### (3) PyTorch Geometric examples — `pyg-team/pytorch_geometric`
- URL: https://github.com/pyg-team/pytorch_geometric
- 참고 섹션: `examples/` 의 GIN/GCN graph classification,
  `torch_geometric.nn` (GINConv, GCNConv, global_mean_pool)
- 활용 포인트:
  - GNN trunk 설계(3×Conv → global_mean_pool → MLP head)의 검증된 레퍼런스.
  - 멀티태스크 멀티헤드 구조로 확장 (`src/models.py`).
  - 분자 그래프 mini-batch 처리(`DataLoader`, `Batch`) 패턴 차용.

> 보너스: `swansonk14/chemprop` (D-MPNN) — ADMET SOTA 비교용 정신적 베이스라인.

---

## 2. TDC ADMET Benchmark Group 사용법 요약

```python
from tdc.benchmark_group import admet_group
group = admet_group(path='data/')          # 최초 1회 자동 다운로드

# 단일 엔드포인트
benchmark = group.get('Caco2_Wang')        # name은 leaderboard 명칭
name = benchmark['name']
train_val, test = benchmark['train_val'], benchmark['test']  # pandas DF: Drug, Y(, Drug_ID)

# 5-seed valid split (공식 제출 프로토콜)
predictions_list = []
for seed in [1, 2, 3, 4, 5]:
    train, valid = group.get_train_valid_split(benchmark=name, split_type='default', seed=seed)
    # ... featurize(RDKit) → train model → predict on test ...
    predictions_list.append({name: y_pred_test})

results = group.evaluate_many(predictions_list)   # {name: [mean, std]} ← 공식 지표
```

- `train_val`/`test` 컬럼: `Drug`(SMILES), `Y`(타깃), `Drug_ID`.
- split은 **scaffold split이 이미 적용**되어 있음 → 우리가 따로 안 나눠도 됨.
  (폴백: TDC 미설치/오프라인 시 RDKit `scaffold_split` 직접 수행 — `src/data.py`.)
- **공식 지표 (엔드포인트별 고정)**:

| 엔드포인트 (leaderboard name) | type | 공식 지표 |
|---|---|---|
| Caco2_Wang | reg | MAE |
| Solubility_AqSolDB | reg | MAE |
| Lipophilicity_AstraZeneca | reg | MAE |
| PPBR_AZ | reg | MAE |
| VDss_Lombardo | reg | Spearman |
| Half_Life_Obach | reg | Spearman |
| Clearance_Hepatocyte_AZ | reg | Spearman |
| Clearance_Microsome_AZ | reg | Spearman |
| LD50_Zhu | reg | MAE |
| HydrationFreeEnergy_FreeSolv | reg | MAE |
| BBB_Martins | cls | ROC-AUC |
| Bioavailability_Ma | cls | ROC-AUC |
| HIA_Hou | cls | ROC-AUC |
| Pgp_Broccatelli | cls | ROC-AUC |
| CYP2D6_Veith | cls | PR-AUC |
| CYP3A4_Veith | cls | PR-AUC |
| CYP2C9_Veith | cls | PR-AUC |
| CYP2D6_Substrate_CarbonMangels | cls | PR-AUC |
| CYP3A4_Substrate_CarbonMangels | cls | ROC-AUC |
| CYP2C9_Substrate_CarbonMangels | cls | PR-AUC |
| hERG | cls | ROC-AUC |
| AMES | cls | ROC-AUC |
| DILI | cls | ROC-AUC |

- `group.evaluate(pred)` 또는 `group.evaluate_many(list)`가 위 지표를 **자동 선택**.
  → 우리 `src/evaluate.py`의 TDC 래퍼가 이를 호출해 리더보드와 동일 기준 출력.

---

## 3. TDC 리더보드 제출 방법

1. **프로토콜 고정**: 각 엔드포인트마다 seed `[1,2,3,4,5]` 5회 학습.
   - 매 seed마다 `get_train_valid_split(..., seed=seed)`로 train/valid 분리,
     valid로 early stopping, **고정된 test로 예측**.
2. `group.evaluate_many(predictions_list)` → `{name: [mean, std]}` 산출.
3. 제출 패키지:
   - 5-seed 평균±표준편차 점수
   - 재현 코드 링크 (이 GitHub repo)
   - 방법 설명 (featurizer=ECFP/graph, model=RF/MLP/GIN, 하이퍼파라미터)
4. 제출 경로: TDC 리더보드 페이지(https://tdcommons.ai/benchmark/admet_group/)의
   "Contribute" → GitHub PR to `mims-harvard/TDC` leaderboard, 또는 폼 제출.
5. 우리 repo의 `04_Comparison_Leaderboard.ipynb`가 5-seed 점수 테이블 +
   리더보드 공개 점수와 나란히 비교(차이/순위 추정)를 생성 → 제출 근거로 사용.

> 주의: 리더보드 공정성 위해 **test 라벨로 튜닝 금지**. valid로만 모델 선택.

---

✅ PHASE 0 완료 — PHASE 1로 진행.
