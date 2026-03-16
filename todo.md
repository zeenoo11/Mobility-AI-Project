# 연구 TODO

**연구 제목:** Cross-Attention 기반 멀티모달 융합을 통한 전기차 운전 패턴-배터리 에너지 소모 인과 해석
**목표 저널:** Applied Energy (IF ~13, Q1) / Batteries (MDPI) 대안
**총 기간:** 6개월 (2026-03 ~ 2026-08)

---

## Phase 0. 환경 구축 (1주차)

- [x] Python 가상환경 생성 — **uv**로 관리 (`pyproject.toml`, `uv.lock`)
  - 설치 패키지: `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `shap`, `scipy`, `jupyter`, `ipykernel`
  - 실행: `uv sync` / Jupyter: `uv run jupyter lab`
- [x] 프로젝트 폴더 구조 확정 (`src/`, `notebooks/`, `data/`, `results/`, `report/`)
- [ ] `src/` 아래 모듈 뼈대 생성 (`data_loader.py`, `models.py`, `train.py`, `evaluate.py`, `visualize.py`)

---

## Phase 1. 데이터 확보 및 전처리 (1~2개월차)

### 1-1. 데이터 다운로드
- [x] **BMW i3 Real Driving Cycles** → `data/bmw_i3_driving_cycles/`
  - TripA 32개 + TripB 38개 = 70개 트립, 세미콜론 구분 CSV, 48개 컬럼
- [x] **McMaster LG 18650HG2** → `data/mcmaster_lg18650hg2/`
  - 6개 온도 조건 (`n20`, `n10`, `0`, `10`, `25`, `40`°C), UDDS/HWFET/US06/LA92/Mixed 사이클

### 1-2. 탐색적 데이터 분석 (EDA)
- [ ] `notebooks/01_eda_bmw_i3.ipynb` 작성
  - 72개 트립 분포 확인 (계절별, 거리별)
  - 주행 변수 12개 시계열 플롯 및 통계량
  - 결측치·이상치 탐지
- [ ] `notebooks/02_eda_battery.ipynb` 작성
  - 배터리 전압·전류·온도·SoC 분포 확인
  - 트립별 에너지 소모량(Wh/km) 계산 및 분포 시각화
  - 계절(외기 온도)에 따른 배터리 응답 차이 분석

### 1-3. 전처리 파이프라인 (`src/data_loader.py`)
- [ ] 모달리티 분리
  - 주행 그룹: 속도, 스로틀, 브레이크, 가속도, 모터 토크, 고도 (6개)
  - 배터리 그룹: 전압, 전류, 온도, SoC, 배터리 파워 (5개)
- [ ] Min-Max 정규화 (트립 단위)
- [ ] 슬라이딩 윈도우 슬라이싱 (window=60초, stride=10초)
- [ ] 타겟 변수 계산
  - **Primary:** 구간별 에너지 소모율 (Wh/km) = `배터리파워(kW) × 시간(h) / 거리(km)`
  - **Secondary:** 배터리 스트레스 지수 (BSI) = `α·|dI/dt| + β·|dT/dt| + γ·|ΔV|` (가중치 EDA 후 결정)
- [ ] 트립 분할: Train(50) / Validation(10) / Test(12) — 여름/겨울 균형 유지
- [ ] `src/dataset.py` — PyTorch `Dataset` / `DataLoader` 클래스 구현

---

## Phase 2. 베이스라인 모델 학습 (2개월차)

- [ ] **단일 LSTM** (`src/models/lstm_baseline.py`)
  - 입력: BMS 데이터만 (배터리 그룹 5개 변수)
  - 목표: RMSE, MAE, R² 측정
- [ ] **XGBoost** (`notebooks/03_xgboost_baseline.ipynb`)
  - 수동 특성 추출: 윈도우별 통계(평균, 표준편차, 최대/최소, 기울기)
  - 목표: 딥러닝의 우위 검증을 위한 기준선 확보
- [ ] **CNN-LSTM Concat** (`src/models/cnn_lstm_concat.py`)
  - 주행+배터리 단순 연결(concatenation) 후 학습
  - 목표: Cross-Attention 대비 단순 융합의 한계 확인
- [ ] 베이스라인 결과 정리 → `results/baseline_comparison.csv`

---

## Phase 3. 제안 모델 구현 및 학습 (3개월차)

### 3-1. Dual-Encoder Cross-Attention Network (`src/models/cross_attention_net.py`)
- [ ] **주행 인코더 (Multi-scale 1D-CNN)**
  - 커널 크기 3, 5, 7 병렬 적용 → feature concat
  - BatchNorm + ReLU + Dropout
- [ ] **배터리 인코더 (Bi-LSTM)**
  - 2-layer Bidirectional LSTM
  - Hidden state 시퀀스 전체를 Cross-Attention 입력으로 전달
- [ ] **양방향 Cross-Attention 융합**
  - Driving→Battery Attention (8 heads): Query=BMS, Key/Value=CAN
  - Battery→Driving Attention (8 heads): Query=CAN, Key/Value=BMS
  - 두 방향 결과 concat → Feed-Forward Network
- [ ] **예측 헤드**
  - Primary head: 에너지 소모율 (Wh/km) 회귀
  - Secondary head: BSI 회귀
  - Multi-task Loss = `L_wh + λ·L_bsi`

### 3-2. 학습 설정 (`src/train.py`)
- [ ] Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- [ ] Scheduler: CosineAnnealingLR
- [ ] 조기 종료 (patience=10, validation loss 기준)
- [ ] 학습 곡선 로깅 → `results/training_logs/`

### 3-3. 하이퍼파라미터 튜닝
- [ ] hidden_dim: {64, 128, 256}
- [ ] num_heads: {4, 8}
- [ ] window_size: {30, 60, 120}초
- [ ] λ (multi-task 가중치): {0.1, 0.3, 0.5}
- [ ] 최적 구성 → `results/hparam_search.json`

---

## Phase 4. Ablation Study 및 해석 분석 (4개월차)

### 4-1. Ablation Study (`notebooks/04_ablation.ipynb`)
- [ ] A1: 단일 LSTM only (베이스라인)
- [ ] A2: 1D-CNN 인코더 제거 (BMS만)
- [ ] A3: Bi-LSTM → 단방향 LSTM으로 교체
- [ ] A4: Cross-Attention → Self-Attention으로 교체
- [ ] A5: 단방향 Attention (Driving→Battery만)
- [ ] **A6: 제안 모델 전체 (Full)**
- [ ] 결과 표 → `results/ablation_table.csv`

### 4-2. Attention Weight 해석 (`notebooks/05_attention_analysis.ipynb`)
- [ ] **교차모달 인과 맵 (Cross-Modal Causal Map)**
  - 주행 변수 × 배터리 변수 평균 Attention Weight 히트맵
  - 정속 구간 vs 급가속 구간 비교
- [ ] **시간적 인과 프로파일**
  - 급가속·급제동·고속 순항 이벤트 전후 Attention Weight 시계열 추적
- [ ] **Head별 역할 분석**
  - 8개 Head의 가중치 패턴 클러스터링
  - 각 Head가 전문화한 교차모달 관계 명명 (예: "가속-전류 관계")

---

## Phase 5. 클러스터링 및 맞춤형 전략 도출 (5개월차)

- [ ] **운전 습관 특성 추출** (`notebooks/06_driving_clustering.ipynb`)
  - 트립별 피처: 평균 가속도, 급가속 빈도, 정지 비율, 최고 속도, 평균 스로틀 개도
- [ ] **클러스터링**
  - k-means (k=2~5) + Silhouette Score로 최적 k 결정
  - 클러스터 시각화 (PCA 2D)
- [ ] **클러스터별 인과 맵 비교**
  - 공격적 운전 vs 에코 운전 클러스터의 Attention Weight 차이 분석
- [ ] **맞춤형 에코 드라이빙 가이드라인 도출**
  - "공격적 운전자는 스로틀 개도 50% 이하 유지 시 BSI X% 감소" 등 정량적 권고안

---

## Phase 6. 최종 평가 및 논문 작성 (6개월차)

### 6-1. 최종 성능 평가 (`notebooks/07_final_evaluation.ipynb`)
- [ ] 테스트셋(12개 트립) 기준 전체 모델 비교
  - 평가지표: RMSE, MAE, R² (에너지 소모율 / BSI)
- [ ] **일반화 검증:** McMaster LG 18650HG2 데이터로 Cross-domain 검증
- [ ] 결과 최종 정리 → `results/final_results.csv`

### 6-2. 논문 작성
- [ ] Abstract, Introduction 초안
- [ ] Methodology 섹션 (아키텍처 다이어그램 포함)
- [ ] Experiments & Results 섹션 (표·그래프)
- [ ] Discussion & Conclusion
- [ ] 대한산업공학회 발표 슬라이드 준비 (중간 성과)
- [ ] **Applied Energy** 투고 준비 (최종 목표)

---

## 성능 목표 (최소 기준)

| 지표 | 목표 |
|---|---|
| 에너지 소모율 RMSE | Baseline(LSTM) 대비 15% 이상 감소 |
| BSI MAE | Baseline 대비 15% 이상 감소 |
| R² (에너지 소모율) | 0.90 이상 |
| Ablation A5→A6 개선 | Cross-Attention 양방향의 유의미한 기여 확인 |

---

## Phase 2.5. MVP 개선 실험 (2026-03-16)

> MVP 결과에서 CrossAttentionNet(R²=0.923)이 CNNLSTMConcat(R²=0.946)보다 열위. 근본 원인 분석 후 개선 실험 수행.

### 2.5-1. 학습 설정 변경 (`src/train.py`)
- [x] CosineAnnealingWarmRestarts 스케줄러 적용 (`LinearWarmupCosineScheduler` 클래스)
- [x] Linear Warmup (5 epoch) 추가
- [x] EPOCHS: 30 → 100, PATIENCE: 5 → 15
- [x] Gradient clipping max_norm 유지 (1.0)

### 2.5-2. 아키텍처 수정 (`src/models.py`)
- [x] CrossAttentionNet: Residual connection 추가 (attn_output + original_proj)
- [x] CrossAttentionNet: Pre-LayerNorm 패턴 적용 (attention 전 정규화)
- [x] CrossAttentionNet: n_heads 기본값 4 → 8
- [x] CrossAttentionNet: ReLU → GELU 활성화 함수 변경
- [x] build_model(): 하이퍼파라미터 외부 주입 지원 (**kwargs)

### 2.5-3. 환경 설정
- [x] PyTorch CUDA 128 (nightly) 설정 (`pyproject.toml` 업데이트)
- [x] GPU 동작 확인 (RTX 5070 Ti, 2 epoch 테스트 통과)

### 2.5-4. 하이퍼파라미터 서치 (`hparam_search.py`)
- [x] 스크립트 작성 완료
- [x] 탐색 공간: lstm_hidden {64, 128}, n_heads {4, 8}, dropout {0.1, 0.2, 0.3}, cnn_out {48, 96}
- [ ] Grid search 실행 및 결과 저장 (`results/hparam_search.json`)
- [ ] 최적 구성으로 최종 재학습 및 평가

### 2.5-5. 결과 분석
- [x] mvp_report.md 업데이트 (11장 문헌 비교 + 12장 근본 원인 분석 + 13장 개선 계획 + 참고문헌 6편)
- [ ] 개선된 MVP 실험 실행 (`mvp_experiment.py` — 100 epoch, cosine warmup)
- [ ] 개선 전/후 성능 비교표 작성

---

## 현재 상태

- [x] 연구계획서 작성 완료 (`docs/A_research_plan.md`)
- [x] 상세 실험 계획서 작성 완료 (`docs/A_research_plan_2.md`)
- [x] 데이터 다운로드 완료 (BMW i3 70트립 + McMaster LG HG2)
- [x] 환경 구축 완료 (uv, Python 3.11, PyTorch CUDA cu128)
- [x] MVP 구현 및 실험 완료 (`src/`, `report/mvp_report.md`)
- [ ] MVP 개선 실험 (Phase 2.5) — 코드 수정 완료, **실험 실행 대기 중**
