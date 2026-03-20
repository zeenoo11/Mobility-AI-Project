# SUMO Graph WaveNet 충전 수요 예측 — 연구 TODO

**최종 업데이트:** 2026-03-21
**목표:** SUMO 시뮬레이션 기반 EV 충전소 시공간 수요 예측 (Graph WaveNet)

---

## 현재 상태 요약

| 항목 | 상태 | 비고 |
|---|---|---|
| SUMO 시뮬레이션 | ✅ 완료 | 162 조합 + 1 테스트 = 163 runs |
| 데이터 파싱 | ✅ 완료 | (163, 53, 24, 6), SoC 로직 개선 반영 |
| 그래프 구축 | ✅ 완료 | 53 nodes, 462 edges |
| Graph WaveNet v1 학습 | ❌ 실패 | R² = -602,350 (치명적) |
| 코드 리팩토링 | ✅ 완료 | `src/sumo/`, `data/sumo/`, `scripts/`, `assets/` 분리 |

### v1 학습 실패 근본 원인 분석

1. **극단적 데이터 희소성**: non-zero 비율 0.7% (99.3%가 0)
   - 163 run × 53 station × 24 bin 중 충전 이벤트가 있는 셀이 전체의 0.7%
   - 27/53 충전소만 한 번이라도 활성화 (51%가 영구 비활성)
   - 활성 충전소도 run당 평균 8~13개만 사용
2. **R² 계산 오류**: 충전소별 R² 평균 → 분산이 0에 가까운 비활성 충전소에서 R²가 -∞
3. **정규화 문제**: 전체 Min-Max 정규화 → 99.3% 0에 의해 비활성 구간 정보가 없어짐
4. **타겟 선정**: num_arrivals(이산, 0~2) 예측 → 대부분 0을 예측해도 MSE가 작음

---

## Phase 1: 데이터 파이프라인 개선 (급선무)

### 1-1. 활성 충전소 필터링
- [ ] 전체 run에서 한 번이라도 활성화된 27개 충전소만 추출
- [ ] 인접 행렬도 27×27로 재구성 (graph_builder에서 서브그래프 추출)
- [ ] `src/sumo_dataset.py` 수정: `active_mask` 파라미터 추가

### 1-2. 타겟 변수 재설정
- [ ] **Primary 타겟**: `total_energy` (feature[2]) — 연속값, 물리적 의미 명확
- [ ] **log1p 변환** 적용: `y = log(1 + total_energy)` → 0과 큰 값의 스케일 차이 완화
- [ ] 역변환 함수(`expm1`) 구현하여 평가 시 원래 스케일로 복원

### 1-3. 정규화 전략 변경
- [ ] Min-Max → **Z-score 정규화** (활성 셀 기준 mean/std 계산)
- [ ] 또는: log1p 변환 후 정규화 (log 공간에서 정규화가 더 안정적)
- [ ] 정규화 통계량 저장 (역변환용)

### 1-4. 입력 윈도우 축소
- [ ] `t_input`: 12 → **6** (30분) — 2시간 시뮬레이션에서 12 step은 과다
- [ ] receptive field: 6 step이면 dilation 1,2 (2 blocks × 2 layers)로 충분

---

## Phase 2: 모델 및 학습 개선

### 2-1. 손실 함수 개선
- [ ] **Weighted MSE**: 비제로 샘플에 높은 가중치 부여
  - `loss = MSE(pred, target) * (1 + α * (target > 0))` (α=5~10)
- [ ] 또는 **Huber Loss** (outlier에 강건)
- [ ] 또는 **Zero-Inflated 접근**: 이진 분류(충전 유무) + 회귀(에너지량) 2-head

### 2-2. R² 메트릭 수정
- [ ] 충전소별 평균 → **Global R²** (전체 예측값을 flat하여 계산)
- [ ] 또는: 활성 충전소만 필터링 후 R² 계산
- [ ] 추가 메트릭: **MAPE** (활성 셀 한정), **F1-like** (충전 발생 여부 정확도)

### 2-3. 모델 구조 조정
- [ ] `t_input=6`에 맞춰 dilation 축소: 2 blocks × 2 layers (dilation 1, 2)
- [ ] hidden_dim: 32 → **48 or 64** (정보 손실 보상)
- [ ] Static node features 통합: 충전소 위치·전력 정보를 input에 concat

### 2-4. 학습 설정
- [ ] Early stopping patience: 15 → **20** (수렴 기회 확보)
- [ ] Batch size: 16 → **32** (163 runs으로 데이터 증가)
- [ ] Learning rate: 1e-3 유지, warmup 5 epoch 유지

---

## Phase 3: 실험 및 비교

### 3-1. 베이스라인 모델 구현
- [ ] **Naive(Last-value)**: 직전 시간 구간 값을 그대로 예측
- [ ] **Historical Average**: 같은 시간대 평균값 예측
- [ ] **LSTM**: 충전소별 독립 LSTM (공간 무시)
- [ ] **GCN + GRU**: 단순 GCN 공간 + GRU 시간 (non-WaveNet)

### 3-2. Ablation Study
- [ ] A1: Graph WaveNet (predefined adj only, adaptive 제거)
- [ ] A2: Graph WaveNet (adaptive adj only, predefined 제거)
- [ ] A3: WaveNet only (그래프 제거, 시간축만)
- [ ] A4: GCN only (시간축 제거, 공간만)
- [ ] A5: Full Graph WaveNet (proposed)

### 3-3. 결과 분석
- [ ] 충전소별 예측 정확도 히트맵
- [ ] Learned adjacency matrix vs predefined adjacency 비교 시각화
- [ ] 시간대별 예측 오차 분석 (피크 시간 vs 비피크)

---

## Phase 4: 시뮬레이션 확장 (선택)

### 4-1. 시뮬레이션 다양성 개선
- [ ] 시간 확장: 2시간 → 6시간 or 24시간 (시간 구간 증가 → 희소성 완화)
- [ ] 경로 다양화: 다중 시드로 경로 파일 재생성
- [ ] 충전소 설정 변형: 충전 파워 (75kW → 50/150kW 혼합)

### 4-2. 외부 변수 통합
- [ ] 시간 임베딩: 시간대(hour)를 cyclical encoding으로 피처 추가
- [ ] 날씨/온도 변수 (시뮬레이션 시나리오로 간접 반영)

---

## Phase 5: 보고서 및 논문

### 5-1. 진행 보고서 업데이트
- [ ] `report/sumo_graph_wavenet_progress.md` 갱신
  - 리팩토링된 프로젝트 구조 반영
  - 현실적 파라미터 (50-100kWh) 반영
  - v1 실패 원인 분석 + v2 개선 사항 기록
  - 새로운 데이터 통계 (163 runs, 0.7% 활성 비율) 기록

### 5-2. 학술 발표/논문 준비
- [ ] 대한산업공학회 추계학술대회 초록 작성
- [ ] 실험 결과 정리 표 + 그래프
- [ ] 축 2 (BMW i3 에너지 소모 예측) 연계 시나리오 정리

---

## 프로젝트 구조 (현재)

```
Mobility-AI-Project/
├── assets/
│   └── sumo_tutorial/          # SUMO 튜토리얼 원본 (5_electric 등)
├── data/
│   ├── bmw_i3_driving_cycles/  # 축 2 데이터
│   ├── mcmaster_lg18650hg2/    # 일반화 검증
│   └── sumo/
│       ├── graph_data/         # adj.npy, positions.npy 등
│       └── sim_outputs/        # 163 runs (station_features.npy)
├── src/
│   ├── sumo/
│   │   ├── generator.py        # SUMO 배치 실행기
│   │   ├── parser.py           # XML → tensor 파서
│   │   └── graph.py            # 충전소 그래프 구축
│   ├── sumo_dataset.py         # PyTorch Dataset
│   └── models/
│       └── graph_wavenet.py    # Graph WaveNet (53K params)
├── scripts/
│   └── train_sumo.py           # 학습 파이프라인
├── report/                     # 진행 보고서
├── results/sumo_demand/        # 학습 결과 (results.json, best_model.pt)
└── todo.md                     # 이 파일
```

---

## 성능 목표

| 지표 | 목표 | 비고 |
|---|---|---|
| R² (Global, total_energy) | ≥ 0.5 | v1: -602,350 → 양수로 전환이 최우선 |
| RMSE (log scale) | Naive 대비 20%↓ | log1p 변환 후 비교 |
| 충전 유무 정확도 | ≥ 0.85 F1 | 이진 분류 관점 보조 평가 |
| Learned adj 해석성 | 정성적 확인 | 지리적 근접 + 교통 흐름 반영 여부 |
