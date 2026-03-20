# SUMO Graph WaveNet 충전 수요 예측 — 진행 보고서

**작성일:** 2026-03-20 (최종 업데이트)
**상태:** Phase 2 — 배치 시뮬레이션 완료, 데이터 파싱 완료, 학습 진행 중

---

## 1. 연구 방향 전환

기존 Cross-Attention 기반 멀티모달 융합에서 **그래프 기반 시공간 모델링**으로 연구 테마를 확장한다.

### 두 가지 적용 축

| 축 | 범위 | 데이터 | 모델 |
|---|---|---|---|
| **축 1 (거시적)** | 충전소별 시공간 충전 수요 예측 | SUMO 시뮬레이션 합성 데이터 | Graph WaveNet |
| **축 2 (미시적)** | 차량별 에너지 소모 + SoH 추정 | BMW i3 실주행 데이터 | Graph WaveNet (동일 아키텍처) |

현재 보고서는 **축 1 (SUMO 충전 수요 예측)** 프로토타입 진행 상황을 기록한다.

---

## 2. SUMO 시뮬레이션 환경

| 항목 | 값 |
|---|---|
| SUMO 버전 | v1.26.0 (Eclipse SUMO) |
| 도로 네트워크 | 베를린 OSM 기반 (6,460 edges, 3,037 nodes) |
| 차량 경로 | 실제 교통 카운트 데이터 기반 샘플링 (32,488대 정의, ~5,100대 삽입/2시간) |
| 시뮬레이션 시간 | 06:00–08:00 (아침 피크 2시간) |
| 충전소 | 53개 (75kW DC, 효율 95%) |
| 주차장 | 1,899개 (충전소의 3%가 주차장에 배치) |
| 1회 시뮬레이션 소요 | ~23–25초 |

### 시뮬레이션 출력 (1회 기준)

- `tripinfos.xml`: 차량별 에너지 소모, 배터리 잔량, 경로 길이, 대기 시간
- `charging_output.xml`: 충전소별 초 단위 충전 이력 (에너지, SOC, 전력)
- `stops_output.xml`: 충전 정차 이벤트 (충전소 ID, 시작/종료 시간, 대기 시간)

### 테스트 시뮬레이션 결과 (기본 파라미터)

| 통계 | 값 |
|---|---|
| 삽입 차량 | 5,594대 |
| 완주 차량 | 5,097대 |
| 충전 이벤트 | 97건 |
| 활성 충전소 | 22/53개 |
| 총 충전 에너지 | 152,831 Wh |
| 배터리 고갈 차량 | 0대 |
| 평균 경로 길이 | 6,250m |
| 평균 속도 | 10.86 m/s |

---

## 3. 데이터 파이프라인

### 3.1 배치 시뮬레이션 (`sumo/data_generator.py`)

파라미터 그리드를 조합하여 다수의 시뮬레이션을 자동 실행한다.

**변형 파라미터:**

| 파라미터 | 값 | 의미 |
|---|---|---|
| `battery_mean` | 3000, 5000, 8000 (Wh) | 초기 배터리 용량 평균 |
| `battery_std` | 2000, 3000 (Wh) | 용량 분포 표준편차 |
| `need_to_charge_level` | 0.05, 0.10, 0.20 | 충전 탐색 시작 SOC 임계값 |
| `saturated_charge_level` | 0.3, 0.5 | 충전 완료 목표 SOC |
| `seed` | 42, 123, 456 | 랜덤 시드 |

**전체 조합: 3 × 2 × 3 × 2 × 3 = 108회**

### 3.2 출력 파서 (`sumo/parse_outputs.py`)

XML 출력을 5분 단위 시간 구간으로 집계하여 (53, 24, 6) 텐서로 변환한다.

**집계 피처 (충전소별, 5분 구간별):**

| 인덱스 | 피처 | 설명 |
|---|---|---|
| 0 | `num_arrivals` | 충전 도착 차량 수 |
| 1 | `avg_charging_dur` | 평균 충전 시간 (초) |
| 2 | `total_energy` | 총 충전 에너지 (Wh) |
| 3 | `max_blocked_dur` | 최대 대기 시간 (초) |
| 4 | `utilization` | 이용률 (충전 시간 / 구간 길이) |
| 5 | `avg_soc_arrival` | 도착 시 평균 SOC 추정 |

### 3.3 그래프 구축 (`sumo/graph_builder.py`)

sumolib API로 충전소 좌표를 추출하고, 유클리드 거리 기반 가우시안 커널로 인접 행렬을 생성한다.

| 항목 | 값 |
|---|---|
| 노드 수 | 53 |
| 엣지 수 | 462 |
| 평균 차수 | 8.7 |
| 거리 임계값 | 2.0 km |
| 가우시안 σ | 1.0 km |
| 평균 엣지 가중치 | 0.553 |
| 최대 충전소 간 거리 | 9,568m |
| 연결된 쌍 평균 거리 | 1,101m |

---

## 4. 모델 아키텍처: Graph WaveNet

Graph WaveNet (Wu et al., IJCAI 2019) 기반 시공간 그래프 신경망.

### 구조

```
Input (B, 53, T_in, 6)
    ↓
Input Projection (Linear: 6 → 32)
    ↓
Mixed Adjacency = Predefined Adj + Adaptive Adj (learned embeddings)
    ↓
ST-Conv Block × 6 (2 blocks × 3 layers, dilation = 1, 2, 4)
  ├─ Dilated Causal Conv (시간축, gated activation)
  ├─ Graph Conv (공간축, diffusion order=2)
  ├─ Residual + LayerNorm
  └─ Skip connection
    ↓
Skip Aggregation → GELU
    ↓
Output MLP (32 → 32 → 1) per node
    ↓
Output (B, 53, T_pred, 1)
```

### 모델 사양

| 항목 | 값 |
|---|---|
| 총 파라미터 | 53,313 |
| Hidden dim | 32 |
| ST-Conv blocks | 6 (2 × 3 layers) |
| Dilation | 1, 2, 4 (반복 2회) |
| Graph diffusion order | 2 |
| Adaptive adj embedding dim | 16 |
| Activation | Gated (tanh ⊙ sigmoid) + GELU |
| Normalization | LayerNorm (residual 후) |

### Adaptive Adjacency Matrix

사전 정의된 거리 기반 인접 행렬에 더해, 학습 가능한 노드 임베딩(src, tgt)으로 데이터 기반 공간 의존성을 자동 학습한다:

```
adj_adaptive = softmax(ReLU(E_src × E_tgt^T))
adj_mixed = adj_predefined + adj_adaptive
```

---

## 5. 학습 설정

| 항목 | 값 |
|---|---|
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | Linear Warmup (5 epochs) + Cosine Annealing |
| Loss | MSE |
| Gradient clipping | max_norm=1.0 |
| Early stopping | patience=15 |
| Input window | 12 time steps (1시간) |
| Prediction horizon | 1 time step (5분) |
| Batch size | 16 |
| Data split | 70% train / 15% val / 15% test (시뮬레이션 단위) |

### 평가 메트릭

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (충전소별 계산 후 평균)

---

## 6. End-to-End 검증 (더미 데이터)

20개 더미 시뮬레이션 (랜덤 데이터)으로 전체 파이프라인 동작을 검증하였다.

| 항목 | 결과 |
|---|---|
| 학습 수렴 | 33 epoch에서 early stop (정상) |
| 더미 데이터 R² | ~0 (랜덤 데이터이므로 정상) |
| Forward pass | 정상 (input → output shape 일치) |
| Gradient flow | 정상 (loss 감소 확인) |
| 체크포인트 저장/로드 | 정상 |
| Learned adjacency 추출 | 정상 (53 × 53) |

---

## 7. 구현 파일 목록

```
Mobility-AI-Project/
├── sumo/
│   ├── data_generator.py       # SUMO 배치 실행기 (108 조합)
│   ├── parse_outputs.py        # XML → (53, 24, 6) 텐서 파서
│   ├── graph_builder.py        # 충전소 그래프 구축 (sumolib)
│   ├── train_demand.py         # 학습 파이프라인 (warmup+cosine, early stop)
│   ├── graph_data/             # 사전 계산된 그래프 데이터
│   │   ├── adj.npy             # (53, 53) 인접 행렬
│   │   ├── edge_index.npy      # (2, 462) COO 엣지
│   │   ├── positions.npy       # (53, 2) 충전소 좌표
│   │   └── node_features.npy   # (53, 4) 정적 노드 피처
│   └── sim_outputs/            # 시뮬레이션 출력 디렉토리
│       └── run_9999/           # 테스트 시뮬레이션 결과
├── src/
│   ├── sumo_dataset.py         # PyTorch Dataset (sliding window)
│   └── models/
│       ├── __init__.py
│       └── graph_wavenet.py    # Graph WaveNet 모델 (53K params)
└── results/
    └── sumo_demand/            # 학습 결과 저장 디렉토리
```

---

## 8. 배치 시뮬레이션 결과 (Phase 2)

### 8.1 실행 요약

| 항목 | 값 |
|---|---|
| 실행 완료 | 109/109회 (108 조합 + 1 테스트) |
| 실패 | 0회 |
| 1회 평균 소요 시간 | ~35초 |
| 총 소요 시간 | ~64분 |
| 배터리 고갈 차량 | 0대 (전 시뮬레이션) |

### 8.2 파싱된 데이터셋

| 항목 | 값 |
|---|---|
| 텐서 Shape | (109, 53, 24, 6) |
| 의미 | (시뮬레이션, 충전소, 5분 구간, 피처) |
| 파일 | `sim_outputs/station_features.npy` + `run_metadata.json` |

### 8.3 데이터 통계

| 피처 | 평균 | 최대 | Non-zero 비율 | Non-zero 평균 |
|---|---|---|---|---|
| num_arrivals | 0.079 | 5.0 | 6.28% | 1.26 |
| avg_charging_dur (s) | 5.02 | 320.5 | 6.28% | 79.9 |
| total_energy (Wh) | 125.5 | 12,686 | 6.33% | 1,983 |
| max_blocked_dur (s) | 0.63 | 227.0 | 0.99% | 63.7 |
| utilization | 0.021 | 1.0 | 6.28% | 0.33 |
| avg_soc_arrival | 0.060 | 0.99 | 6.28% | 0.95 |

### 8.4 주요 관찰

- **높은 희소성**: 충전 이벤트가 발생하는 셀은 전체의 ~6%에 불과
- **충전소 활용 편차**: 시뮬레이션당 평균 15.7/53개 충전소만 활성 (30%)
- **충전 이벤트 범위**: 0~257건/run (파라미터에 따라 크게 변동)
- **에너지 범위**: 0~422,611 Wh/run (배터리 용량/충전 임계값에 강하게 의존)
- **대기 시간 희귀**: blocked duration이 발생하는 셀은 ~1%

### 8.5 데이터 희소성 대응 전략

1. **활성 충전소 필터링**: 전체 시뮬레이션에서 한 번이라도 활성화된 충전소만 사용
2. **Log 변환**: total_energy, avg_charging_dur에 log(1+x) 적용
3. **피처 정규화**: 시뮬레이션 단위 min-max or z-score 정규화
4. **예측 타겟 선정**: num_arrivals (이산) 또는 total_energy (연속) 중 선택

---

## 9. 다음 단계

1. ~~배치 시뮬레이션 실행~~ ✅
2. ~~데이터 파싱~~ ✅
3. **Graph WaveNet 학습**: 실제 데이터로 학습 (진행 중)
4. **베이스라인 비교**: ARIMA, LSTM, GCN+GRU, DCRNN vs Graph WaveNet
5. **해석 가능성 분석**: Learned adjacency matrix 시각화, 충전소 클러스터 분석
6. **축 2 확장**: BMW i3 데이터에 동일 Graph WaveNet 적용 (에너지 소모 + SoH)
