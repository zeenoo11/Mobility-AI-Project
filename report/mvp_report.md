# MVP 실험 보고서: Cross-Attention 기반 멀티모달 융합 EV 에너지 소모 예측

**실험 일자:** 2026-03-16
**실험 환경:** Python 3.11, PyTorch 2.10.0, CPU

---

## 1. 실험 목적

BMW i3 실주행 데이터를 활용하여, 주행 데이터(CAN)와 배터리 상태 데이터(BMS)의 멀티모달 융합이 에너지 소모율(Wh/km) 예측 성능에 미치는 영향을 검증한다. 3가지 모델 아키텍처를 비교하여 Cross-Attention 기반 융합의 유효성을 확인한다.

---

## 2. 데이터 요약

| 항목 | 값 |
|---|---|
| 데이터셋 | BMW i3 Real Driving Cycles (TripB 시리즈) |
| 유효 트립 수 | 37개 (TripB38 컬럼 누락으로 제외) |
| Train / Val / Test | 26 / 6 / 5 트립 |
| 윈도우 크기 | 60초 (stride 10초) |
| 총 윈도우 수 | 46,719개 (Train: 35,847 / Val: 5,516 / Test: 5,356) |
| 주행 피처 (6개) | Velocity, Throttle, Motor Torque, Longitudinal Acceleration, Regenerative Braking, Elevation |
| 배터리 피처 (4개) | Battery Voltage, Battery Current, Battery Temperature, SoC |

### 타겟 변수: 에너지 소모율 (Wh/km)

| 통계량 | 값 |
|---|---|
| 평균 | 256.3 Wh/km |
| 표준편차 | 207.2 Wh/km |
| 최솟값 | 3.5 Wh/km |
| 최댓값 | 1,499.8 Wh/km |

> **전처리 참고:** 정차/극저속 구간(총 이동거리 < 10m)과 비현실적 값(< 0.5 또는 > 1,500 Wh/km)은 NaN 처리 후 제외. Min-Max 정규화는 학습 세트 기준으로 적합(fit)하여 검증/테스트에 적용.

---

## 3. 모델 아키텍처 비교

### 3-1. LSTMBaseline (베이스라인)

- **입력:** BMS 피처만 사용 (배터리 4개 변수)
- **구조:** 2-layer LSTM (hidden=64) → FC(64→32→1)
- **파라미터:** 53,313개
- **목적:** 주행 데이터 없이 배터리 데이터만으로의 예측 한계 확인

### 3-2. CNNLSTMConcat (단순 융합 베이스라인)

- **입력:** 주행(6) + 배터리(4) 양쪽 모두 사용
- **구조:**
  - 주행 브랜치: 1D-CNN (kernel=3, channels=32, 2층) → Global Average Pooling
  - 배터리 브랜치: 2-layer LSTM (hidden=64)
  - 융합: Concatenation → FC(96→64→32→1)
- **파라미터:** 63,361개
- **목적:** 단순 연결(concat) 방식의 멀티모달 융합 성능 기준선

### 3-3. CrossAttentionNet (제안 모델)

- **입력:** 주행(6) + 배터리(4)
- **구조:**
  - 주행 인코더: Multi-scale 1D-CNN (커널 3, 5, 7 병렬) → 48채널
  - 배터리 인코더: 2-layer Bidirectional LSTM (hidden=64) → 128차원
  - 양방향 Cross-Attention (4 heads, embed_dim=64):
    - BMS→Driving: Query=BMS, Key/Value=Driving
    - Driving→BMS: Query=Driving, Key/Value=BMS
  - LayerNorm → FC(128→64→32→1)
- **파라미터:** 192,049개
- **목적:** 교차모달 어텐션을 통한 주행-배터리 인과관계 학습

---

## 4. 학습 설정

| 항목 | 설정 |
|---|---|
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Loss | MSELoss |
| Gradient Clipping | max_norm=1.0 |
| Early Stopping | patience=5 (validation loss 기준) |
| 최대 Epoch | 30 |
| Batch Size | 64 |

---

## 5. 실험 결과

### 5-1. 테스트셋 성능 비교

| 모델 | RMSE (Wh/km) | MAE (Wh/km) | R² | 파라미터 수 | 학습 Epoch |
|---|---|---|---|---|---|
| **LSTMBaseline** | 159.90 | 117.25 | 0.3802 | 53,313 | 10 (early stop) |
| **CNNLSTMConcat** | **47.28** | **31.37** | **0.9458** | 63,361 | 15 (early stop) |
| **CrossAttentionNet** | 56.35 | 35.32 | 0.9230 | 192,049 | 12 (early stop) |

### 5-2. 베이스라인 대비 개선율

| 비교 | RMSE 개선율 | MAE 개선율 | R² 변화 |
|---|---|---|---|
| LSTMBaseline → CNNLSTMConcat | -70.4% | -73.2% | +0.5656 |
| LSTMBaseline → CrossAttentionNet | -64.8% | -69.9% | +0.5428 |
| CNNLSTMConcat → CrossAttentionNet | +19.2% (열위) | +12.6% (열위) | -0.0228 |

---

## 6. 학습 곡선 분석

![Training Curves](../results/figures/training_curves.png)

- **LSTMBaseline:** Train loss는 지속 감소하나 Val loss가 epoch 3 이후 상승 → 과적합 경향. BMS 단일 모달리티의 정보량 한계.
- **CNNLSTMConcat:** 안정적으로 수렴. Train/Val loss 모두 일관되게 감소하며 epoch 10 근처에서 최저.
- **CrossAttentionNet:** 초기 3 epoch에서 급격한 loss 감소 후 안정. Val loss의 변동이 다소 크나 전반적으로 수렴.

---

## 7. Cross-Attention 가중치 분석

![Attention Heatmap](../results/figures/attention_heatmap.png)

### BMS→Driving Attention (좌측)

- BMS가 Driving 시퀀스에 질의할 때, 초기 타임스텝(0~3)에서는 낮은 가중치, 이후 타임스텝에서는 균일하게 높은 가중치 패턴을 보임.
- 이는 배터리 상태가 "현재 및 최근 과거"의 주행 패턴에 주로 영향받음을 시사.

### Driving→BMS Attention (우측)

- Driving이 BMS에 질의할 때, 초기 타임스텝(0~3)의 BMS 상태에 집중하는 경향.
- 윈도우 초반의 배터리 초기 상태(SoC, 전압 수준)가 주행 에너지 소모 예측에 중요한 컨텍스트를 제공함을 의미.

---

## 8. 핵심 발견 및 논의

### (1) 멀티모달 융합의 유효성 확인

BMS 단독(R²=0.38) → 멀티모달 융합(R²≥0.92)으로 **R²가 0.54 이상 향상**. 주행 패턴 정보가 에너지 소모 예측에 결정적 역할을 함을 실증.

### (2) CNNLSTMConcat이 CrossAttentionNet보다 우수한 원인 분석

MVP 단계에서 CrossAttentionNet이 CNNLSTMConcat 대비 다소 열위인 원인:

- **파라미터 비율:** CrossAttentionNet(192K)이 CNNLSTMConcat(63K)의 약 3배. 현재 데이터 규모(35K 학습 샘플)에서 과적합 리스크.
- **학습 부족:** Early stopping으로 12 epoch만 학습. 더 복잡한 모델은 수렴에 더 많은 epoch이 필요.
- **하이퍼파라미터 미조정:** head 수(현재 4), attention dimension(64), dropout 등 최적화 미수행.
- **스케줄러 차이:** CosineAnnealingLR 대신 ReduceLROnPlateau 사용 중.

### (3) 성능 목표 대비 달성도

| 지표 | 목표 | 현재 달성 | 상태 |
|---|---|---|---|
| RMSE 베이스라인 대비 15% 감소 | < 135.9 | 56.4 (64.8% 감소) | **달성** |
| R² (에너지 소모율) | ≥ 0.90 | 0.923 | **달성** |
| Cross-Attention이 Concat 대비 개선 | 유의미한 개선 | 미달 (Concat이 우위) | **미달성** |

---

## 9. 다음 단계 (Phase 3~4 개선 방향)

### 즉시 적용 가능

1. **학습 강화:** Epoch 증가 (30→100), CosineAnnealingLR 스케줄러 적용
2. **하이퍼파라미터 튜닝:**
   - `n_heads`: {4, 8}
   - `lstm_hidden`: {64, 128}
   - `cnn_out`: {48, 96}
   - `dropout`: {0.1, 0.2, 0.3}
3. **데이터 증강:** TripA 시리즈 추가 투입 (현재 TripB만 사용)
4. **Multi-task Learning:** BSI(배터리 스트레스 지수) 보조 타겟 추가

### Phase 4 (Ablation & 해석)

5. **Ablation Study:** Self-Attention vs Cross-Attention, 단방향 vs 양방향 비교
6. **변수별 Attention 분석:** 시간축이 아닌 피처 축으로 Attention 집계
7. **운전 패턴 클러스터링:** k-means 기반 공격적/에코 운전 유형 분류 후 클러스터별 인과 맵 비교

---

## 10. 파일 목록

| 파일 | 설명 |
|---|---|
| `results/final_results.json` | 전체 실험 결과 (메트릭, 설정, 데이터 통계) |
| `results/figures/training_curves.png` | 3개 모델 학습 곡선 |
| `results/figures/attention_heatmap.png` | Cross-Attention 가중치 히트맵 |
| `results/checkpoints/LSTMBaseline_best.pt` | LSTMBaseline 최적 체크포인트 |
| `results/checkpoints/CNNLSTMConcat_best.pt` | CNNLSTMConcat 최적 체크포인트 |
| `results/checkpoints/CrossAttentionNet_best.pt` | CrossAttentionNet 최적 체크포인트 |

---

## 11. 문헌 비교 분석

### 11-1. 관련 연구 요약

| 논문 | 연도/학회 | 구조 | 타겟 | Best R² | 비고 |
|------|----------|------|------|---------|------|
| Feng et al. [1] | 2024 / *Energy* | LSTM + Transformer | SOC/에너지 | - | MAPE 4.63%, 운전 스타일 반영 |
| DLLT [2] | 2024 / *PLOS ONE* | Dual-Layer LSTM + Transformer | 연료/전력/가속도 | 0.9945 (연료) | 모드 분류→회귀 2단계 |
| IFS-Former [3] | 2025 / *Technologies (MDPI)* | Transformer + Transfer Learning | Wh/km | 0.97 | 32MB 경량, 837대 사전학습 |
| KD Cross-Modal [4] | 2024 / *Complex & Intell. Systems* | Transformer→CNN 증류 | SOH | >0.99 | 교차모달 KD |
| ST-MAN [5] | 2024 / *Heliyon* | CNN + ST-Attention + LSTM | RUL | - | 멀티모달 어텐션 융합 |
| Puri [6] | 2022 / *SAE Technical Paper* | Classical ML | SOC | - | BMW i3 동일 데이터셋 사용 |

### 11-2. 본 연구의 위치

- **CAN + BMS 듀얼 인코더 + Cross-Attention으로 에너지 소모율(Wh/km)을 예측한 연구는 부재.** 이것이 본 연구의 핵심 연구 갭.
- LSTM-Transformer 하이브리드가 현재 SOTA이나 단일 스트림 구조 → 멀티모달 융합이 아님.
- Knowledge Distillation 기반 교차모달 학습은 SOH에서 입증되었으나 에너지 소모 예측에는 미적용.
- IFS-Former(R²=0.97)가 현재 BEV 에너지 예측 최고 수준이나, 837대 대규모 사전학습 기반이므로 직접 비교는 부적절.

### 11-3. 현재 결과 대비 문헌 수준 평가

| 지표 | 본 연구 (CNNLSTMConcat) | 본 연구 (CrossAttentionNet) | 문헌 수준 |
|------|------------------------|---------------------------|----------|
| R² | 0.9458 | 0.9230 | 0.95~0.99 (대규모 데이터 기준) |
| MAPE | 미측정 | 미측정 | 4~7% (Feng et al.) |

현재 37개 트립(35K 학습 샘플) 규모에서 R²≥0.92는 합리적 수준이나, CrossAttentionNet의 성능이 CNNLSTMConcat보다 낮은 점은 개선이 필요.

---

## 12. CrossAttentionNet 열위 원인의 근본 분석

### 12-1. 학습 설정 문제

| 문제 | 현재 설정 | 영향 |
|------|----------|------|
| **에포크 부족** | `EPOCHS=30`, early stop으로 12 epoch만 학습 | 192K 파라미터 모델이 수렴하기엔 턱없이 부족 |
| **Early Stopping patience 과소** | `PATIENCE=5` | Val loss의 자연 변동에 의해 조기 중단 |
| **스케줄러 충돌** | `ReduceLROnPlateau(patience=3)` + Early Stop patience=5 | LR 1회 감소 후 바로 early stop 발동 |
| **Warmup 부재** | lr=1e-3에서 즉시 시작 | Attention 가중치 초기화 불안정 → 초기 gradient exploding |

### 12-2. 아키텍처 문제

| 문제 | 현재 구현 | 개선 방향 |
|------|----------|----------|
| **Residual Connection 부재** | Cross-attention 출력만 사용 | `output = attn_output + original_proj` 잔차 연결 추가 |
| **Attention Head 수 부족** | `n_heads=4` | CLAUDE.md 설계대로 8 heads |
| **LayerNorm 위치** | Post-LN (융합 후) | Pre-LN 패턴 (attention 전) → 학습 안정성 향상 |
| **인코더 깊이 불균형** | CNN 1층 / LSTM 2층 | CNN에 두 번째 Multi-scale 블록 추가 고려 |

### 12-3. 데이터 규모 대비 파라미터 비율

| 모델 | 파라미터 | 학습 샘플 | 샘플/파라미터 비율 |
|------|---------|----------|-----------------|
| CNNLSTMConcat | 63,361 | 35,847 | 0.566 |
| CrossAttentionNet | 192,049 | 35,847 | 0.187 |

CrossAttentionNet의 샘플/파라미터 비율이 CNNLSTMConcat의 약 1/3 수준 → **과적합 리스크가 구조적으로 높음.** Dropout, weight decay 강화와 함께 데이터 증강(TripA 추가)이 필요.

---

## 13. 개선 실험 계획

위 분석에 기반하여 다음 개선을 순차 적용하고 재실험한다:

1. **학습 설정 강화**: EPOCHS 100, PATIENCE 15, CosineAnnealingWarmRestarts + Linear Warmup (5 epoch)
2. **아키텍처 수정**: Residual connection, Pre-LayerNorm, n_heads=8
3. **하이퍼파라미터 서치**: lstm_hidden {64, 128}, n_heads {4, 8}, dropout {0.1, 0.2, 0.3}, cnn_out {48, 96}

---

## 참고문헌

[1] Z. Feng, J. Zhang, H. Jiang, X. Yao, Y. Qian, H. Zhang, "Energy consumption prediction strategy for electric vehicle based on LSTM-transformer framework," *Energy*, vol. 302, 2024.

[2] "DLLT: A dual-layer LSTM-transformer model for real-time energy and dynamics prediction in plug-in hybrid electric vehicles," *PLOS ONE*, 2024.

[3] "Prediction of Battery Electric Vehicle Energy Consumption via Pre-Trained Model Under Inconsistent Feature Spaces," *Technologies (MDPI)*, vol. 13, no. 11, 2025.

[4] "A knowledge distillation based cross-modal learning framework for the lithium-ion battery state of health estimation," *Complex & Intelligent Systems*, 2024.

[5] "Remaining useful life prediction of lithium-ion batteries using spatio-temporal multimodal attention networks," *Heliyon*, vol. 10, no. 16, e36236, 2024.

[6] S. Puri, "Utilizing Machine Learning Algorithms in a Data-Driven Approach to the Prediction of Vehicle Battery State of Charge with BMW i3 Datasets," *SAE Technical Paper* 2022-01-5088, 2022.

---

## 부록: 실험 재현

```bash
# 환경 설정
uv sync

# MVP 실험 실행
uv run python mvp_experiment.py

# 하이퍼파라미터 서치 실행
uv run python hparam_search.py
```
