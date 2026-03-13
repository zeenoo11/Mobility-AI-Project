# SDV 온디바이스 환경을 위한 지식 증류 기반 멀티모달 배터리 에너지 최적화 실험 계획서

본 실험 계획서는 `연구계획서2.md`에서 제안한 **"멀티모달 데이터(CAN+BMS) 융합"**, **"주파수 도메인 지식 증류(SDKD/FDKD)"**, 그리고 **"SHAP 기반 인과관계 해석"**을 실증하기 위한 구체적인 실험 및 검증 로드맵입니다.

---

## 1. 실험 목적
1. **멀티모달 융합 성능 검증:** CAN 데이터(고주파)와 BMS 데이터(저주파) 융합 모델이 단일 모달(BMS 전용) 모델 대비 배터리 상태(급격한 전압 강하 및 온도 상승) 예측 정확도에서 우수함을 증명.
2. **지식 증류(KD) 효용성 검증:** 무거운 교사(Teacher) 모델의 스펙트럼 지식을 초경량 학생(Student) 모델(DLinear)로 전이하여, 정확도 손실을 최소화하면서 온디바이스 구동 가능한 수준의 연산량(FLOPs) 감축 달성.
3. **인과관계 설명력(XAI) 입증:** SHAP 기법을 통해 극한의 배터리 소모 구간에서 운전자의 특정 제어 패턴(원인)이 배터리 상태(결과)에 미친 기여도를 정량적으로 도출.

---

## 2. 데이터셋 구성 및 전처리 (Data Pipeline)

### 2.1. 데이터셋 확보
* **제어 데이터 (Cause):** ORNL ROAD Dataset (가속 페달 개도량, 브레이크 압력, 조향각, RPM 등) - **수 ms 단위 샘플링**
* **상태 데이터 (Effect):** McMaster University EV Dataset (배터리 팩 전압, 셀 전류, 셀 온도 등) - **수 초~분 단위 샘플링**

### 2.2. 물리적 정규성 기반 전처리
* **시점 동기화 (Time Synchronization):** 샘플링 주파수가 다른 두 데이터셋을 정렬하기 위해 **동적 시간 워핑(DTW)** 및 선형 보간법(Linear Interpolation) 적용. (기준 해상도: 1초 단위로 통합)
* **스펙트럼 노이즈 필터링:** 고속 푸리에 변환(FFT)을 활용하여 센서 결측치나 비물리적인 고주파 노이즈(Sensor Glitch)를 제거하여 데이터의 에너지 스펙트럼 분포(ESD) 안정화.

---

## 3. 실험 모델 아키텍처 및 지식 증류 설계

### 3.1. Teacher 모델 구성 (High-Capacity Multimodal Model)
* **CAN 인코더 (High-pass Filter):** 1D-CNN을 사용하여 급가속/급제동과 같은 국소적인(Local) 고주파 변동성 추출.
* **BMS 인코더 (Low-pass Filter):** Transformer 기반 Self-Attention을 사용하여 배터리 열화 및 방전의 전반적인(Global) 저주파 추세 추출.
* **융합층 (Fusion Layer):** Cross-Attention 메커니즘을 통해 CAN의 특정 패턴이 BMS의 상태 변화에 미치는 영향을 매핑하여 잠재 표현(Latent Representation) 생성.

### 3.2. Student 모델 구성 (Lightweight On-device Model)
* **아키텍처:** 파라미터 수천 개 수준의 초경량 구조인 **DLinear** (또는 경량화된 iTransformer) 적용. 트렌드(Trend)와 잔차(Residual) 분해를 통해 시계열 예측.

### 3.3. 지식 증류 (Knowledge Distillation) 매커니즘
* **적용 기법:** **SDKD (Spectral Decoupled KD) 및 FDKD (Frequency Domain KD)**
* **손실 함수 (Loss Function):**
  * $L_{task}$: 실제 배터리 상태값과의 MSE Loss (과제 손실)
  * $L_{freq}$: 교사와 학생 모델의 예측 결과를 FFT로 변환한 뒤, **주파수 스펙트럼(Log-magnitude)의 차이를 최소화**하는 손실.
  * $L_{align}$: 교사 모델의 다중 모달리티 결합 특징맵(Hidden States)을 학생 모델의 잠재 공간에 맞추는 투영(Projection) 손실.

---

## 4. 단계별 실험 및 검증 시나리오

### Phase 1: 베이스라인 구축 및 Teacher 모델 학습
* **비교군 세팅 (Baselines):**
  * 단일 모달 모델 1 (BMS 데이터 + LSTM)
  * 단일 모달 모델 2 (BMS 데이터 + PatchTST)
* **실험 방법:** Teacher 모델(CAN+BMS 융합)을 학습시키고, 단일 모달 베이스라인 모델들과 정확도(MAE, RMSE) 지표 비교.
* **목표:** 멀티모달 접근법이 극한 주행 상황(급방전 구간)에서 단일 모달보다 오차가 적음을 확인.

### Phase 2: 주파수 도메인 지식 증류(FDKD) 및 Student 모델 평가
* **실험 방법:** Phase 1에서 학습된 Teacher 모델을 기반으로 DLinear(Student) 모델에 FDKD를 적용하여 증류 학습. 일반적인 KD(단순 Soft Label 학습)를 적용한 모델과 성능 비교.
* **온디바이스 성능 측정:** 데스크톱 GPU 환경과 타겟 엣지 디바이스(예: NVIDIA Jetson Nano 또는 Raspberry Pi)에서의 추론 지연시간(Inference Latency) 및 메모리 사용량 측정.
* **목표:** 일반 KD 대비 주파수 도메인 증류(SDKD)가 성능 저하를 방어하며, 추론 속도 면에서 차량용 제어기 탑재 기준(예: <50ms)을 충족함을 검증.

### Phase 3: SHAP 기반 인과성 분석 (XAI)
* **실험 방법:** 학습이 완료된 융합 모델(Student)에 Kernel SHAP 또는 Deep SHAP 알고리즘 적용.
* **분석 시나리오:** 테스트 셋 중 배터리 소모가 극심한 특정 시점(Window)을 추출하여, 해당 시점의 예측값에 가장 큰 긍정적/부정적 영향을 미친 CAN 변수(예: 페달 개도량 80% 이상 지속 시간)의 기여도 점수(SHAP Value) 시각화.
* **목표:** "운전자의 조작(원인)이 배터리 소모(결과)에 미치는 영향"을 정량적 지표로 설명 가능함을 증명 (에코 드라이빙 가이드의 근거 데이터 확보).

---

## 5. 성능 평가 지표 (Evaluation Metrics)

| 평가 항목 | 평가지표 | 세부 설명 | 목표 / 성공 기준 |
| :--- | :--- | :--- | :--- |
| **전체 예측 정확도** | MSE, MAE | 전체 시계열 구간에 대한 배터리 전압/온도 예측 오차 | Baseline 대비 오차 15% 이상 감소 |
| **극한 상황 예측력** | **MAPE** (Mean Absolute Percentage Error) | 배터리 소모가 가장 극심한 상위 5% 구간에서의 예측 오차율 | Baseline 대비 극한 구간 오차 20% 이상 감소 |
| **이상 징후 탐지율** | **HR** (Hit Rate) | 임계치를 초과하는 온도/전압 변동 시점(Time)을 허용 오차 내에 맞춘 비율 | Hit Rate 85% 이상 달성 |
| **온디바이스 효율성** | Parameters, FLOPs, Latency | 모델 크기 및 엣지 디바이스 기준 1회 추론에 걸리는 시간 | Teacher 대비 파라미터 90% 이상 감축, 지연시간 < 50ms |

---

## 6. 추진 일정 (6개월 기준 예시)

* **1~2개월차 (Data & Baseline):** CAN 및 BMS 데이터셋 전처리, 주파수 필터링 적용, Baseline 모델(LSTM, PatchTST) 학습 및 성능 측정.
* **3개월차 (Teacher Model):** 1D-CNN + Transformer 융합 Teacher 모델 설계 및 학습, 멀티모달 상호작용 검증.
* **4개월차 (KD & Student Model):** FDKD/SDKD 지식 증류 파이프라인 구축, DLinear Student 모델 학습.
* **5개월차 (On-device & XAI):** 엣지 디바이스 환경에서의 Inference 성능 측정, SHAP 기법을 통한 극한 구간 인과관계 시각화 분석.
* **6개월차 (Analysis & Report):** 최종 성능 평가 (MAE, HR 등 통합 분석), 결과 정리 및 학회/저널 논문 초안 작성.

---

## 7. 참조 (References)
1. **[XAI / SHAP]** Explainable AI Techniques for Robust Energy Management (2023)
2. **[SDKD / FDKD]** Frequency-Aligned Knowledge Distillation for Lightweight Spatiotemporal Forecasting (2025)
3. **[DistilTS]** Distilling Time Series Foundation Models for Efficient Forecasting (2026)

---

## 8. 관련 코드 (Appendix: Related Code)
아래 코드는 제안된 실험 계획(멀티모달 융합 Teacher, FDKD 지식 증류, DLinear Student, SHAP 기반 인과성 분석)을 구현하기 위한 PyTorch 기반의 수도코드(Pseudo-code) 및 핵심 로직입니다.

### 8.1. Multimodal Teacher Model (1D-CNN + Transformer)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalTeacher(nn.Module):
    def __init__(self, can_dim, bms_dim, hidden_dim, seq_len):
        super(MultimodalTeacher, self).__init__()
        # CAN 데이터 (고주파 제어 신호): 1D-CNN 인코더
        self.can_cnn = nn.Sequential(
            nn.Conv1d(in_channels=can_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # BMS 데이터 (저주파 상태 신호): Transformer 인코더
        self.bms_proj = nn.Linear(bms_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.bms_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Cross-Attention (CAN -> BMS) 융합
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        # 최종 예측기 (예: 배터리 온도 및 전압 예측)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1) # 단일 출력 또는 Multi-output
        )

    def forward(self, can_seq, bms_seq):
        # CAN 데이터: (Batch, Channels, SeqLen)
        can_feat = self.can_cnn(can_seq.transpose(1, 2)).transpose(1, 2)
        
        # BMS 데이터: (Batch, SeqLen, Channels)
        bms_feat = self.bms_proj(bms_seq)
        bms_feat = self.bms_transformer(bms_feat)
        
        # Cross-Attention 융합 (Query: BMS, Key/Value: CAN)
        fused_feat, _ = self.cross_attn(query=bms_feat, key=can_feat, value=can_feat)
        
        # 시계열의 마지막 Hidden State를 사용하여 예측
        out = self.regressor(fused_feat[:, -1, :])
        return out, fused_feat  # KD를 위해 Hidden State 함께 반환
```

### 8.2. DLinear Student Model
```python
class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.transpose(1, 2)).transpose(1, 2)
        return x

class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinearStudent(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in):
        super(DLinearStudent, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decomp = series_decomp(kernel_size=25)
        
        # Trend & Residual 분해 모델링
        self.Linear_Trend = nn.Linear(seq_len, pred_len)
        self.Linear_Seasonal = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: (Batch, SeqLen, Channels) - 멀티모달 병합 데이터 등 입력
        seasonal_init, trend_init = self.decomp(x)
        
        trend_part = self.Linear_Trend(trend_init.transpose(1, 2)).transpose(1, 2)
        seasonal_part = self.Linear_Seasonal(seasonal_init.transpose(1, 2)).transpose(1, 2)
        
        out = trend_part + seasonal_part
        # Feature 출력을 위해 중간 과정의 정보도 함께 반환
        return out[:, -1, :], out
```

### 8.3. Frequency Domain Knowledge Distillation (FDKD) Loss
```python
def freq_kd_loss(student_pred, teacher_pred):
    """
    고속 푸리에 변환(FFT)을 사용하여 주파수 대역에서의 예측 분포를 정렬하는 손실 함수
    """
    # 실수부 FFT 적용
    student_fft = torch.fft.rfft(student_pred, dim=1)
    teacher_fft = torch.fft.rfft(teacher_pred, dim=1)
    
    # Log-magnitude 계산
    student_mag = torch.log(torch.abs(student_fft) + 1e-8)
    teacher_mag = torch.log(torch.abs(teacher_fft) + 1e-8)
    
    # 주파수 도메인에서의 MSE 계산
    loss_freq = F.mse_loss(student_mag, teacher_mag)
    return loss_freq

def total_kd_loss(student_out, student_feat, teacher_out, teacher_feat, y_true, alpha=0.7, beta=0.1):
    # 1. 과제 손실 (Supervised Loss)
    l_sup = F.mse_loss(student_out, y_true)
    
    # 2. 주파수 도메인 증류 손실 (FDKD)
    l_freq = freq_kd_loss(student_out, teacher_out)
    
    # 3. 특징맵 정렬 손실 (Feature Alignment / FTA) - 필요 시 차원 맞춤 프로젝션 계층 적용
    # 여기서는 단순 사이즈가 동일하다고 가정한 예시
    if student_feat.shape == teacher_feat.shape:
        l_align = F.mse_loss(student_feat, teacher_feat)
    else:
        l_align = 0.0 # 별도 Projection 처리 필요
        
    return l_sup + alpha * l_freq + beta * l_align
```

### 8.4. SHAP을 이용한 인과성 분석 (XAI)
```python
import shap
import numpy as np

# SHAP 설명자 래퍼 함수: 딥러닝 모델의 입력을 받아 예측을 수행
def model_predict_wrapper(can_data_numpy, bms_data_numpy):
    # Numpy -> Tensor 변환 후 모델 예측
    can_tensor = torch.tensor(can_data_numpy, dtype=torch.float32)
    bms_tensor = torch.tensor(bms_data_numpy, dtype=torch.float32)
    
    with torch.no_grad():
        out, _ = teacher_model(can_tensor, bms_tensor)
    return out.numpy()

# SHAP Explainer 초기화 (예: DeepExplainer 또는 GradientExplainer 활용)
# 단순화를 위해 DeepExplainer 사용 (단일 모달화 된 입력 포맷으로 가정)
# 멀티모달의 경우 리스트 형태로 입력을 넘겨줄 수 있습니다.
background_can = torch.randn(100, 10, 5) # (Batch, SeqLen, CAN_Dim) 백그라운드 데이터
background_bms = torch.randn(100, 10, 3) # (Batch, SeqLen, BMS_Dim) 백그라운드 데이터

# 모델과 백그라운드 데이터를 Explainer에 전달
explainer = shap.DeepExplainer(teacher_model, [background_can, background_bms])

# 특정 이상(Extreme) 구간 테스트 데이터 추출
test_can = torch.randn(5, 10, 5) # (Batch, SeqLen, CAN_Dim)
test_bms = torch.randn(5, 10, 3) # (Batch, SeqLen, BMS_Dim)

# SHAP Value 계산
shap_values = explainer.shap_values([test_can, test_bms])

# 시각화 (예: CAN 데이터의 첫 번째 샘플에 대한 기여도)
# shap.summary_plot(shap_values[0], test_can.numpy())
```
