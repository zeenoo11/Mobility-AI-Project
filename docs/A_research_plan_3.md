# 연구계획서 (수정본)

## Cross-Attention 기반 멀티모달 융합을 통한 전기차 운전 패턴-배터리 에너지 소모 인과 해석 연구

_Interpretable Cross-Attention Fusion of Driving Behavior and Battery Response for Causal Energy Consumption Analysis in Electric Vehicles_

---

## 1. 문제 정의 및 연구 필요성

**현상:** 전기차(EV) 보급 확대에 따라 주행 거리를 좌우하는 배터리 효율 최적화가 모빌리티 AI의 핵심 과제로 부상하였다. 그러나 동일한 경로를 주행하더라도 운전자의 가속/감속 습관에 따라 배터리 에너지 소모량은 최대 30% 이상 차이가 발생한다.

**문제점:** 기존 연구는 (1) 주행 최적화 측면에서 GPS/속도 등 외부 지표만 활용하거나, (2) 배터리 상태 예측 측면에서 BMS 데이터만 단독 사용하여, 두 시스템 간의 상호작용을 통합적으로 규명하지 못하였다. 특히 **어떤 운전 조작 패턴이 배터리 소모를 유발하는지, 그 인과관계를 정량적으로 해석한 연구는 전무**하다.

**연구 필요성:** 운전자의 차량 제어 신호(원인)와 배터리 상태 변화(결과)를 멀티모달로 동시 학습하되, 단순 예측 정확도 향상이 아닌 **Cross-Attention의 가중치 해석을 통해 교차모달 인과 관계를 정량화**하고, 이를 기반으로 운전자 맞춤형 에너지 관리 전략을 제안하는 해석 가능한(Interpretable) AI 모델의 도입이 필수적이다.

---

## 2. 선행연구 분석 및 한계점

### 선행연구 1 (주행 최적화)

> Leech & Yoon (2025), "Model-Based Deep Reinforcement Learning for Energy Efficient Routing of a Connected and Automated Vehicle", _Sustainability_, 17(13), 5727.

**한계점:** AlphaGo Zero 기반 심층 강화학습을 통해 에너지 효율 경로를 제안하였으나, 시뮬레이션 환경에서만 검증되었으며, 실시간 배터리 내부 상태(전압, 전류, 셀 온도)를 반영하지 못하였다. 주행 경로 최적화에만 집중하여 운전 조작과 배터리 응답 간의 직접적 상호작용은 모델링하지 않았다.

### 선행연구 2 (멀티모달 배터리 SOH)

> Liu et al. (2025), "Multi-modal framework for battery state of health evaluation using open-source electric vehicle data", _Nature Communications_, 16, 1137.

**한계점:** 300대 EV의 3년간 운행 데이터를 활용하여 멀티모달 SOH 추정을 수행한 선구적 연구이나, 포인트 특성(point feature) 기반 융합을 사용하여 주행 패턴의 시간적 동역학을 충분히 반영하지 못하였다. 또한 어떤 운전 습관이 배터리 소모를 유발하는지 인과 해석이 불가능한 블랙박스 모델을 사용하였다.

### 선행연구 3 (CNN-LSTM 하이브리드)

다수의 CNN-LSTM, CNN-Informer 기반 배터리 상태 예측 연구(2024-2025)가 존재하나, 이들은 모두 단일 데이터 소스에 대한 단일 모달 아키텍처이며, 주행-배터리 데이터 결합 시에도 단순 연결(concatenation)이나 선형 결합만을 사용하여 교차모달 상호작용을 명시적으로 모델링하지 못하였다.

### 본 연구의 차별성 (3중 노벨티)

> 🟢 **방법론:** Cross-Attention을 통한 주행-배터리 교차모달 상호작용 명시적 학습 (최초)
>
> 🟢 **분석:** Attention Weight 시각화 기반 교차모달 인과 맵 도출 (정량적 해석)
>
> 🟢 **응용:** 운전 습관 클러스터링 기반 맞춤형 에코 드라이빙 가이드라인 제안

---

## 3. 사용 데이터 및 전처리 계획

### 🔴 `변경` 주요 데이터: BMW i3 Real Driving Cycles Dataset

Steinstraeter, Buberger & Trifonov (2020), Technical University of Munich.
IEEE DataPort, DOI: 10.21227/6jr9-5235. **Open Access** 공개 데이터셋.

| 항목                           | 세부 내용                                                                      |
| ------------------------------ | ------------------------------------------------------------------------------ |
| **수집 차량**                  | BMW i3 (60 Ah), 실제 도로 주행 (독일)                                          |
| **트립 수**                    | 72개 실주행 트립 (여름/겨울 포함, 다양한 도로 조건)                            |
| **주행 모달리티 (12개 변수)**  | 차량 속도, 스로틀(가속 페달), 브레이크 신호, 종방향 가속도, 모터 토크, 고도 등 |
| **배터리 모달리티 (6개 변수)** | 배터리 전압(V), 전류(A), 온도(°C), SoC(%), 배터리 파워(kW), 충방전 상태        |
| **환경 변수 (6개 변수)**       | 외기 온도, 에어컨 파워, 히터 파워, 실내 온도, 난방 회로 데이터                 |
| **데이터 형식**                | .csv / .mat (동기화된 타임스탬프, DTW 불필요)                                  |

**데이터셋 선정 근거:** 동일 차량에서 주행 신호와 배터리 신호가 동시 수집되어 타임스탬프가 완전 동기화되어 있으므로, 이종 데이터셋 결합 시 발생하는 DTW/보간법의 방법론적 한계가 원천적으로 존재하지 않는다. 24개 변수의 풍부한 특성은 주행 모달리티와 배터리 모달리티의 명확한 분리를 가능하게 한다.

### 보조 데이터 (일반화 검증용)

McMaster University LG 18650HG2 Li-ion Battery Data (Mendeley Data, DOI: 10.17632/cp3473x7xv/3): 12개 표준 주행 사이클(UDDS, HWFET, US06 등) × 6개 온도 조건의 배터리 응답 데이터를 활용하여, BMW i3에서 학습한 모델의 일반화 성능을 검증한다.

### 전처리 계획

1. **모달리티 분리:** 24개 변수를 주행 그룹(속도, 스로틀, 브레이크, 가속도, 모터 토크, 고도)과 배터리 그룹(전압, 전류, 온도, SoC, 파워)으로 분리
2. **정규화:** Min-Max 또는 Z-score 정규화로 스케일 통일
3. **윈도우 슬라이싱:** 고정 길이 슬라이딩 윈도우(예: 60초)로 시퀀스 생성
4. **트립 분할:** 72개 트립을 Train(50)/Validation(10)/Test(12)로 분할 (계절 균형 유지)

---

## 4. 방법론 개요

### 4.1 전체 아키텍처: Dual-Encoder Cross-Attention Network

제안 모델은 세 단계로 구성된다: (1) 모달리티별 특징 인코딩, (2) 양방향 Cross-Attention 융합, (3) 예측 및 해석.

### 4.2 특징 인코딩 (Feature Encoding)

**주행 인코더 (1D-CNN):** 가속 페달, 브레이크, 속도, 가속도 등 주행 시계열에 다층 1D Convolution을 적용하여, 급가속/급감속 등 운전 패턴의 공간적·맥락적 특징을 추출한다. 커널 크기를 다양화(3, 5, 7)하여 단기 및 중기 운전 패턴을 동시 포착한다.

**배터리 인코더 (Bi-LSTM):** 전압, 전류, 온도 시계열에 양방향 LSTM을 적용하여, 전압 강하 곡선, 전류 스파이크, 온도 상승 패턴의 시간적 동역학을 학습한다. Hidden state 시퀀스 전체를 Cross-Attention의 입력으로 전달한다.

### 4.3 양방향 Cross-Attention 융합 🟢 `핵심 모듈`

Tsai et al. (ACL 2019)의 Multimodal Transformer에서 영감을 받아 양방향(bidirectional) Cross-Attention을 적용한다:

- **① Driving→Battery Attention:** 배터리 상태(Query)가 주행 패턴(Key, Value)에서 관련 정보를 검색
  → "이 배터리 전압 강하는 어떤 운전 조작 때문인가?"

- **② Battery→Driving Attention:** 주행 패턴(Query)이 배터리 상태(Key, Value)에서 관련 정보를 검색
  → "이 급가속은 배터리에 어떤 영향을 미쳤는가?"

Multi-Head Attention (8개 헤드)을 사용하여 다양한 유형의 교차모달 관계를 동시 포착한다. 융합된 표현은 Feed-Forward Network를 거쳐 최종 예측에 사용된다.

### 4.4 예측 태스크 및 타겟 변수 🔴 `변경`

단순 SOC 예측(포화 분야)을 피하고, 다음 두 가지 태스크를 동시 수행한다:

- **Primary:** 구간별 에너지 소모율(Wh/km) 예측 — 실질적 응용 가치가 높은 메트릭
- **Secondary:** 배터리 스트레스 지수(Battery Stress Index) 예측 — 전류 변화율, 온도 상승률, 전압 편차를 종합한 사용자 정의 복합 메트릭으로, 이 지수 자체가 방법론적 기여가 됨

### 4.5 해석 가능성 분석 (Interpretability) 🟢 `핵심 기여`

**본 연구의 핵심 기여.** Cross-Attention Weight 행렬을 시각화하여 다음을 정량적으로 분석한다:

**① 교차모달 인과 맵(Cross-Modal Causal Map):** 주행 변수별 → 배터리 변수별 영향 가중치 히트맵. 예: "급가속 구간에서 배터리 전압 강하에 대한 Attention Weight가 정속 구간 대비 2.3배 높다"

**② 시간적 인과 프로파일:** 특정 운전 이벤트(급가속, 급제동, 고속 순항) 발생 시점 전후의 Attention Weight 변화를 시계열로 추적

**③ Head별 역할 분석:** 8개 Attention Head가 각각 어떤 유형의 교차모달 관계를 전문화하여 학습하는지 분석 (예: Head 1은 가속-전류 관계, Head 3은 제동-회생 관계 등)

### 4.6 운전 습관 클러스터링 및 맞춤형 전략 도출 🟢 `NEW`

72개 트립의 주행 특성(평균 가속도, 정지 비율, 급가속 빈도, 최고 속도 등)을 추출한 뒤, 비지도 클러스터링(k-means + Silhouette 분석)으로 운전자 유형을 분류한다. 각 클러스터별로 Cross-Attention 인과 맵의 패턴 차이를 비교 분석하여, 운전 유형별 배터리 영향 메커니즘을 규명하고, 유형별 맞춤형 에코 드라이빙 가이드라인을 도출한다.

### 4.7 베이스라인 및 Ablation Study

| 모델                            | 설명                                   | 비교 목적                   |
| ------------------------------- | -------------------------------------- | --------------------------- |
| 단일 LSTM                       | BMS 데이터만 사용하는 기본 시계열 모델 | 멀티모달의 필요성 검증      |
| XGBoost                         | 수동 특성 기반 앙상블 기법             | 딥러닝의 우위 검증          |
| CNN-LSTM (Concat)               | 두 모달리티를 단순 연결 후 학습        | Cross-Attention의 우위 검증 |
| Self-Attention Only             | 단일 모달 내 Self-Attention 적용       | 교차모달 Attention의 필요성 |
| **제안 모델 (Cross-Attention)** | **양방향 Cross-Attention 융합 모델**   | **최종 성능**               |

평가 지표: RMSE, MAE, R² Score. Ablation Study에서 각 구성 요소(1D-CNN, LSTM, Cross-Attention, Bidirectional)의 개별 기여도를 체계적으로 검증한다.

---

## 5. 기대 성과 및 연구 일정

### 학술적 기여 (3중 노벨티)

- **방법론적 기여:** 주행-배터리 Cross-Attention 융합 아키텍처 최초 제안 (기존 연구는 모두 concatenation 또는 단일 모달)
- **분석적 기여:** Attention Weight 기반 교차모달 인과 해석 프레임워크 (운전 패턴 → 배터리 영향의 정량적 인과 맵)
- **응용적 기여:** 운전 습관 유형별 맞춤형 에코 드라이빙 가이드라인 (데이터 기반 정량적 권고안)

### 목표 학술지 / 학회

| 구분      | 대상                        | IF / 등급   | 투고 시기 |
| --------- | --------------------------- | ----------- | --------- |
| 1차 목표  | Applied Energy              | IF ~13 (Q1) | 6개월 후  |
| 2차 목표  | Batteries (MDPI)            | IF 4.8 (Q1) | 대안      |
| 학회 발표 | 대한산업공학회 추계학술대회 | 국내 학회   | 중간 성과 |

### 연구 일정 (6개월)

| 기간    | 내용                                                       |
| ------- | ---------------------------------------------------------- |
| 1개월차 | 데이터 다운로드, 탐색적 분석(EDA), 전처리 파이프라인 구축  |
| 2개월차 | 1D-CNN / LSTM 단일 모달 인코더 구현 및 베이스라인 학습     |
| 3개월차 | Cross-Attention 융합 모듈 구현, 하이퍼파라미터 튜닝        |
| 4개월차 | Ablation Study, Attention Weight 해석 분석, 인과 맵 시각화 |
| 5개월차 | 운전 습관 클러스터링, 유형별 분석, 맞춤형 전략 도출        |
| 6개월차 | 논문 작성 및 투고, 대한산업공학회 발표 준비                |

### 기대 효과

제안된 해석 가능한 멀티모달 AI 모델은 (1) 배터리 관리 시스템(BMS)의 지능화, (2) 소프트웨어 중심 자동차(SDV)의 운전자 맞춤형 에코 드라이빙 코칭, (3) 보험/플릿 관리에서의 배터리 건강 위험도 평가 등에 활용될 수 있다.

---

## 참고문헌

[1] Leech, D. & Yoon, S. (2025). Model-Based Deep Reinforcement Learning for Energy Efficient Routing. _Sustainability_, 17(13), 5727.

[2] Liu, Z. et al. (2025). Multi-modal framework for battery state of health evaluation using open-source EV data. _Nature Communications_, 16, 1137.

[3] Steinstraeter, M. et al. (2020). Battery and Heating Data in Real Driving Cycles. IEEE DataPort. DOI: 10.21227/6jr9-5235.

[4] Tsai, Y.-H. H. et al. (2019). Multimodal Transformer for Unaligned Multimodal Language Sequences. _ACL 2019_.

[5] Kollmeyer, P. et al. (2020). LG 18650HG2 Li-ion Battery Data. Mendeley Data. DOI: 10.17632/cp3473x7xv/3.

[6] Baltrusaitis, T. et al. (2019). Multimodal Machine Learning: A Survey and Taxonomy. _IEEE TPAMI_, 41(2), 423-443.

[7] Zhang, S. et al. (2024). Deep Multimodal Data Fusion. _ACM Computing Surveys_, 56(9).
