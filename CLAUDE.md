# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 개발 환경

패키지 관리는 **uv**를 사용한다.

```bash
# 가상환경 생성 및 패키지 설치
uv sync

# 패키지 추가
uv add <package>

# Jupyter 실행
uv run jupyter lab

# 스크립트 실행
uv run python src/train.py
```

Python 버전: 3.11 / 가상환경: `.venv/`

---

## 프로젝트 개요

ITRC(정보통신기술연구센터) 연구 프로젝트로, **SDV(소프트웨어 중심 자동차) 온디바이스 환경을 위한 멀티모달 배터리 에너지 최적화 AI 모델** 개발을 목표로 한다.

- 연구 핵심: 차량 제어 데이터(CAN)와 배터리 상태 데이터(BMS)를 융합하여 배터리 소모를 예측하고, 지식 증류(Knowledge Distillation)를 통해 엣지 디바이스에 탑재 가능한 경량 모델로 압축
- 목표 저널/학회: *Batteries* 저널, 대한산업공학회 추계학술대회

## 데이터셋

| 역할 | 데이터셋 | 내용 |
|---|---|---|
| 주 학습 데이터 | BMW i3 Real Driving Cycles | 72개 실주행 트립, 24개 변수, 타임스탬프 완전 동기화. `data/bmw_i3_driving_cycles/` |
| 일반화 검증 | McMaster LG 18650HG2 | 12개 표준 주행 사이클 × 6개 온도 조건. `data/mcmaster_lg18650hg2/` |

다운로드는 각 폴더의 `README.md` 참고. BMW i3 데이터는 타임스탬프가 이미 동기화되어 DTW 불필요.

## 모델 아키텍처: Dual-Encoder Cross-Attention Network

- **주행 인코더 (Multi-scale 1D-CNN)**: 커널 크기 3·5·7 병렬 적용 → 급가속/급제동 패턴 추출
- **배터리 인코더 (Bi-LSTM)**: 양방향 LSTM → 전압·전류·온도 동역학 학습
- **양방향 Cross-Attention 융합 (8 heads)**
  - Driving→Battery: Query=BMS, Key/Value=CAN ("이 전압 강하는 어떤 조작 때문인가?")
  - Battery→Driving: Query=CAN, Key/Value=BMS ("이 급가속은 배터리에 어떤 영향인가?")
- **예측 헤드 (Multi-task)**: 에너지 소모율(Wh/km) + 배터리 스트레스 지수(BSI)

## 예측 타겟

- **Primary**: 에너지 소모율 (Wh/km)
- **Secondary**: 배터리 스트레스 지수 (BSI) = `α·|dI/dt| + β·|dT/dt| + γ·|ΔV|` (사용자 정의 복합 메트릭)

## 해석 가능성 분석 (핵심 기여)

Attention Weight 행렬을 시각화하여:
1. **교차모달 인과 맵**: 주행 변수 × 배터리 변수 영향 히트맵
2. **시간적 인과 프로파일**: 운전 이벤트 전후 Attention Weight 추적
3. **Head별 역할 분석**: 8개 Head의 전문화된 교차모달 관계 규명

## 기술 스택

- **프레임워크**: PyTorch
- **전처리**: Min-Max 정규화, 슬라이딩 윈도우 (60초 window, 10초 stride)
- **클러스터링**: k-means + Silhouette (운전 습관 유형 분류)
- **베이스라인 비교**: 단일 LSTM, XGBoost, CNN-LSTM Concat, Self-Attention Only

## 프로젝트 구조

```
Mobility-AI-Project/
├── docs/               # 연구계획서
├── data/
│   ├── bmw_i3_driving_cycles/   # 주 학습 데이터 (다운로드 필요)
│   └── mcmaster_lg18650hg2/     # 일반화 검증 데이터 (다운로드 필요)
├── src/                # 모듈 코드
├── notebooks/          # EDA, 실험, 분석
├── results/            # 학습 로그, 평가 결과
└── todo.md             # 단계별 연구 TODO
```

## 문서

- `docs/A_research_plan.md` – 최신 연구계획서 (수정본, Cross-Attention 중심)
- `docs/A_research_plan_old.md` – 초기 연구계획서 (Teacher-Student KD 방식)
- `docs/A_research_plan_2.md` – FDKD 상세 실험 계획서 (수도코드 포함, 구버전 참고용)
- `todo.md` – 6개월 연구 단계별 체크리스트
