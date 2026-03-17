# Mobility-AI-Project

ITRC(정보통신기술연구센터) 연구 프로젝트로, SDV(소프트웨어 중심 자동차) 온디바이스 환경에서 동작 가능한
멀티모달 배터리 에너지 최적화 AI 모델을 개발하는 저장소입니다.

## 프로젝트 목표

- 차량 제어 데이터(CAN)와 배터리 상태 데이터(BMS)를 융합해 에너지 소모를 예측
- Cross-Attention 기반 모델로 주행 이벤트와 배터리 반응 간 관계를 해석
- 경량화/지식 증류를 통해 엣지 디바이스 탑재 가능성 검증

## 데이터셋

- **BMW i3 Real Driving Cycles**: `data/bmw_i3_driving_cycles/`
- **McMaster LG 18650HG2**: `data/mcmaster_lg18650hg2/`

각 데이터셋 다운로드 방법은 해당 폴더의 `README.md`를 참고하세요.

## 개발 환경

- Python 3.11
- 패키지 매니저: `uv`

```bash
# 가상환경 생성 및 의존성 설치
uv sync

# 스크립트 실행 예시
uv run python src/train.py
```

## 저장소 구조

```text
Mobility-AI-Project/
├── docs/      # 연구계획서 및 문서
├── data/      # 데이터셋 관련 파일
├── src/       # 모델/학습/평가 코드
├── notebooks/ # 실험 및 분석 노트북
├── results/   # 학습 로그 및 결과
└── todo.md    # 단계별 TODO
```

## 주요 문서

- `docs/A_research_plan.md`: 최신 연구계획서
- `docs/A_research_plan_old.md`: 초기 연구계획서
- `docs/A_research_plan_2.md`: FDKD 상세 실험 계획서 (참고용)
