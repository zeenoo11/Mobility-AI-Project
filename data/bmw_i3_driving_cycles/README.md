# BMW i3 Real Driving Cycles Dataset

**출처:** Steinstraeter, Buberger & Trifonov (2020), Technical University of Munich
**DOI:** 10.21227/6jr9-5235
**라이선스:** Open Access (IEEE DataPort)

---

## 데이터셋 개요

BMW i3 (60 Ah) 차량으로 독일 실도로에서 수집한 72개 주행 트립 데이터.
주행 신호와 배터리 신호가 동일 차량에서 동시 수집되어 타임스탬프 완전 동기화.

| 변수 그룹 | 변수 수 | 주요 변수 |
|---|---|---|
| 주행 모달리티 | 12개 | 차량 속도, 스로틀, 브레이크, 종방향 가속도, 모터 토크, 고도 |
| 배터리 모달리티 | 6개 | 전압(V), 전류(A), 온도(°C), SoC(%), 배터리 파워(kW) |
| 환경 변수 | 6개 | 외기 온도, 에어컨 파워, 히터 파워, 실내 온도 |

- **Category A:** 여름 트립 (일부 측정 이슈 있음)
- **Category B:** 겨울 트립 (완전하고 일관된 데이터)

---

## 다운로드 방법

### 방법 1: Kaggle (권장, 36.7 MB ZIP)

1. [Kaggle 계정](https://www.kaggle.com) 로그인
2. 아래 페이지에서 Download 버튼 클릭:
   ```
   https://www.kaggle.com/datasets/atechnohazard/battery-and-heating-data-in-real-driving-cycles
   ```
3. 다운로드한 ZIP을 이 폴더에 압축 해제

**또는 Kaggle CLI 사용:**
```bash
pip install kaggle
# ~/.kaggle/kaggle.json 에 API 토큰 설정 후:
kaggle datasets download -d atechnohazard/battery-and-heating-data-in-real-driving-cycles -p data/bmw_i3_driving_cycles --unzip
```

### 방법 2: IEEE DataPort (원본)

1. [IEEE DataPort](https://ieee-dataport.org/open-access/battery-and-heating-data-real-driving-cycles) 접속
2. IEEE 계정으로 로그인 (무료 가입)
3. "Download" 버튼 클릭 후 이 폴더에 저장

---

## 예상 폴더 구조 (압축 해제 후)

```
bmw_i3_driving_cycles/
├── README.md               ← 이 파일
├── readin.m                ← MATLAB 읽기 스크립트
├── A/                      ← 여름 트립
│   ├── Trip_1.csv
│   └── ...
└── B/                      ← 겨울 트립
    ├── Trip_1.csv
    └── ...
```

---

## 인용

```bibtex
@data{6jr9-5235-20,
  author    = {Steinstraeter, Matthias and Buberger, Johannes and Trifonov, Dimitar},
  title     = {Battery and Heating Data in Real Driving Cycles},
  year      = {2020},
  publisher = {IEEE DataPort},
  doi       = {10.21227/6jr9-5235}
}
```
