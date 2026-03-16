# McMaster University LG 18650HG2 Li-ion Battery Dataset

**출처:** Kollmeyer, Vidal, Naguib & Skells (2020), McMaster University
**DOI:** 10.17632/cp3473x7xv/3
**라이선스:** CC BY 4.0

---

## 데이터셋 개요

본 프로젝트에서 **일반화 검증(Cross-domain Validation)** 용도로 사용.
BMW i3에서 학습한 제안 모델이 다른 배터리/주행 조건에서도 성능을 유지하는지 검증.

| 항목 | 내용 |
|---|---|
| 배터리 | LG 18650HG2 (3Ah Li-ion) |
| 테스트 환경 | 8 cu.ft. 열챔버, Digatron 75A/5V 테스터 |
| 주행 사이클 | 12개 표준 사이클 (UDDS, HWFET, US06, LA92, Neural 등) |
| 온도 조건 | 6개 온도 (-20°C ~ +40°C) |
| 측정 변수 | 배터리 전압(V), 전류(A), 온도(°C), SoC(%) |
| 샘플링 | 1초 단위 |

---

## 다운로드 방법

### 방법 1: Mendeley Data (권장, 무료 직접 다운로드)

1. 아래 링크 접속:
   ```
   https://data.mendeley.com/datasets/cp3473x7xv/3
   ```
2. **"Download All"** 버튼 클릭 (계정 불필요 또는 무료 가입)
3. 다운로드한 ZIP을 이 폴더에 압축 해제

### 방법 2: Kaggle

1. [Kaggle 페이지](https://www.kaggle.com/datasets/aditya9790/lg-18650hg2-liion-battery-data) 접속
2. Download 버튼 클릭

**또는 Kaggle CLI:**
```bash
kaggle datasets download -d aditya9790/lg-18650hg2-liion-battery-data -p data/mcmaster_lg18650hg2 --unzip
```

---

## 예상 폴더 구조 (압축 해제 후)

```
mcmaster_lg18650hg2/
├── README.md                          ← 이 파일
├── FNN_xEV_Li_ion_SOC_EstimatorScript_March_2020.mlx   ← MATLAB 예제
├── 25degC/
│   ├── UDDS_25degC.mat
│   ├── HWFET_25degC.mat
│   └── ...
├── 0degC/
│   └── ...
└── ...
```

---

## 사용 방법 (Python)

```python
import scipy.io
import pandas as pd

# .mat 파일 로드 예시
mat = scipy.io.loadmat('data/mcmaster_lg18650hg2/25degC/UDDS_25degC.mat')
# 주요 키: 'Voltage', 'Current', 'Temperature', 'SOC', 'Time'
time = mat['Time'].flatten()
voltage = mat['Voltage'].flatten()
current = mat['Current'].flatten()
temp = mat['Temperature'].flatten()
soc = mat['SOC'].flatten()

df = pd.DataFrame({'time': time, 'voltage': voltage, 'current': current,
                   'temperature': temp, 'soc': soc})
```

---

## 인용

```bibtex
@data{cp3473x7xv/3,
  author    = {Kollmeyer, Philip and Vidal, Carlos and Naguib, Mina and Skells, Michael},
  title     = {LG 18650HG2 Li-ion Battery Data and Example Deep Neural Network xEV SOC Estimator Script},
  year      = {2020},
  publisher = {Mendeley Data},
  doi       = {10.17632/cp3473x7xv/3}
}
```
