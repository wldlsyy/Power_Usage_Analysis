# 전력소비량 예측 프로젝트 - 데이터 전처리 및 이상치 탐지

이 프로젝트는 전력소비량 예측을 위한 데이터 전처리와 이상치 탐지 기능을 제공합니다.

## 📁 주요 파일 구조

```
Power_Usage_Analysis/
├── preprocessing.py              # 데이터 전처리 클래스
├── anomaly.py                   # 이상치 탐지 클래스
├── example_usage.py             # 전처리 사용 예시
├── example_anomaly_detection.py # 이상치 탐지 사용 예시
├── 01_EDA.ipynb                # 탐색적 데이터 분석 노트북
├── config/
│   ├── config.py               # 설정 파일
│   └── logger_config.py        # 로깅 설정
├── data/
│   ├── power_usage/raw/        # 원본 데이터
│   └── power_usage/processed/  # 전처리된 데이터
└── reports/                    # 분석 결과 보고서
```

## 🚀 빠른 시작

### 1. 기본 전처리

```python
from preprocessing import PowerUsagePreprocessor

# 전처리기 생성
preprocessor = PowerUsagePreprocessor()

# 전체 전처리 실행
train, test = preprocessor.run_full_preprocessing(
    save_data=True,
    remove_outliers=False
)
```

### 2. 이상치 탐지

```python
from anomaly import AnomalyDetector

# 이상치 탐지기 생성
detector = AnomalyDetector(train)

# 이상치 탐지
outliers = detector.detect_anomalies(
    columns=['power_usage'],
    method='rolling',  # 'iqr', 'zscore', 'seasonal', 'stl', 'isolation_forest'
    threshold=3.0
)

# 결과 시각화
detector.visualize_anomalies()

# 이상치 제거된 데이터 생성
cleaned_data = detector.remove_outliers()
```

### 3. 통합 워크플로우 (전처리 + 이상치 탐지)

```python
from preprocessing import PowerUsagePreprocessor

preprocessor = PowerUsagePreprocessor()

# 전처리와 이상치 탐지 한번에 실행
train, test, anomaly_results = preprocessor.run_with_anomaly_detection(
    anomaly_method='rolling',
    anomaly_threshold=3.0,
    remove_anomalies=True  # 이상치 제거까지 수행
)

# 결과 확인
print(f"탐지된 이상치: {anomaly_results['summary']['total_outliers']}개")
```

## 📊 PowerUsagePreprocessor 클래스

### 주요 기능

1. **데이터 로드**: 원본 CSV 파일들을 로드
2. **컬럼명 변환**: 한글 컬럼명을 영어로 변환
3. **시간 특성 추출**: 월, 일, 시간, 요일, 계절 등
4. **건물 정보 전처리**: 결측값 처리, 데이터 타입 변환
5. **데이터 병합**: 건물 정보와 사용량 데이터 결합
6. **결측값 처리**: 다양한 전략으로 결측값 처리
7. **특성 생성**: 파생 변수 자동 생성

### 사용 가능한 메소드

```python
# 단계별 실행
preprocessor.load_data()
preprocessor.rename_columns()
preprocessor.preprocess_datetime()
preprocessor.preprocess_building_info()
preprocessor.merge_data()
preprocessor.handle_missing_values()
preprocessor.create_features()

# 한번에 실행
train, test = preprocessor.run_full_preprocessing()

# 데이터 정보 확인
info = preprocessor.get_data_info()
```

### 생성되는 새로운 특성들

- `month`, `day`, `hour`: 시간 구성요소
- `day_of_week`: 요일 (0=월요일, 6=일요일)
- `is_weekend`: 주말 여부 (0/1)
- `day_type`: 'weekday' 또는 'weekend'
- `is_holiday`: 공휴일 여부
- `time_period`: 시간대 구분 ('morning', 'afternoon', etc.)
- `season`: 계절 구분 ('spring', 'summer', etc.)
- `temp_category`: 온도 구간 ('cold', 'mild', 'warm', 'hot')

## 🔍 AnomalyDetector 클래스

### 지원하는 이상치 탐지 방법

1. **IQR 방법** (`method='iqr'`): 사분위수 기반
2. **Z-Score 방법** (`method='zscore'`): 표준점수 기반
3. **Rolling 방법** (`method='rolling'`): 이동 창 기반 (추천)
4. **계절성 분해** (`method='seasonal'`): 시계열 분해 기반
5. **STL 분해** (`method='stl'`): 고급 시계열 분해
6. **Isolation Forest** (`method='isolation_forest'`): 머신러닝 기반

### 시각화 기능

```python
# 이상치 시각화 (시계열 플롯)
detector.visualize_anomalies(
    columns=['power_usage'],
    building_nums=[1, 2, 3],  # 특정 건물만
    figsize=(15, 10)
)

# 이상치 분포 분석
detector.plot_anomaly_distribution()
```

### 분석 및 내보내기

```python
# 결과 요약
summary = detector.get_anomaly_summary()

# 이상치 데이터 내보내기
detector.export_outliers("outliers.csv", format='csv')
detector.export_outliers("outliers.xlsx", format='excel')
```

## 🛠 고급 사용법

### 1. 커스텀 결측값 처리

```python
custom_strategy = {
    'solar_capacity': 'zero',    # 0으로 처리
    'ess_capacity': 'median',    # 중간값으로 처리
    'pcs_capacity': 'mean'       # 평균값으로 처리
}

train, test = preprocessor.run_full_preprocessing(
    missing_value_strategy=custom_strategy
)
```

### 2. 다양한 이상치 탐지 방법 비교

```python
from anomaly import compare_anomaly_methods

results = compare_anomaly_methods(
    data=train,
    columns=['power_usage'],
    methods=['rolling', 'iqr', 'seasonal']
)

for method, result in results.items():
    if result['success']:
        print(f"{method}: {result['summary']['total_outliers']}개 이상치")
```

### 3. 건물별 개별 분석

```python
# 특정 건물만 필터링
hospital_data = train[train['building_type'] == '병원']
detector = AnomalyDetector(hospital_data)

# 병원 건물들의 이상치 탐지
outliers = detector.detect_anomalies(method='rolling', threshold=2.5)
```

### 4. 시간대별 세부 분석

```python
# 피크 시간대 (14-17시) 데이터만
peak_hours_data = train[train['hour'].isin([14, 15, 16, 17])]
detector = AnomalyDetector(peak_hours_data)

# 피크 시간대 이상치 탐지
outliers = detector.detect_anomalies(method='seasonal')
```

## 📈 성능 최적화 팁

### 대용량 데이터 처리

```python
# 샘플링을 통한 빠른 분석
sample_data = train.sample(n=10000)
detector = AnomalyDetector(sample_data)

# 메모리 효율적인 방법 사용
outliers = detector.detect_anomalies(method='iqr')  # 빠른 방법
```

### 병렬 처리 (건물별)

```python
from concurrent.futures import ProcessPoolExecutor

def analyze_building(building_num):
    building_data = train[train['building_num'] == building_num]
    detector = AnomalyDetector(building_data)
    return detector.detect_anomalies()

# 여러 건물 동시 분석
with ProcessPoolExecutor() as executor:
    results = list(executor.map(analyze_building, train['building_num'].unique()[:5]))
```

## 📊 보고서 생성

### HTML 보고서 자동 생성

```python
from anomaly import create_anomaly_report

# 종합 분석 보고서 생성
create_anomaly_report(train, "reports/anomaly_analysis.html")
```

## ⚠️ 주의사항

1. **메모리 사용량**: 대용량 데이터의 경우 샘플링 권장
2. **계절성 방법**: 최소 72시간(3일) 이상의 연속 데이터 필요
3. **STL 분해**: statsmodels 라이브러리 필요
4. **Isolation Forest**: scikit-learn 라이브러리 필요

## 🔧 문제 해결

### 자주 발생하는 오류

1. **모듈을 찾을 수 없음**: 
   ```bash
   pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels
   ```

2. **메모리 부족**: 데이터 샘플링 또는 청크 단위 처리
   ```python
   sample_data = train.sample(frac=0.1)  # 10% 샘플링
   ```

3. **시각화 오류**: 한글 폰트 설정
   ```python
   import matplotlib.pyplot as plt
   plt.rc("font", family="Malgun Gothic")
   ```

## 📞 지원

- 이슈나 문제가 있으면 GitHub Issues에 등록해 주세요.
- 추가 기능 요청이나 개선사항도 환영합니다.

## 📝 업데이트 로그

### v1.0.0
- PowerUsagePreprocessor 클래스 구현
- AnomalyDetector 클래스 구현  
- 통합 워크플로우 지원
- 다양한 이상치 탐지 방법 제공
- 시각화 및 보고서 기능 추가
