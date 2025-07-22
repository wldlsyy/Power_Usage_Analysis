import pandas as pd
import numpy as np
import config.config as cf
from typing import Tuple, Dict, List, Optional
import logging

class PowerUsagePreprocessor:
    def __init__(self, data_dir: str = None):
        self.data_dir = cf.RAWDATA_DIR
        self.train = None
        self.test = None
        self.building = None
        self.sample_submission = None
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 컬럼명 매핑 딕셔너리
        self.column_mappings = {
            'train': ['num_date_time', 'building_num', 'date_time', 'temperature', 
                     'rain', 'wind', 'humidity', 'sun', 'solar', 'power_usage'],
            'test': ['num_date_time', 'building_num', 'date_time', 'temperature', 
                    'rain', 'wind', 'humidity'],
            'building': ['building_num', 'building_type', 'floor_area', 'cool_area', 
                        'solar_capacity', 'ess_capacity', 'pcs_capacity']
        }
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logging.info("데이터 로딩 시작...")
        
        try:
            self.train = pd.read_csv(f'{self.data_dir}/train.csv')
            self.test = pd.read_csv(f'{self.data_dir}/test.csv')
            self.sample_submission = pd.read_csv(f'{self.data_dir}/sample_submission.csv')
            self.building = pd.read_csv(f'{self.data_dir}/building_info.csv')
            
            logging.info(f"데이터 로딩 완료 - Train: {self.train.shape}, Test: {self.test.shape}, "
                           f"Sample: {self.sample_submission.shape}, Building: {self.building.shape}")
            
            return self.train, self.test, self.sample_submission, self.building
            
        except Exception as e:
            logging.error(f"데이터 로딩 중 오류 발생: {e}")
            raise
    
    def _rename_columns(self) -> None:
        """
        데이터프레임 컬럼명을 영어로 변환
        """
        logging.info("컬럼명 영어 변환 중...")
        
        if self.train is not None:
            self.train.columns = self.column_mappings['train']
        if self.test is not None:
            self.test.columns = self.column_mappings['test']
        if self.building is not None:
            self.building.columns = self.column_mappings['building']
            
        logging.info("컬럼명 변환 완료")
    
    def preprocess_datetime(self) -> None:
        """
        날짜/시간 특성 추출 및 전처리
        - datetime 변환
        - 월, 일, 시간 추출
        - 주말/평일 구분
        """
        logging.info("날짜/시간 특성 추출 중...")
        
        for df in [self.train, self.test]:
            if df is not None:
                # datetime 변환 (YYYYMMDD HH 형태 -> datetime)
                df['date_time'] = pd.to_datetime(df['date_time'], format='%Y%m%d %H')
                
                # 시간 특성 추출
                df['month'] = df['date_time'].dt.month
                df['day'] = df['date_time'].dt.day
                df['hour'] = df['date_time'].dt.hour
                df['day_of_week'] = df['date_time'].dt.dayofweek
                
                # 주말/평일 구분
                df['day_type'] = df['day_of_week'].apply(lambda x: 'weekend' if x >= 5 else 'weekday')
                
                # 공휴일 표시 (2024-06-06, 2024-08-15)
                holiday_dates = ['2024-06-06', '2024-08-15']
                df['is_holiday'] = df['date_time'].dt.date.isin([
                    pd.to_datetime(date).date() for date in holiday_dates
                ]).astype(int)
        
        logging.info("날짜/시간 특성 추출 완료")

    def drop_columns(self, columns: List[str]) -> None:
        logging.info(f"제거할 컬럼: {columns}")
        
        for df in [self.train, self.test]:
            if df is not None:
                df.drop(columns=columns, inplace=True, errors='ignore')
        
        logging.info("지정된 컬럼 제거 완료")

    def add_columns(self, target_attr: str, columns: Dict[str, List[str]]) -> None:
        """
        특정 데이터프레임 속성(self.train, self.test 등)에 새로운 컬럼 추가
        Args:
            target_attr: 대상 데이터프레임 속성명 ('train', 'test', 'building' 등)
            columns: {column_name: [value for each row], ...}
        """
        logging.info(f"{target_attr}에 추가할 컬럼: {list(columns.keys())}") 
        
        # 대상 데이터프레임 속성 가져오기
        if not hasattr(self, target_attr):
            logging.error(f"'{target_attr}' 속성이 존재하지 않습니다.")
            return
            
        df = getattr(self, target_attr)
        if df is None:
            logging.error(f"'{target_attr}' 데이터프레임이 None입니다.")
            return
            
        # 컬럼 추가
        for col_name, values in columns.items():
            if col_name not in df.columns:
                df[col_name] = values
                logging.info(f"'{col_name}' 칼럼이 추가되었습니다.")
                print(df.head())  # 추가된 칼럼의 일부 데이터 출력
            else:
                logging.warning(f"'{col_name}' 칼럼이 이미 존재합니다. 덮어쓰지 않습니다.")
        
        # 업데이트된 데이터프레임을 다시 클래스 속성에 할당
        setattr(self, target_attr, df)

    def preprocess_building_info(self) -> None:
        """
        건물 정보 전처리
        - '-' 값을 NaN으로 변환
        - 수치형 컬럼 변환
        """
        logging.info("건물 정보 전처리 중...")
        
        if self.building is not None:
            # '-' 값을 NaN으로 변환
            self.building = self.building.replace('-', np.nan)
            
            # 수치형 컬럼 변환
            numeric_cols = ['solar_capacity', 'ess_capacity', 'pcs_capacity']
            for col in numeric_cols:
                self.building[col] = pd.to_numeric(self.building[col], errors='coerce')
            
            logging.info(f"건물 정보 - 태양광, ESS, PCS 용량 데이터 수치 데이터로 변경 완료: ")

    def merge_data(self) -> None:
        """
        train/test 데이터에 건물 정보 병합
        """
        logging.info("데이터 병합 중...")
        
        if self.building is not None:
            if self.train is not None:
                self.train = self.train.merge(self.building, on='building_num', how='left')
            if self.test is not None:
                self.test = self.test.merge(self.building, on='building_num', how='left')
        
        logging.info("데이터 병합 완료")
    
    def handle_missing_values(self, strategy: Dict[str, str] = None) -> None:
        """
        결측값 처리
        
        Args:
            strategy: 컬럼별 결측값 처리 전략 딕셔너리
                     예: {'solar_capacity': 'median', 'ess_capacity': 'mean'}
        """
        logging.info("결측값 처리 중...")
        
        default_strategy = {
            'solar_capacity': 'zero',  # 태양광이 없는 건물은 0으로 처리
            'ess_capacity': 'zero',    # ESS가 없는 건물은 0으로 처리
            'pcs_capacity': 'zero'     # PCS가 없는 건물은 0으로 처리
        }
        
        strategy = strategy or default_strategy
        
        for df in [self.train, self.test]:
            if df is not None:
                for col, method in strategy.items():
                    if col in df.columns and df[col].isnull().sum() > 0:
                        if method == 'zero':
                            df[col].fillna(0, inplace=True)
                        elif method == 'mean':
                            df[col].fillna(df[col].mean(), inplace=True)
                        elif method == 'median':
                            df[col].fillna(df[col].median(), inplace=True)
                        elif method == 'mode':
                            df[col].fillna(df[col].mode()[0], inplace=True)
        
        logging.info("결측값 처리 완료")
    
    def detect_outliers(self, columns: List[str] = None, method: str = 'rolling', 
                       threshold: float = 3.0, window: int = 24) -> Dict[str, pd.Series]:
        """
        시계열 데이터에 적합한 이상치 탐지, 특히 power_usage를 위한 방법
        
        Args:
            columns: 이상치를 탐지할 컬럼 리스트
            method: 이상치 탐지 방법 ('iqr', 'zscore', 'rolling', 'seasonal', 'stl', 'isolation_forest')
            threshold: 이상치 임계값
            window: rolling 방법 사용 시 윈도우 크기
        Returns:
            Dict containing outlier indices for each column
        """
        logging.info(f"시계열 데이터의 이상치 탐지 중 (방법: {method})...")
        
        if self.train is None:
            return {}
        
        if columns is None:
            # power_usage가 주요 타겟
            columns = ['power_usage']
            if 'temperature' in self.train.columns:  # 전력 사용량과 관련된 주요 변수들
                columns.extend(['temperature'])
        
        outliers = {}
        
        for col in columns:
            if col not in self.train.columns:
                continue
                
            if method == 'iqr':
                # 각 건물별, 시간대별로 IQR 기반 이상치 탐지 (시계열 특성 고려)
                outliers[col] = pd.Series(dtype=int)
                for bldg in self.train['building_num'].unique():
                    for hour in range(24):
                        subset = self.train[(self.train['building_num'] == bldg) & 
                                          (self.train['hour'] == hour)][col]
                        if len(subset) < 10:  # 충분한 데이터가 없으면 건너뜀
                            continue
                        Q1 = subset.quantile(0.25)
                        Q3 = subset.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        outlier_idx = subset[(subset < lower_bound) | (subset > upper_bound)].index
                        outliers[col] = pd.concat([outliers[col], pd.Series(outlier_idx)])

            elif method == 'zscore':
                # 건물별, 요일/시간대별 Z-score 계산
                outliers[col] = pd.Series(dtype=int)
                for bldg in self.train['building_num'].unique():
                    for day_type in ['weekday', 'weekend']:
                        for hour in range(24):
                            subset = self.train[(self.train['building_num'] == bldg) &
                                              (self.train['day_type'] == day_type) &
                                              (self.train['hour'] == hour)][col]
                            if len(subset) < 10:
                                continue
                            from scipy import stats
                            z_scores = np.abs(stats.zscore(subset.dropna()))
                            outlier_idx = subset.dropna().index[z_scores > threshold]
                            outliers[col] = pd.concat([outliers[col], pd.Series(outlier_idx)])

            elif method == 'rolling':
                # 건물별 시계열 롤링 윈도우 기반 이상치 탐지
                outliers[col] = pd.Series(dtype=int)
                for bldg in self.train['building_num'].unique():
                    bldg_data = self.train[self.train['building_num'] == bldg].sort_values('date_time')
                    rolling_mean = bldg_data[col].rolling(window, min_periods=5, center=True).mean()
                    rolling_std = bldg_data[col].rolling(window, min_periods=5, center=True).std()
                    diff = np.abs(bldg_data[col] - rolling_mean)
                    outlier_mask = diff > (threshold * rolling_std)
                    # NaN 처리
                    outlier_mask = outlier_mask.fillna(False)
                    outlier_idx = bldg_data.loc[outlier_mask].index
                    outliers[col] = pd.concat([outliers[col], pd.Series(outlier_idx)])

            elif method == 'seasonal':
                # 시계열 데이터의 계절성 고려 (시간대별, 요일별 패턴)
                from statsmodels.tsa.seasonal import seasonal_decompose
                
                outliers[col] = pd.Series(dtype=int)
                for bldg in self.train['building_num'].unique():
                    bldg_data = self.train[self.train['building_num'] == bldg].sort_values('date_time')
                    
                    # 충분한 데이터가 있는지 확인
                    if len(bldg_data) < 48:  # 최소 2일치 데이터
                        continue
                    
                    try:
                        # 하루 단위(24시간) 주기로 분해
                        decomposition = seasonal_decompose(bldg_data[col], period=24, model='additive')
                        residual = decomposition.resid
                        
                        # 잔차의 표준편차를 기준으로 이상치 탐지
                        std_resid = np.std(residual.dropna())
                        outlier_idx = bldg_data[np.abs(residual) > threshold * std_resid].index
                        outliers[col] = pd.concat([outliers[col], pd.Series(outlier_idx)])
                    except:
                        logging.warning(f"건물 {bldg}의 계절성 분해 실패, 건너뜀")
                        continue
                        
            elif method == 'stl':
                # STL 분해 방법 (계절성-트렌드-잔차)
                try:
                    from statsmodels.tsa.seasonal import STL
                    
                    outliers[col] = pd.Series(dtype=int)
                    for bldg in self.train['building_num'].unique():
                        bldg_data = self.train[self.train['building_num'] == bldg].sort_values('date_time')
                        
                        # 충분한 데이터가 있는지 확인 (STL은 더 많은 데이터가 필요)
                        if len(bldg_data) < 72:  # 최소 3일치 데이터
                            continue
                            
                        # 연속적인 시계열 데이터인지 확인
                        if bldg_data['date_time'].diff().dt.total_seconds().max() > 3600:
                            # 시간 단위 데이터가 연속적이지 않으면 재샘플링
                            bldg_data = bldg_data.set_index('date_time').resample('H').mean().reset_index()
                            
                        try:
                            # STL 분해 수행 (일간, 주간 계절성 모두 고려)
                            # period=24: 일간 주기, period=168: 주간 주기 (24*7)
                            stl = STL(bldg_data[col], 
                                     period=24,
                                     seasonal=13,  # 계절성 윈도우 (2*period+1 권장)
                                     trend=25,     # 트렌드 윈도우
                                     robust=True)  # 이상치에 강건한 분해
                            result = stl.fit()
                            
                            # 잔차 기반 이상치 탐지
                            residual = result.resid
                            resid_std = np.std(residual.dropna())
                            
                            # 임계값 적용
                            outlier_mask = np.abs(residual) > threshold * resid_std
                            outlier_idx = bldg_data.index[outlier_mask]
                            outliers[col] = pd.concat([outliers[col], pd.Series(outlier_idx)])
                            
                            logging.info(f"건물 {bldg} STL 분해 완료: {sum(outlier_mask)} 이상치 발견")
                        except Exception as e:
                            logging.warning(f"건물 {bldg}의 STL 분해 실패: {e}")
                            continue
                except ImportError:
                    logging.warning("statsmodels STL 모듈을 불러올 수 없습니다")
                    continue

            elif method == 'isolation_forest':
                # 고급 머신러닝 기반 이상치 탐지
                try:
                    from sklearn.ensemble import IsolationForest
                    
                    outliers[col] = pd.Series(dtype=int)
                    for bldg in self.train['building_num'].unique():
                        bldg_data = self.train[self.train['building_num'] == bldg]
                        
                        # 특성 선택 (시간 특성 + 관련 수치 특성)
                        features = ['hour', 'day_of_week']
                        for feat in ['temperature', 'humidity', col]:
                            if feat in bldg_data.columns:
                                features.append(feat)
                        
                        if len(bldg_data) < 50:  # 데이터가 충분하지 않으면 건너뜀
                            continue
                        
                        X = bldg_data[features].fillna(method='ffill')
                        model = IsolationForest(contamination=0.05, random_state=42)
                        preds = model.fit_predict(X)
                        outlier_idx = bldg_data.index[preds == -1]
                        outliers[col] = pd.concat([outliers[col], pd.Series(outlier_idx)])
                except ImportError:
                    logging.warning("sklearn 모듈이 없어 isolation_forest 방법을 사용할 수 없습니다")
                    continue

        # 중복 제거
        for col in outliers:
            outliers[col] = pd.Series(outliers[col].unique())
            
        total_outliers = sum(len(indices) for indices in outliers.values())
        logging.info(f"시계열 이상치 탐지 완료 - 총 {total_outliers}개 이상치 발견")
        
        return outliers
    
    def remove_outliers(self, outlier_indices: Dict[str, pd.Series] = None, 
                       columns: List[str] = None, method: str = 'iqr',
                       threshold: float = 1.5) -> None:
        """
        이상치 제거
        
        Args:
            outlier_indices: 미리 탐지된 이상치 인덱스
            columns: 이상치를 제거할 컬럼 리스트
            method: 이상치 탐지 방법
            threshold: 이상치 임계값
        """
        if outlier_indices is None:
            outlier_indices = self.detect_outliers(columns, method, threshold)
        
        if self.train is not None and outlier_indices:
            # 모든 이상치 인덱스 합집합
            all_outlier_indices = set()
            for indices in outlier_indices.values():
                all_outlier_indices.update(indices)
            
            original_shape = self.train.shape[0]
            self.train = self.train.drop(list(all_outlier_indices))
            
            logging.info(f"이상치 제거 완료 - {original_shape}개 → {self.train.shape[0]}개 "
                           f"({len(all_outlier_indices)}개 제거)")
    
    def create_features(self) -> None:
        """
        추가 특성 생성
        """
        logging.info("추가 특성 생성 중...")
        
        for df in [self.train, self.test]:
            if df is not None:
                # 시간대 구분
                df['time_period'] = df['hour'].apply(self._categorize_time_period)
                
                # 계절 구분
                # df['season'] = df['month'].apply(self._categorize_season) # 6~8월 데이터라 필요 X
                
                # 기온 구간 나누기
                if 'temperature' in df.columns:
                    df['temp_category'] = pd.cut(df['temperature'], 
                                               bins=[-np.inf, 20, 25, 30, np.inf],
                                               labels=['cold', 'mild', 'warm', 'hot'])
                
                # 건물 용량 대비 비율 (있는 경우에만)
                if all(col in df.columns for col in ['solar_capacity', 'floor_area']):
                    df['solar_per_area'] = df['solar_capacity'] / (df['floor_area'] + 1e-6)
                
                if all(col in df.columns for col in ['cool_area', 'floor_area']):
                    df['cool_ratio'] = df['cool_area'] / (df['floor_area'] + 1e-6)
        
        logging.info("추가 특성 생성 완료")
    
    def _categorize_time_period(self, hour: int) -> str:
        """시간대 구분"""
        if 6 <= hour < 9:
            return 'morning'
        elif 9 <= hour < 12:
            return 'forenoon'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    
    def get_data_info(self) -> Dict[str, Dict]:
        """
        전처리된 데이터 정보 반환
        
        Returns:
            Dictionary containing information about each dataset
        """
        info = {}
        
        for name, df in [('train', self.train), ('test', self.test), ('building', self.building)]:
            if df is not None:
                info[name] = {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'dtypes': df.dtypes.to_dict(),
                    'missing_values': df.isnull().sum().to_dict(),
                    'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
                }
        
        return info
    
    def save_processed_data(self, output_dir: str = None) -> None:
        """
        전처리된 데이터 저장
        
        Args:
            output_dir: 저장할 디렉토리 경로 (기본값: data/power_usage/processed)
        """
        if output_dir is None:
            output_dir = 'data/power_usage/processed'
        
        logging.info(f"전처리된 데이터 저장 중... (경로: {output_dir})")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if self.train is not None:
            self.train.to_csv(f'{output_dir}/train_processed.csv', index=False)
        if self.test is not None:
            self.test.to_csv(f'{output_dir}/test_processed.csv', index=False)
        if self.building is not None:
            self.building.to_csv(f'{output_dir}/building_processed.csv', index=False)
        
        logging.info("데이터 저장 완료")
    
    def run_full_preprocessing(self, save_data: bool = True, 
                             remove_outliers: bool = False,
                             missing_value_strategy: Dict[str, str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        전체 전처리 파이프라인 실행
        
        Args:
            save_data: 전처리된 데이터 저장 여부
            remove_outliers: 이상치 제거 여부
            missing_value_strategy: 결측값 처리 전략
            
        Returns:
            Tuple containing processed train and test DataFrames
        """
        logging.info("전체 전처리 파이프라인 시작...")
        
        # 1. 데이터 로드
        self.load_data()
        
        # 2. 컬럼명 변환
        self._rename_columns()
        
        # 3. 날짜/시간 전처리
        self.preprocess_datetime()
        
        # 4. 건물 정보 전처리
        self.preprocess_building_info()
        
        # 5. 데이터 병합
        self.merge_data()
        
        # 6. 결측값 처리
        self.handle_missing_values(missing_value_strategy)
        
        # 7. 이상치 처리 (선택사항)
        if remove_outliers:
            self.remove_outliers()
        
        # 8. 추가 특성 생성
        # self.create_features()
        
        # 9. 데이터 저장 (선택사항)
        if save_data:
            self.save_processed_data()
        
        self.logger.info("전체 전처리 파이프라인 완료!")
        
        return self.train, self.test

# 사용 예시
if __name__ == "__main__":
    # 전처리기 인스턴스 생성
    preprocessor = PowerUsagePreprocessor()
    
    # 전체 전처리 실행
    train_processed, test_processed = preprocessor.run_full_preprocessing(
        save_data=True,
        remove_outliers=False  # 이상치 제거는 선택사항
    )
    
    # 데이터 정보 출력
    data_info = preprocessor.get_data_info()
    print("전처리 완료된 데이터 정보:")
    for dataset_name, info in data_info.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"  Shape: {info['shape']}")
        print(f"  Memory Usage: {info['memory_usage']}")
        print(f"  Missing Values: {sum(info['missing_values'].values())} total")
