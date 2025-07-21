import pandas as pd
import numpy as np
import config.config as cf
from typing import Tuple, Dict, List, Optional
from config.logger_config import setup_logger
import logging

class PowerUsagePreprocessor:
    def __init__(self, data_dir: str = None):
        self.data_dir = cf.RAWDATA_DIR
        self.train = None
        self.test = None
        self.building = None
        self.sample = None
        
        # 로깅 설정
        setup_logger()
        
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
            self.sample = pd.read_csv(f'{self.data_dir}/sample_submission.csv')
            self.building = pd.read_csv(f'{self.data_dir}/building_info.csv')
            
            logging.info(f"데이터 로딩 완료 - Train: {self.train.shape}, Test: {self.test.shape}, "
                           f"Sample: {self.sample.shape}, Building: {self.building.shape}")
            
            return self.train, self.test, self.sample, self.building
            
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
    
    def detect_outliers(self, columns: List[str] = None, method: str = 'iqr', 
                       threshold: float = 1.5) -> Dict[str, pd.Series]:
        """
        이상치 탐지
        
        Args:
            columns: 이상치를 탐지할 컬럼 리스트
            method: 이상치 탐지 방법 ('iqr', 'zscore')
            threshold: 이상치 임계값
            
        Returns:
            Dict containing outlier indices for each column
        """
        logging.info(f"이상치 탐지 중 (방법: {method})...")
        
        if self.train is None:
            return {}
        
        if columns is None:
            columns = ['power_usage', 'temperature', 'rain', 'wind', 'humidity', 'sun', 'solar']
            columns = [col for col in columns if col in self.train.columns]
        
        outliers = {}
        
        for col in columns:
            if method == 'iqr':
                Q1 = self.train[col].quantile(0.25)
                Q3 = self.train[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers[col] = self.train[(self.train[col] < lower_bound) | 
                                         (self.train[col] > upper_bound)].index
            
            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(self.train[col].dropna()))
                outliers[col] = self.train.loc[self.train[col].dropna().index[z_scores > threshold]].index
        
        total_outliers = sum(len(indices) for indices in outliers.values())
        logging.info(f"이상치 탐지 완료 - 총 {total_outliers}개 이상치 발견")
        
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
        
        logging.info("전체 전처리 파이프라인 완료!")
        
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
