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
    
    def rename_columns(self, column_mappings: Dict[str, List[str]]) -> None:
        """
        데이터프레임 컬럼명을 영어로 변환
        
        Args:
            column_mappings: 각 데이터프레임별 새로운 컬럼명 리스트
                          예: {'train': ['col1', 'col2', ...], 'test': [...], 'building': [...]}
        """
        logging.info("컬럼명 영어 변환 중...")
        
        if 'train' in column_mappings and self.train is not None:
            self.train.columns = column_mappings['train']
        if 'test' in column_mappings and self.test is not None:
            self.test.columns = column_mappings['test']
        if 'building' in column_mappings and self.building is not None:
            self.building.columns = column_mappings['building']
            
        logging.info("컬럼명 변환 완료")
    
    def map_values(self, target_df: str, column: str, value_mapping: Dict[str, str]) -> None:
        """
        특정 데이터프레임의 특정 컬럼 값을 매핑
        
        Args:
            target_df: 대상 데이터프레임 ('train', 'test', 'building')
            column: 매핑할 컬럼명
            value_mapping: 값 매핑 딕셔너리 {기존값: 새로운값, ...}
        """
        logging.info(f"{target_df}의 {column} 컬럼 값 매핑 중...")
        
        # 대상 데이터프레임 가져오기
        df = getattr(self, target_df, None)
        if df is None:
            logging.error(f"'{target_df}' 데이터프레임이 존재하지 않습니다.")
            return
            
        if column not in df.columns:
            logging.error(f"'{column}' 컬럼이 '{target_df}' 데이터프레임에 존재하지 않습니다.")
            return
            
        # 값 매핑 적용
        df[column] = df[column].map(value_mapping)
        
        # 매핑되지 않은 값 확인
        unmapped_values = df[df[column].isnull()][column].unique()
        if len(unmapped_values) > 0:
            logging.warning(f"매핑되지 않은 값들: {unmapped_values}")
        
        logging.info(f"{column} 컬럼 값 매핑 완료")
    
    def preprocess_datetime(self) -> None:
        """
        날짜/시간 특성 추출 및 전처리
        - datetime 변환
        - 월, 일, 시간 추출
        - 주말/평일 구분
        """
        logging.info("날짜/시간 특성 추출 중...")
        for df_name in ['train', 'test']:
            df = getattr(self, df_name, None)
            if df is not None:
                # datetime 변환 (YYYYMMDD HH 형태 -> datetime)
                df['date_time'] = pd.to_datetime(df['date_time'], format='%Y%m%d %H')
            
                # 시간 특성 추출
                df['month'] = df['date_time'].dt.month
                df['day'] = df['date_time'].dt.day
                df['hour'] = df['date_time'].dt.hour
                df['day_of_week'] = df['date_time'].dt.dayofweek
                df['date'] = df['date_time'].dt.date
                
                # 주말/평일 구분
                df['weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

                # 공휴일 표시 (2024-06-06, 2024-08-15)
                holiday_dates = ['2024-06-06', '2024-08-15']
                df['is_holiday'] = df['date_time'].dt.date.isin([
                    pd.to_datetime(date).date() for date in holiday_dates
                ]).astype(int)
        
        logging.info("날짜/시간 특성 추출 완료")

    def sin_cosine_date_features(self) -> None:
        """
        시간 변수의 순환적 성격을 반영하기 위해 sine, cosine 함수 적용
        - 시간대(0~23시), 날짜, 월, 요일을 sine/cosine 값으로 변환
        """
        logging.info("시간 변수의 순환적 성격 반영 중...")
        
        for df_name in ['train', 'test']:
            df = getattr(self, df_name, None)
            if df is not None and 'hour' in df.columns:
                # 시간
                df['sin_hour'] = np.sin(2 * np.pi * df['hour']/23.0)
                df['cos_hour'] = np.cos(2 * np.pi * df['hour']/23.0)
                
                # 날짜
                df['sin_date'] = -np.sin(2 * np.pi * (df['month']+df['day']/31)/12)
                df['cos_date'] = -np.cos(2 * np.pi * (df['month']+df['day']/31)/12)
                
                # 월
                df['sin_month'] = -np.sin(2 * np.pi * df['month']/12.0)
                df['cos_month'] = -np.cos(2 * np.pi * df['month']/12.0)

                # 요일
                df['sin_dayofweek'] = -np.sin(2 * np.pi * (df['day_of_week']+1)/7.0)
                df['cos_dayofweek'] = -np.cos(2 * np.pi * (df['day_of_week']+1)/7.0)
        
        logging.info("시간 변수의 순환적 성격 반영 완료")

    def drop_columns(self, target_df: str, columns: List[str]) -> None:
        """
        지정된 컬럼을 특정 데이터프레임에서 제거

        Args:
            target_df: 대상 데이터프레임 이름 ('train', 'test', 'building' 등). None이면 train/test 모두 적용
            columns: 제거할 컬럼 리스트
        """
        logging.info(f"제거할 컬럼: {columns} (대상: {target_df if target_df else 'train, test'})")
        
        dfs = []
        if target_df is None:
            dfs = [self.train, self.test]
        else:
            df = getattr(self, target_df, None)
            if df is not None:
                dfs = [df]
            else:
                logging.warning(f"'{target_df}' 데이터프레임이 존재하지 않습니다.")
                return

        for df in dfs:
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
    
    def handle_missing_values(self, strategy: Dict[str, str] = None, target_df: str = None) -> None:
        """
        결측값 처리
        
        Args:
            strategy: 컬럼별 결측값 처리 전략 딕셔너리
                     예: {'solar_capacity': 'median', 'ess_capacity': 'zero'}
            target_df: 결측값을 처리할 데이터프레임 이름 ('train', 'test', 'building')
                      None이면 모든 데이터프레임에 적용
        """
        logging.info("결측값 처리 중...")
        
        if strategy is None:
            logging.warning("결측값 처리 전략이 지정되지 않았습니다.")
            return
        
        # 타겟 데이터프레임 선택
        dataframes = []
        if target_df is not None:
            if not hasattr(self, target_df) or getattr(self, target_df) is None:
                logging.error(f"'{target_df}' 데이터프레임이 존재하지 않습니다.")
                return
            dataframes = [getattr(self, target_df)]

        # 각 데이터프레임에 결측값 처리 적용
        for df in dataframes:
            for col, method in strategy.items():
                if col in df.columns and df[col].isnull().sum() > 0:
                    missing_count = df[col].isnull().sum()
                    
                    if method == 'zero':
                        df[col].fillna(0, inplace=True)
                    elif method == 'mean':
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif method == 'median':
                        df[col].fillna(df[col].median(), inplace=True)
                    elif method == 'mode':
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    elif method == 'ffill':
                        df[col].fillna(method='ffill', inplace=True)
                    elif method == 'bfill':
                        df[col].fillna(method='bfill', inplace=True)
                    
                    logging.info(f"컬럼 '{col}'의 결측값 {missing_count}개를 '{method}' 방식으로 처리했습니다.")
        
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