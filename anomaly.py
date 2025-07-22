import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    """
    전력소비량 데이터에 특화된 이상치 탐지 클래스
    
    - IQR 기반 (건물별, 시간대별)
    - Z-score 기반 (건물별, 요일/시간대별)
    - Rolling window 기반
    - 계절성 분해 기반 (STL, Seasonal Decompose)
    - Isolation Forest (머신러닝 기반)
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        이상치 탐지기 초기화
        
        Args:
            data: 전처리된 전력소비량 데이터프레임
        """
        self.data = data.copy()
        self.outliers = {}
        self.outlier_summary = {}
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 필요한 컬럼 확인
        required_cols = ['building_num', 'date_time', 'power_usage']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"필수 컬럼이 누락되었습니다: {missing_cols}")
    
    def detect_anomalies(self, columns: List[str] = None, method: str = 'rolling', 
                        threshold: float = 3.0, **kwargs) -> Dict[str, pd.Series]:
        """
        이상치 탐지 메인 메소드
        
        Args:
            columns: 이상치를 탐지할 컬럼 리스트
            method: 탐지 방법 ('iqr', 'zscore', 'rolling', 'seasonal', 'stl', 'isolation_forest')
            threshold: 이상치 임계값
            **kwargs: 각 방법별 추가 파라미터
        
        Returns:
            컬럼별 이상치 인덱스 딕셔너리
        """
        if columns is None:
            columns = ['power_usage']
            # 사용 가능한 추가 컬럼들
            for col in ['temperature', 'humidity', 'solar']:
                if col in self.data.columns:
                    columns.append(col)
        
        # 적절한 탐지 방법 호출
        if method == 'iqr':
            self.outliers = self._detect_iqr_outliers(columns, threshold, **kwargs)
        elif method == 'zscore':
            self.outliers = self._detect_zscore_outliers(columns, threshold, **kwargs)
        elif method == 'rolling':
            window = kwargs.pop('window', 24)  # pop을 사용해서 kwargs에서 제거
            self.outliers = self._detect_rolling_outliers(columns, threshold, window, **kwargs)
        elif method == 'seasonal':
            self.outliers = self._detect_seasonal_outliers(columns, threshold, **kwargs)
        elif method == 'stl':
            self.outliers = self._detect_stl_outliers(columns, threshold, **kwargs)
        elif method == 'isolation_forest':
            self.outliers = self._detect_isolation_forest_outliers(columns, threshold, **kwargs)
        else:
            raise ValueError(f"지원하지 않는 방법입니다: {method}")
        
        self._create_summary()
        return self.outliers
    
    def _detect_iqr_outliers(self, columns: List[str], threshold: float = 1.5, **kwargs) -> Dict[str, pd.Series]:
        """
        IQR 방법으로 이상치 탐지 (건물별, 시간대별)
        
        Args:
            columns: 분석할 컬럼들
            threshold: IQR 배수 임계값
            
        Returns:
            컬럼별 이상치 인덱스
        """
        self.logger.info(f"IQR 방법으로 이상치 탐지 중 (임계값: {threshold})...")
        outliers = {}
        
        for col in columns:
            if col not in self.data.columns:
                continue
                
            outliers[col] = pd.Series(dtype='int64')
            
            # 각 건물별, 시간대별로 IQR 기반 이상치 탐지
            for building_num in self.data['building_num'].unique():
                for hour in range(24):
                    subset = self.data[
                        (self.data['building_num'] == building_num) & 
                        (self.data['hour'] == hour)
                    ][col]
                    
                    if len(subset) < 10:  # 충분한 데이터가 없으면 건너뜀
                        continue
                    
                    Q1 = subset.quantile(0.25)
                    Q3 = subset.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    if IQR == 0:  # IQR이 0이면 건너뜀
                        continue
                    
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    outlier_mask = (subset < lower_bound) | (subset > upper_bound)
                    outlier_indices = subset[outlier_mask].index
                    
                    if len(outlier_indices) > 0:
                        outliers[col] = pd.concat([outliers[col], pd.Series(outlier_indices)])
        
        # 중복 제거
        for col in outliers:
            outliers[col] = pd.Series(outliers[col].unique())
        
        total_outliers = sum(len(indices) for indices in outliers.values())
        self.logger.info(f"IQR 이상치 탐지 완료 - 총 {total_outliers}개 이상치 발견")
        
        return outliers
    
    def _detect_zscore_outliers(self, columns: List[str], threshold: float = 3.0, **kwargs) -> Dict[str, pd.Series]:
        """
        Z-Score 방법으로 이상치 탐지 (건물별, 요일/시간대별)
        
        Args:
            columns: 분석할 컬럼들
            threshold: Z-Score 임계값
            
        Returns:
            컬럼별 이상치 인덱스
        """
        self.logger.info(f"Z-Score 방법으로 이상치 탐지 중 (임계값: {threshold})...")
        
        try:
            from scipy import stats
        except ImportError:
            self.logger.error("scipy가 설치되지 않아 Z-Score 방법을 사용할 수 없습니다.")
            return {}
        
        outliers = {}
        
        for col in columns:
            if col not in self.data.columns:
                continue
                
            outliers[col] = pd.Series(dtype='int64')
            
            # 건물별, 요일/시간대별 Z-score 계산
            for building_num in self.data['building_num'].unique():
                # day_type이 있으면 사용, 없으면 전체 데이터 사용
                day_types = ['weekday', 'weekend'] if 'day_type' in self.data.columns else [None]
                
                for day_type in day_types:
                    for hour in range(24):
                        if day_type is not None:
                            subset = self.data[
                                (self.data['building_num'] == building_num) &
                                (self.data['day_type'] == day_type) &
                                (self.data['hour'] == hour)
                            ][col]
                        else:
                            subset = self.data[
                                (self.data['building_num'] == building_num) &
                                (self.data['hour'] == hour)
                            ][col]
                        
                        if len(subset) < 10:
                            continue
                        
                        z_scores = np.abs(stats.zscore(subset.dropna()))
                        outlier_indices = subset.dropna().index[z_scores > threshold]
                        
                        if len(outlier_indices) > 0:
                            outliers[col] = pd.concat([outliers[col], pd.Series(outlier_indices)])
        
        # 중복 제거
        for col in outliers:
            outliers[col] = pd.Series(outliers[col].unique())
        
        total_outliers = sum(len(indices) for indices in outliers.values())
        self.logger.info(f"Z-Score 이상치 탐지 완료 - 총 {total_outliers}개 이상치 발견")
        
        return outliers
    
    def _detect_rolling_outliers(self, columns: List[str], threshold: float = 3.0, window: int = 24, **kwargs) -> Dict[str, pd.Series]:
        """
        Rolling Window 방법으로 이상치 탐지 (건물별 시계열)
        
        Args:
            columns: 분석할 컬럼들
            threshold: 표준편차 배수 임계값
            window: 롤링 윈도우 크기
            
        Returns:
            컬럼별 이상치 인덱스
        """
        self.logger.info(f"Rolling Window 방법으로 이상치 탐지 중 (윈도우: {window}, 임계값: {threshold})...")
        outliers = {}
        
        for col in columns:
            if col not in self.data.columns:
                continue
                
            outliers[col] = pd.Series(dtype='int64')
            
            # 건물별 시계열 롤링 윈도우 기반 이상치 탐지
            for building_num in self.data['building_num'].unique():
                building_data = self.data[self.data['building_num'] == building_num].sort_values('date_time')
                
                if len(building_data) < window:
                    continue
                
                # 롤링 평균과 표준편차 계산
                rolling_mean = building_data[col].rolling(window, min_periods=max(5, window//4), center=True).mean()
                rolling_std = building_data[col].rolling(window, min_periods=max(5, window//4), center=True).std()
                
                # 차이 계산
                diff = np.abs(building_data[col] - rolling_mean)
                outlier_mask = diff > (threshold * rolling_std)
                
                # NaN 처리
                outlier_mask = outlier_mask.fillna(False)
                outlier_indices = building_data.loc[outlier_mask].index
                
                if len(outlier_indices) > 0:
                    outliers[col] = pd.concat([outliers[col], pd.Series(outlier_indices)])
        
        # 중복 제거
        for col in outliers:
            outliers[col] = pd.Series(outliers[col].unique())
        
        total_outliers = sum(len(indices) for indices in outliers.values())
        self.logger.info(f"Rolling Window 이상치 탐지 완료 - 총 {total_outliers}개 이상치 발견")
        
        return outliers
    
    def _detect_seasonal_outliers(self, columns: List[str], threshold: float = 3.0, **kwargs) -> Dict[str, pd.Series]:
        """
        계절성 분해 방법으로 이상치 탐지
        
        Args:
            columns: 분석할 컬럼들
            threshold: 잔차 표준편차 배수 임계값
            
        Returns:
            컬럼별 이상치 인덱스
        """
        self.logger.info(f"계절성 분해 방법으로 이상치 탐지 중 (임계값: {threshold})...")
        
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
        except ImportError:
            self.logger.error("statsmodels가 설치되지 않아 계절성 분해 방법을 사용할 수 없습니다.")
            return {}
        
        outliers = {}
        
        for col in columns:
            if col not in self.data.columns:
                continue
                
            outliers[col] = pd.Series(dtype='int64')
            
            # 건물별 계절성 분해
            for building_num in self.data['building_num'].unique():
                building_data = self.data[self.data['building_num'] == building_num].sort_values('date_time')
                
                # 충분한 데이터가 있는지 확인 (최소 2일치)
                if len(building_data) < 48:
                    continue
                
                try:
                    # 일간 주기(24시간)로 분해
                    decomposition = seasonal_decompose(
                        building_data[col].dropna(), 
                        period=24, 
                        model='additive',
                        extrapolate_trend='freq'
                    )
                    residual = decomposition.resid
                    
                    # 잔차의 표준편차를 기준으로 이상치 탐지
                    std_resid = np.std(residual.dropna())
                    if std_resid > 0:
                        outlier_mask = np.abs(residual) > threshold * std_resid
                        outlier_indices = building_data.loc[residual.index[outlier_mask]].index
                        
                        if len(outlier_indices) > 0:
                            outliers[col] = pd.concat([outliers[col], pd.Series(outlier_indices)])
                    
                except Exception as e:
                    self.logger.warning(f"건물 {building_num}의 계절성 분해 실패: {e}")
                    continue
        
        # 중복 제거
        for col in outliers:
            outliers[col] = pd.Series(outliers[col].unique())
        
        total_outliers = sum(len(indices) for indices in outliers.values())
        self.logger.info(f"계절성 분해 이상치 탐지 완료 - 총 {total_outliers}개 이상치 발견")
        
        return outliers
    
    def _detect_stl_outliers(self, columns: List[str], threshold: float = 3.0, **kwargs) -> Dict[str, pd.Series]:
        """
        STL 분해 방법으로 이상치 탐지
        
        Args:
            columns: 분석할 컬럼들
            threshold: 잔차 표준편차 배수 임계값
            
        Returns:
            컬럼별 이상치 인덱스
        """
        self.logger.info(f"STL 분해 방법으로 이상치 탐지 중 (임계값: {threshold})...")
        
        try:
            from statsmodels.tsa.seasonal import STL
        except ImportError:
            self.logger.error("statsmodels STL 모듈이 설치되지 않아 사용할 수 없습니다.")
            return {}
        
        outliers = {}
        
        for col in columns:
            if col not in self.data.columns:
                continue
                
            outliers[col] = pd.Series(dtype='int64')
            
            # 건물별 STL 분해
            for building_num in self.data['building_num'].unique():
                building_data = self.data[self.data['building_num'] == building_num].sort_values('date_time')
                
                # 충분한 데이터가 있는지 확인 (STL은 더 많은 데이터 필요)
                if len(building_data) < 72:  # 최소 3일치
                    continue
                
                try:
                    # STL 분해 수행
                    stl = STL(
                        building_data[col].dropna(),
                        period=24,
                        seasonal=13,  # 계절성 윈도우
                        trend=25,     # 트렌드 윈도우
                        robust=True   # 이상치에 강건한 분해
                    )
                    result = stl.fit()
                    
                    # 잔차 기반 이상치 탐지
                    residual = result.resid
                    resid_std = np.std(residual.dropna())
                    
                    if resid_std > 0:
                        outlier_mask = np.abs(residual) > threshold * resid_std
                        outlier_indices = building_data.loc[residual.index[outlier_mask]].index
                        
                        if len(outlier_indices) > 0:
                            outliers[col] = pd.concat([outliers[col], pd.Series(outlier_indices)])
                    
                    self.logger.debug(f"건물 {building_num} STL 분해 완료: {sum(outlier_mask)}개 이상치 발견")
                    
                except Exception as e:
                    self.logger.warning(f"건물 {building_num}의 STL 분해 실패: {e}")
                    continue
        
        # 중복 제거
        for col in outliers:
            outliers[col] = pd.Series(outliers[col].unique())
        
        total_outliers = sum(len(indices) for indices in outliers.values())
        self.logger.info(f"STL 분해 이상치 탐지 완료 - 총 {total_outliers}개 이상치 발견")
        
        return outliers
    
    def _detect_isolation_forest_outliers(self, columns: List[str], threshold: float = 0.05, **kwargs) -> Dict[str, pd.Series]:
        """
        Isolation Forest 방법으로 이상치 탐지
        
        Args:
            columns: 분석할 컬럼들
            threshold: contamination 비율 (이상치 비율)
            
        Returns:
            컬럼별 이상치 인덱스
        """
        self.logger.info(f"Isolation Forest 방법으로 이상치 탐지 중 (contamination: {threshold})...")
        
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            self.logger.error("scikit-learn이 설치되지 않아 Isolation Forest 방법을 사용할 수 없습니다.")
            return {}
        
        outliers = {}
        
        for col in columns:
            if col not in self.data.columns:
                continue
                
            outliers[col] = pd.Series(dtype='int64')
            
            # 건물별 Isolation Forest 적용
            for building_num in self.data['building_num'].unique():
                building_data = self.data[self.data['building_num'] == building_num]
                
                if len(building_data) < 50:  # 데이터가 충분하지 않으면 건너뜀
                    continue
                
                # 특성 선택 (시간 특성 + 관련 수치 특성)
                features = ['hour']
                if 'day_of_week' in building_data.columns:
                    features.append('day_of_week')
                
                # 추가 수치 특성들
                for feat in ['temperature', 'humidity', 'solar', col]:
                    if feat in building_data.columns:
                        features.append(feat)
                
                try:
                    X = building_data[features].fillna(method='ffill').fillna(method='bfill')
                    
                    if X.isnull().any().any():  # 여전히 NaN이 있으면 건너뜀
                        continue
                    
                    model = IsolationForest(
                        contamination=threshold,
                        random_state=42,
                        n_estimators=100
                    )
                    predictions = model.fit_predict(X)
                    outlier_indices = building_data.index[predictions == -1]
                    
                    if len(outlier_indices) > 0:
                        outliers[col] = pd.concat([outliers[col], pd.Series(outlier_indices)])
                
                except Exception as e:
                    self.logger.warning(f"건물 {building_num}의 Isolation Forest 실패: {e}")
                    continue
        
        # 중복 제거
        for col in outliers:
            outliers[col] = pd.Series(outliers[col].unique())
        
        total_outliers = sum(len(indices) for indices in outliers.values())
        self.logger.info(f"Isolation Forest 이상치 탐지 완료 - 총 {total_outliers}개 이상치 발견")
        
        return outliers
    
    def visualize_anomalies(self, columns: List[str] = None, building_nums: List[int] = None,
                           figsize: Tuple[int, int] = (15, 10), save_path: str = None):
        """
        이상치 시각화 (건물 유형별로 나누고, 각 유형 내에서 3개씩 건물 그룹화)
        
        Args:
            columns: 시각화할 컬럼 리스트
            building_nums: 시각화할 건물 번호 리스트 (None이면 전체)
            figsize: 그래프 크기
            save_path: 저장 경로 템플릿 (None이면 reports/figures/anomalies_{}_group{}.png로 저장)
        """
        if save_path is None:
            save_path = "reports/figures/anomalies_{}_group{}.png"
            
        # 한글 폰트 설정
        plt.rc("font", family = "Malgun Gothic")
        sns.set_theme(font="Malgun Gothic")

        if not self.outliers:
            self.logger.warning("이상치가 탐지되지 않았습니다. detect_anomalies()를 먼저 실행하세요.")
            return
        
        if columns is None:
            columns = list(self.outliers.keys())
        
        if building_nums is None:
            building_nums = sorted(self.data['building_num'].unique())
        
        try:
            # 건물 정보 읽기 및 건물 유형별 그룹화
            building_info = pd.read_csv("data/power_usage/raw/building_info.csv")
            building_types = building_info.set_index('building_num')['building_type']
            
            # 건물 유형별로 건물 그룹화
            buildings_by_type = {}
            for building_num in building_nums:
                building_type = building_types.get(building_num, 'Unknown')
                if building_type not in buildings_by_type:
                    buildings_by_type[building_type] = []
                buildings_by_type[building_type].append(building_num)

            # 각 건물 유형별로 시각화
            for building_type, type_buildings in buildings_by_type.items():
                # 3개씩 건물 그룹화
                building_groups = [type_buildings[i:i+3] for i in range(0, len(type_buildings), 3)]
                
                # 각 그룹별로 시각화
                for group_idx, group_buildings in enumerate(building_groups, 1):
                    n_buildings = len(group_buildings)
                    n_cols = len(columns)
                    
                    # 서브플롯 생성
                    fig, axes = plt.subplots(n_buildings, n_cols, figsize=figsize, squeeze=False)
                    fig.suptitle(f'건물 유형: {building_type} - 그룹 {group_idx}', fontsize=14)
                    
                    # 각 건물별 시각화
                    for i, building_num in enumerate(group_buildings):
                        building_data = self.data[self.data['building_num'] == building_num]
                        
                        for j, col in enumerate(columns):
                            ax = axes[i, j]
                            
                            # 정상 데이터 플롯
                            ax.plot(building_data['date_time'], building_data[col], 
                                   'b-', alpha=0.7, label='정상 데이터')
                            
                            # 이상치 플롯
                            if col in self.outliers and len(self.outliers[col]) > 0:
                                outlier_data = building_data[building_data.index.isin(self.outliers[col])]
                                if len(outlier_data) > 0:
                                    ax.scatter(outlier_data['date_time'], outlier_data[col], 
                                             color='red', s=50, label='이상치', zorder=5)
                            
                            # 그래프 설정
                            ax.set_title(f'건물 {building_num}번 - {col}')
                            ax.set_xlabel('시간')
                            ax.set_ylabel(col)
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            
                            # x축 레이블 회전
                            plt.setp(ax.get_xticklabels(), rotation=45)
                    
                    # 서브플롯 간격 조정
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    
                    # 그래프 저장
                    current_save_path = save_path.format(building_type, group_idx)
                    plt.savefig(current_save_path, bbox_inches='tight', dpi=300)
                    plt.close()
                    
                    self.logger.info(f"그래프 저장 완료: {current_save_path}")
                    
        except FileNotFoundError:
            self.logger.warning("building_info.csv 파일을 찾을 수 없습니다. 건물 유형 정보 없이 시각화를 진행합니다.")
            # 건물 유형 정보 없이 기본 시각화 수행
            building_groups = [building_nums[i:i+3] for i in range(0, len(building_nums), 3)]
            
            for group_num, group_buildings in enumerate(building_groups, 1):
                n_buildings = len(group_buildings)
                n_cols = len(columns)
                
                fig, axes = plt.subplots(n_buildings, n_cols, figsize=figsize, squeeze=False)
                fig.suptitle(f'건물 그룹 {group_num}', fontsize=14)
                
                for i, building_num in enumerate(group_buildings):
                    building_data = self.data[self.data['building_num'] == building_num]
                    
                    for j, col in enumerate(columns):
                        ax = axes[i, j]
                        
                        # 정상 데이터 플롯
                        ax.plot(building_data['date_time'], building_data[col], 
                               'b-', alpha=0.7, label='정상 데이터')
                        
                        # 이상치 플롯
                        if col in self.outliers and len(self.outliers[col]) > 0:
                            outlier_data = building_data[building_data.index.isin(self.outliers[col])]
                            if len(outlier_data) > 0:
                                ax.scatter(outlier_data['date_time'], outlier_data[col], 
                                         color='red', s=50, label='이상치', zorder=5)
                        
                        # 그래프 설정
                        ax.set_title(f'건물 {building_num}번 - {col}')
                        ax.set_xlabel('시간')
                        ax.set_ylabel(col)
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        # x축 레이블 회전
                        plt.setp(ax.get_xticklabels(), rotation=45)
                
                # 서브플롯 간격 조정
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                # 그래프 저장
                current_save_path = save_path.format('unknown', group_num)
                plt.savefig(current_save_path, bbox_inches='tight', dpi=300)
                plt.close()
                
                self.logger.info(f"그래프 저장 완료: {current_save_path}")
        
        except Exception as e:
            self.logger.error(f"시각화 중 오류 발생: {e}")
            raise
    
    def plot_anomaly_distribution(self, figsize: Tuple[int, int] = (12, 8)):
        """
        이상치 분포 시각화
        """
        if not self.outliers:
            self.logger.warning("이상치가 탐지되지 않았습니다.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # 1. 건물별 이상치 개수
        building_outlier_counts = {}
        for col, outlier_idx in self.outliers.items():
            if len(outlier_idx) > 0:
                outlier_data = self.data.loc[outlier_idx]
                building_counts = outlier_data['building_num'].value_counts()
                for building, count in building_counts.items():
                    if building not in building_outlier_counts:
                        building_outlier_counts[building] = 0
                    building_outlier_counts[building] += count
        
        if building_outlier_counts:
            buildings = list(building_outlier_counts.keys())
            counts = list(building_outlier_counts.values())
            axes[0].bar(buildings, counts)
            axes[0].set_title('건물별 이상치 개수')
            axes[0].set_xlabel('건물 번호')
            axes[0].set_ylabel('이상치 개수')
        
        # 2. 시간대별 이상치 분포
        hourly_outlier_counts = {}
        for col, outlier_idx in self.outliers.items():
            if len(outlier_idx) > 0:
                outlier_data = self.data.loc[outlier_idx]
                if 'hour' in outlier_data.columns:
                    hour_counts = outlier_data['hour'].value_counts().sort_index()
                    for hour, count in hour_counts.items():
                        if hour not in hourly_outlier_counts:
                            hourly_outlier_counts[hour] = 0
                        hourly_outlier_counts[hour] += count
        
        if hourly_outlier_counts:
            hours = sorted(hourly_outlier_counts.keys())
            counts = [hourly_outlier_counts[h] for h in hours]
            axes[1].plot(hours, counts, marker='o')
            axes[1].set_title('시간대별 이상치 분포')
            axes[1].set_xlabel('시간')
            axes[1].set_ylabel('이상치 개수')
            axes[1].grid(True)
        
        # 3. 요일별 이상치 분포 (가능한 경우)
        if 'day_of_week' in self.data.columns:
            daily_outlier_counts = {}
            for col, outlier_idx in self.outliers.items():
                if len(outlier_idx) > 0:
                    outlier_data = self.data.loc[outlier_idx]
                    day_counts = outlier_data['day_of_week'].value_counts()
                    for day, count in day_counts.items():
                        if day not in daily_outlier_counts:
                            daily_outlier_counts[day] = 0
                        daily_outlier_counts[day] += count
            
            if daily_outlier_counts:
                days = list(daily_outlier_counts.keys())
                counts = list(daily_outlier_counts.values())
                axes[2].bar(days, counts)
                axes[2].set_title('요일별 이상치 분포')
                axes[2].set_xlabel('요일')
                axes[2].set_ylabel('이상치 개수')
        
        # 4. 이상치 크기 분포
        power_outliers = self.outliers.get('power_usage', pd.Series())
        if len(power_outliers) > 0:
            outlier_values = self.data.loc[power_outliers, 'power_usage']
            axes[3].hist(outlier_values, bins=30, alpha=0.7, edgecolor='black')
            axes[3].set_title('전력소비량 이상치 값 분포')
            axes[3].set_xlabel('전력소비량')
            axes[3].set_ylabel('빈도')
        
        plt.tight_layout()
        plt.show()
    
    def remove_outliers(self, outlier_indices: Dict[str, pd.Series] = None) -> pd.DataFrame:
        """
        이상치 제거
        
        Args:
            outlier_indices: 미리 탐지된 이상치 인덱스 (None이면 현재 탐지된 이상치 사용)
            
        Returns:
            이상치가 제거된 데이터프레임
        """
        if outlier_indices is None:
            outlier_indices = self.outliers
        
        if outlier_indices:
            # 모든 이상치 인덱스 합집합
            all_outlier_indices = set()
            for indices in outlier_indices.values():
                all_outlier_indices.update(indices)
            
            original_shape = self.data.shape[0]
            cleaned_data = self.data.drop(list(all_outlier_indices))
            
            self.logger.info(f"이상치 제거 완료 - {original_shape}개 → {cleaned_data.shape[0]}개 "
                           f"({len(all_outlier_indices)}개 제거)")
            
            return cleaned_data
        else:
            self.logger.info("제거할 이상치가 없습니다.")
            return self.data.copy()
    
    def get_anomaly_summary(self) -> Dict:
        """
        이상치 탐지 결과 요약 반환
        """
        if not self.outliers:
            return {"message": "이상치가 탐지되지 않았습니다."}
        
        return self.outlier_summary
    
    def _create_summary(self):
        """
        이상치 탐지 결과 요약 생성
        """
        summary = {
            'total_data_points': len(self.data),
            'columns_analyzed': list(self.outliers.keys()),
            'outliers_by_column': {},
            'total_outliers': 0,
            'outlier_percentage': 0,
            'buildings_affected': set(),
            'outlier_by_building': {}
        }
        
        for col, outlier_idx in self.outliers.items():
            n_outliers = len(outlier_idx)
            summary['outliers_by_column'][col] = n_outliers
            summary['total_outliers'] += n_outliers
            
            if n_outliers > 0:
                outlier_data = self.data.loc[outlier_idx]
                affected_buildings = outlier_data['building_num'].unique()
                summary['buildings_affected'].update(affected_buildings)
                
                # 건물별 이상치 개수
                building_counts = outlier_data['building_num'].value_counts().to_dict()
                for building, count in building_counts.items():
                    if building not in summary['outlier_by_building']:
                        summary['outlier_by_building'][building] = 0
                    summary['outlier_by_building'][building] += count
        
        summary['buildings_affected'] = list(summary['buildings_affected'])
        summary['outlier_percentage'] = (summary['total_outliers'] / summary['total_data_points']) * 100
        
        self.outlier_summary = summary
    
    def export_outliers(self, filepath: str, format: str = 'csv'):
        """
        이상치 데이터 내보내기
        
        Args:
            filepath: 저장 경로
            format: 파일 형식 ('csv', 'excel')
        """
        if not self.outliers:
            self.logger.warning("내보낼 이상치가 없습니다.")
            return
        
        # 모든 이상치 인덱스 수집
        all_outlier_indices = set()
        for indices in self.outliers.values():
            all_outlier_indices.update(indices)
        
        outlier_data = self.data.loc[list(all_outlier_indices)].copy()
        
        # 이상치 유형 표시
        outlier_data['anomaly_type'] = ''
        for col, indices in self.outliers.items():
            mask = outlier_data.index.isin(indices)
            current_types = outlier_data.loc[mask, 'anomaly_type']
            new_types = current_types.apply(lambda x: f"{x},{col}" if x else col)
            outlier_data.loc[mask, 'anomaly_type'] = new_types
        
        if format.lower() == 'csv':
            outlier_data.to_csv(filepath, index=False, encoding='utf-8-sig')
        elif format.lower() == 'excel':
            outlier_data.to_excel(filepath, index=False)
        else:
            raise ValueError("지원되는 형식: 'csv', 'excel'")
        
        self.logger.info(f"이상치 데이터 내보내기 완료: {filepath}")


# 사용 예시 및 유틸리티 함수
def compare_anomaly_methods(data: pd.DataFrame, columns: List[str] = None, 
                           methods: List[str] = None) -> Dict:
    """
    여러 이상치 탐지 방법 비교
    
    Args:
        data: 입력 데이터
        columns: 분석할 컬럼들
        methods: 비교할 방법들
    
    Returns:
        방법별 결과 비교
    """
    if methods is None:
        methods = ['rolling', 'iqr', 'zscore', 'seasonal']
    
    if columns is None:
        columns = ['power_usage']
    
    results = {}
    
    for method in methods:
        try:
            detector = AnomalyDetector(data)
            outliers = detector.detect_anomalies(columns=columns, method=method)
            summary = detector.get_anomaly_summary()
            results[method] = {
                'outliers': outliers,
                'summary': summary,
                'success': True
            }
        except Exception as e:
            results[method] = {
                'error': str(e),
                'success': False
            }
    
    return results

def create_anomaly_report(data: pd.DataFrame, output_path: str = "anomaly_report.html"):
    """
    이상치 분석 보고서 생성
    
    Args:
        data: 입력 데이터
        output_path: 보고서 저장 경로
    """
    detector = AnomalyDetector(data)
    
    # 여러 방법으로 이상치 탐지
    methods = ['rolling', 'iqr', 'seasonal']
    method_results = {}
    
    for method in methods:
        try:
            outliers = detector.detect_anomalies(method=method)
            method_results[method] = detector.get_anomaly_summary()
        except Exception as e:
            print(f"{method} 방법 실패: {e}")
    
    # HTML 보고서 생성
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>전력소비량 이상치 분석 보고서</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .summary {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>전력소비량 이상치 분석 보고서</h1>
        <div class="summary">
            <h2>데이터 개요</h2>
            <p>총 데이터 포인트: {len(data):,}개</p>
            <p>분석 기간: {data['date_time'].min()} ~ {data['date_time'].max()}</p>
            <p>건물 수: {data['building_num'].nunique()}개</p>
        </div>
    """
    
    for method, summary in method_results.items():
        html_content += f"""
        <div class="summary">
            <h2>{method.upper()} 방법 결과</h2>
            <p>총 이상치: {summary.get('total_outliers', 0):,}개 ({summary.get('outlier_percentage', 0):.2f}%)</p>
            <p>영향받은 건물: {len(summary.get('buildings_affected', []))}개</p>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"이상치 분석 보고서 생성 완료: {output_path}")


if __name__ == "__main__":
    # 사용 예시
    print("AnomalyDetector 클래스 로드 완료!")
    print("사용법:")
    print("1. detector = AnomalyDetector(data)")
    print("2. outliers = detector.detect_anomalies(method='rolling')")
    print("3. detector.visualize_anomalies()")
    print("4. cleaned_data = detector.remove_outliers()")
    print("\n지원하는 탐지 방법:")
    print("- 'iqr': IQR 기반 (건물별, 시간대별)")
    print("- 'zscore': Z-Score 기반 (건물별, 요일/시간대별)")
    print("- 'rolling': Rolling Window 기반 (추천)")
    print("- 'seasonal': 계절성 분해 기반")
    print("- 'stl': STL 분해 기반 (고급)")
    print("- 'isolation_forest': Isolation Forest (머신러닝 기반)")
