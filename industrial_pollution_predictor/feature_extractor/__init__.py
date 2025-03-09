# industrial_pollution_predictor/feature_extractor/__init__.py
"""
特征提取模块，用于从各类输入数据中提取特征。
"""

from .traditional import TraditionalFeatureExtractor
from .cnn import CNNFeatureExtractor
from .utils import FeatureUtils

class FeatureExtractor:
    """
    特征提取器，用于从各类输入数据中提取特征。
    """
    
    def __init__(self, use_cnn=True):
        """
        初始化特征提取器
        
        参数:
            use_cnn (bool): 是否使用CNN特征提取
        """
        self.use_cnn = use_cnn
        self.traditional_extractor = TraditionalFeatureExtractor()
        self.cnn_extractor = CNNFeatureExtractor() if use_cnn else None
        self.utils = FeatureUtils()
        self.feature_names = []
        self.latest_features = None
        
    def extract_features(self, rs_data, monitoring_data, infrastructure_data,
                         risk_data=None, study_area=None, **kwargs):
        """
        从输入数据中提取特征
        
        参数:
            rs_data: 遥感数据
            monitoring_data: 监测点位数据
            infrastructure_data: 基础设施数据
            risk_data: 风险评估数据
            study_area: 研究区域边界
            **kwargs: 其他参数
            
        返回:
            包含特征的字典
        """
        # 1. 提取传统特征
        trad_features = self.traditional_extractor.extract(
            monitoring_data=monitoring_data,
            infrastructure_data=infrastructure_data,
            risk_data=risk_data,
            **kwargs
        )
        
        # 2. 提取CNN特征（如果启用）
        if self.use_cnn and rs_data is not None:
            cnn_features = self.cnn_extractor.extract(
                rs_data=rs_data,
                monitoring_data=monitoring_data,
                patch_size=kwargs.get('cnn_patch_size', 32),
                **kwargs
            )
            
            # 合并特征
            X = self.utils.merge_features(trad_features['X'], cnn_features['X'])
            feature_names = trad_features['feature_names'] + cnn_features['feature_names']
        else:
            X = trad_features['X']
            feature_names = trad_features['feature_names']
        
        # 3. 特征选择（如果需要）
        if kwargs.get('feature_selection', False):
            X, selected_indices = self.utils.select_features(
                X, trad_features['y'], 
                method=kwargs.get('feature_selection_method', 'mutual_info'),
                k=kwargs.get('feature_selection_k', min(20, X.shape[1]))
            )
            feature_names = [feature_names[i] for i in selected_indices]
        
        # 保存特征名称和最新特征
        self.feature_names = feature_names
        self.latest_features = {
            'X': X, 
            'y': trad_features['y'],
            'monitoring_points': trad_features['monitoring_points']
        }
        
        return self.latest_features
    
    def extract_grid_features(self, grid, rs_data, infrastructure_data, risk_data=None, **kwargs):
        """
        为预测网格提取特征
        
        参数:
            grid: 预测网格
            rs_data: 遥感数据
            infrastructure_data: 基础设施数据
            risk_data: 风险评估数据
            **kwargs: 其他参数
            
        返回:
            包含网格特征的字典
        """
        # 1. 提取传统特征
        trad_features = self.traditional_extractor.extract_for_grid(
            grid=grid,
            infrastructure_data=infrastructure_data,
            risk_data=risk_data,
            **kwargs
        )
        
        # 2. 提取CNN特征（如果启用）
        if self.use_cnn and rs_data is not None:
            cnn_features = self.cnn_extractor.extract_for_grid(
                rs_data=rs_data,
                grid=grid,
                patch_size=kwargs.get('cnn_patch_size', 32),
                **kwargs
            )
            
            # 合并特征
            X = self.utils.merge_features(trad_features['X'], cnn_features['X'])
        else:
            X = trad_features['X']
        
        # 3. 特征选择（需要与训练时保持一致）
        if kwargs.get('feature_selection', False) and hasattr(self, 'selected_indices'):
            X = X[:, self.selected_indices]
        
        return {'X': X, 'grid': grid}
    
    def get_feature_names(self):
        """获取特征名称列表"""
        return self.feature_names
    
    def get_latest_features(self):
        """获取最新提取的特征"""
        return self.latest_features