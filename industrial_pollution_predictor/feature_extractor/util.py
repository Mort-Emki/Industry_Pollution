# industrial_pollution_predictor/feature_extractor/utils.py
"""
特征工具函数，用于特征处理和选择。
"""

import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional


class FeatureUtils:
    """特征工具函数，用于特征处理和选择"""
    
    def __init__(self):
        """初始化特征工具类"""
        self.scaler = StandardScaler()
        self.selected_indices = None
        
    def merge_features(self, *features):
        """
        合并多个特征矩阵
        
        参数:
            *features: 要合并的特征矩阵
            
        返回:
            合并后的特征矩阵
        """
        if not features:
            return np.array([])
            
        # 确保所有特征矩阵的样本数相同
        n_samples = features[0].shape[0]
        for i, X in enumerate(features):
            if X.shape[0] != n_samples:
                raise ValueError(f"特征矩阵 {i} 的样本数 {X.shape[0]} 与第一个特征矩阵的样本数 {n_samples} 不一致")
                
        # 水平合并特征
        return np.hstack(features)
    
    def select_features(self, X, y, method='mutual_info', k=10) -> Tuple[np.ndarray, List[int]]:
        """
        特征选择
        
        参数:
            X: 特征矩阵
            y: 目标变量
            method: 特征选择方法，'mutual_info'或'f_regression'
            k: 选择的特征数量
            
        返回:
            选择后的特征矩阵和所选特征的索引
        """
        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_regression, k=min(k, X.shape[1]))
        elif method == 'f_regression':
            selector = SelectKBest(f_regression, k=min(k, X.shape[1]))
        else:
            raise ValueError(f"不支持的特征选择方法: {method}")
            
        # 应用特征选择
        X_selected = selector.fit_transform(X, y)
        
        # 获取所选特征的索引
        self.selected_indices = np.where(selector.get_support())[0].tolist()
        
        return X_selected, self.selected_indices
    
    def scale_features(self, X_train, X_test=None):
        """
        特征标准化
        
        参数:
            X_train: 训练特征矩阵
            X_test: 测试特征矩阵（可选）
            
        返回:
            标准化后的特征矩阵
        """
        # 对训练数据进行拟合和转换
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 如果提供了测试数据，对其进行转换
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        else:
            return X_train_scaled