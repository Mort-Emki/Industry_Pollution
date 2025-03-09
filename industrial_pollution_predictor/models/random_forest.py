# industrial_pollution_predictor/models/random_forest.py
"""
随机森林模型，用于预测污染物浓度。
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional


class RandomForestModel:
    """
    随机森林模型，用于预测污染物浓度
    """
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42, **kwargs):
        """
        初始化随机森林模型
        
        参数:
            n_estimators: 树的数量
            max_depth: 树的最大深度
            random_state: 随机种子
            **kwargs: 其他参数
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.trained = False
        self.feature_importance_ = None
        
    def fit(self, X, y, graph=None, **kwargs):
        """
        训练随机森林模型
        
        参数:
            X: 特征矩阵
            y: 目标变量
            graph: 图结构（对随机森林无效）
            **kwargs: 其他参数
            
        返回:
            self: 返回实例本身
        """
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 创建并训练模型
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=kwargs.get('n_jobs', -1),
            **{k: v for k, v in kwargs.items() if k in RandomForestRegressor().get_params()}
        )
        
        self.model.fit(X_scaled, y)
        
        # 获取特征重要性
        self.feature_importance_ = self.model.feature_importances_
        
        self.trained = True
        return self
    
    def predict(self, X, graph=None, **kwargs):
        """
        预测目标变量
        
        参数:
            X: 特征矩阵
            graph: 图结构（对随机森林无效）
            **kwargs: 其他参数
            
        返回:
            预测结果
        """
        if not self.trained or self.model is None:
            raise ValueError("模型尚未训练")
            
        # 数据标准化
        X_scaled = self.scaler.transform(X)
        
        # 预测
        return self.model.predict(X_scaled)
    
    def predict_uncertainty(self, X, graph=None, **kwargs):
        """
        预测不确定性
        
        参数:
            X: 特征矩阵
            graph: 图结构（对随机森林无效）
            **kwargs: 其他参数
            
        返回:
            不确定性估计
        """
        if not self.trained or self.model is None:
            raise ValueError("模型尚未训练")
            
        # 数据标准化
        X_scaled = self.scaler.transform(X)
        
        # 使用随机森林的预测方差作为不确定性度量
        # 让每棵树单独预测，然后计算标准差
        predictions = np.array([tree.predict(X_scaled) 
                               for tree in self.model.estimators_])
        
        # 计算标准差
        uncertainty = np.std(predictions, axis=0)
        
        return uncertainty