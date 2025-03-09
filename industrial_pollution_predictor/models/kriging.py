# industrial_pollution_predictor/models/kriging.py
"""
Kriging模型，用于预测污染物浓度的空间分布。
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
    SKLEARN_GP_AVAILABLE = True
except ImportError:
    SKLEARN_GP_AVAILABLE = False
    print("警告: scikit-learn的高斯过程模块未安装，Kriging模型将不可用")

try:
    import pykriging
    PYKRIGING_AVAILABLE = True
except ImportError:
    PYKRIGING_AVAILABLE = False
    if not SKLEARN_GP_AVAILABLE:
        print("警告: pykriging也未安装，将使用简单代替方法")


class KrigingModel:
    """
    Kriging模型，用于预测污染物浓度的空间分布
    """
    
    def __init__(self, kernel_type='matern', nugget=1e-5, **kwargs):
        """
        初始化Kriging模型
        
        参数:
            kernel_type: 核函数类型，'rbf', 'matern'或'exponential'
            nugget: 噪声参数
            **kwargs: 其他参数
        """
        self.kernel_type = kernel_type
        self.nugget = nugget
        self.model = None
        self.scaler = StandardScaler()
        self.trained = False
        self.feature_importance_ = None
        self._coords_idx = [0, 1]  # 假设前两列是坐标
    def fit(self, X, y, graph=None, **kwargs):
        """
        训练Kriging模型
        
        参数:
            X: 特征矩阵
            y: 目标变量
            graph: 图结构（对Kriging无效）
            **kwargs: 其他参数
            
        返回:
            self: 返回实例本身
        """
        # 提取坐标列（通常是前两列）
        coords_idx = kwargs.get('coords_idx', self._coords_idx)
        coords = X[:, coords_idx]
        
        # 数据标准化
        coords_scaled = self.scaler.fit_transform(coords)
        
        # 创建并训练模型
        if SKLEARN_GP_AVAILABLE:
            # 使用scikit-learn的高斯过程回归
            if self.kernel_type == 'rbf':
                kernel = RBF() + WhiteKernel(noise_level=self.nugget)
            elif self.kernel_type == 'matern':
                kernel = Matern() + WhiteKernel(noise_level=self.nugget)
            else:
                kernel = RBF() + WhiteKernel(noise_level=self.nugget)
                
            self.model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=kwargs.get('alpha', 1e-10),
                n_restarts_optimizer=kwargs.get('n_restarts_optimizer', 5),
                random_state=kwargs.get('random_state', 42)
            )
            
            self.model.fit(coords_scaled, y)
            
        elif PYKRIGING_AVAILABLE:
            # 使用pykriging库
            self.model = pykriging.kriging(coords_scaled, y)
            self.model.train()
            
        else:
            # 简单的替代方法：使用距离加权插值
            from scipy.spatial import distance
            self.model = {
                'coords': coords_scaled,
                'values': y,
                'power': kwargs.get('power', 2)  # 距离权重的幂
            }
        
        # 计算"特征重要性"（对于Kriging，只考虑坐标重要性）
        self.feature_importance_ = np.zeros(X.shape[1])
        for idx in coords_idx:
            self.feature_importance_[idx] = 1.0
        
        self.trained = True
        return self
    
    def predict(self, X, graph=None, **kwargs):
        """
        预测目标变量
        
        参数:
            X: 特征矩阵
            graph: 图结构（对Kriging无效）
            **kwargs: 其他参数
            
        返回:
            预测结果
        """
        if not self.trained:
            raise ValueError("模型尚未训练")
            
        # 提取坐标列
        coords_idx = kwargs.get('coords_idx', self._coords_idx)
        coords = X[:, coords_idx]
        
        # 坐标标准化
        coords_scaled = self.scaler.transform(coords)
        
        # 预测
        if SKLEARN_GP_AVAILABLE and isinstance(self.model, GaussianProcessRegressor):
            return self.model.predict(coords_scaled)
            
        elif PYKRIGING_AVAILABLE and 'pykriging' in str(type(self.model)):
            return self.model.predict(coords_scaled)
            
        else:
            # 简单的距离加权插值
            n_points = coords_scaled.shape[0]
            predictions = np.zeros(n_points)
            
            train_coords = self.model['coords']
            train_values = self.model['values']
            power = self.model['power']
            
            for i in range(n_points):
                # 计算到所有训练点的距离
                dist = distance.cdist([coords_scaled[i]], train_coords).flatten()
                
                # 避免除以零
                dist = np.maximum(dist, 1e-10)
                
                # 计算权重
                weights = 1 / (dist ** power)
                weights /= np.sum(weights)
                
                # 加权平均
                predictions[i] = np.sum(weights * train_values)
            
            return predictions
    
    def predict_uncertainty(self, X, graph=None, **kwargs):
        """
        预测不确定性
        
        参数:
            X: 特征矩阵
            graph: 图结构（对Kriging无效）
            **kwargs: 其他参数
            
        返回:
            不确定性估计
        """
        if not self.trained:
            raise ValueError("模型尚未训练")
            
        # 提取坐标列
        coords_idx = kwargs.get('coords_idx', self._coords_idx)
        coords = X[:, coords_idx]
        
        # 坐标标准化
        coords_scaled = self.scaler.transform(coords)
        
        # 预测不确定性
        if SKLEARN_GP_AVAILABLE and isinstance(self.model, GaussianProcessRegressor):
            _, std = self.model.predict(coords_scaled, return_std=True)
            return std
            
        elif PYKRIGING_AVAILABLE and 'pykriging' in str(type(self.model)):
            # pykriging可能有自己的不确定性度量
            if hasattr(self.model, 'predict_variance'):
                var = self.model.predict_variance(coords_scaled)
                return np.sqrt(var)
            else:
                # 回退到基于距离的简单不确定性估计
                return self._distance_based_uncertainty(coords_scaled)
            
        else:
            # 基于距离的简单不确定性估计
            return self._distance_based_uncertainty(coords_scaled)
    
    def _distance_based_uncertainty(self, coords_scaled):
        """
        基于到最近训练点的距离计算不确定性
        
        参数:
            coords_scaled: 标准化的坐标
            
        返回:
            不确定性估计
        """
        train_coords = self.model['coords'] if isinstance(self.model, dict) else self.model.X_train_
        
        # 计算到最近训练点的距离
        from scipy.spatial import distance
        min_distances = np.zeros(coords_scaled.shape[0])
        
        for i in range(coords_scaled.shape[0]):
            # 计算到所有训练点的距离
            dist = distance.cdist([coords_scaled[i]], train_coords).flatten()
            # 记录最小距离
            min_distances[i] = np.min(dist)
        
        # 将距离转换为不确定性：距离越远，不确定性越大
        uncertainty = min_distances
        
        # 归一化
        if len(uncertainty) > 1:
            uncertainty = (uncertainty - np.min(uncertainty)) / (np.max(uncertainty) - np.min(uncertainty) + 1e-10)
            # 缩放到合理范围
            uncertainty = 0.1 + 0.3 * uncertainty
        
        return uncertainty