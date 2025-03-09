# industrial_pollution_predictor/models/ensemble.py
"""
集成模型，用于组合多个基础模型。
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union


class EnsembleModel:
    """
    集成模型，用于组合多个基础模型
    """
    
    def __init__(self, models=None, weights=None, **kwargs):
        """
        初始化集成模型
        
        参数:
            models: 基础模型列表
            weights: 模型权重列表，如果为None则自动确定
            **kwargs: 其他参数
        """
        self.models = models or []
        self.weights = weights
        self.trained = False
        self.feature_importance_ = None
        
    def fit(self, X, y, graph=None, **kwargs):
        """
        训练集成模型
        
        参数:
            X: 特征矩阵
            y: 目标变量
            graph: 图结构（对集成模型无效）
            **kwargs: 其他参数
            
        返回:
            self: 返回实例本身
        """
        if not self.models:
            raise ValueError("模型列表不能为空")
            
        # 训练每个基础模型
        for i, model in enumerate(self.models):
            print(f"训练基础模型 {i+1}/{len(self.models)}: {model.__class__.__name__}")
            model.fit(X, y, **kwargs)
        
        # 如果未指定权重，使用交叉验证确定最优权重
        if self.weights is None:
            self.weights = self._optimize_weights(X, y, **kwargs)
        
        # 计算加权特征重要性
        self._compute_feature_importance()
        
        self.trained = True
        return self
    
    def predict(self, X, graph=None, **kwargs):
        """
        预测目标变量
        
        参数:
            X: 特征矩阵
            graph: 图结构（对集成模型无效）
            **kwargs: 其他参数
            
        返回:
            预测结果
        """
        if not self.trained:
            raise ValueError("模型尚未训练")
            
        # 获取每个基础模型的预测结果
        predictions = []
        for model in self.models:
            y_pred = model.predict(X, **kwargs)
            predictions.append(y_pred)
        
        # 计算加权平均
        predictions = np.array(predictions)
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        
        return weighted_pred
    
    def predict_uncertainty(self, X, graph=None, **kwargs):
        """
        预测不确定性
        
        参数:
            X: 特征矩阵
            graph: 图结构（对集成模型无效）
            **kwargs: 其他参数
            
        返回:
            不确定性估计
        """
        if not self.trained:
            raise ValueError("模型尚未训练")
            
        # 获取每个基础模型的预测结果
        predictions = []
        for model in self.models:
            # 如果模型支持不确定性预测，使用模型的不确定性
            if hasattr(model, 'predict_uncertainty'):
                y_pred = model.predict(X, **kwargs)
            else:
                y_pred = model.predict(X, **kwargs)
            predictions.append(y_pred)
        
        # 计算预测的标准差作为不确定性估计
        predictions = np.array(predictions)
        uncertainty = np.std(predictions, axis=0)
        
        return uncertainty
    
    def _optimize_weights(self, X, y, n_folds=5, **kwargs):
        """
        使用交叉验证优化模型权重
        
        参数:
            X: 特征矩阵
            y: 目标变量
            n_folds: 交叉验证折数
            **kwargs: 其他参数
            
        返回:
            最优模型权重
        """
        from sklearn.model_selection import KFold
        from scipy.optimize import minimize
        
        # 使用K折交叉验证
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # 存储每个模型在每个折上的预测结果
        fold_predictions = [[] for _ in range(len(self.models))]
        fold_targets = []
        
        # 执行交叉验证
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 记录验证集目标变量
            fold_targets.append(y_val)
            
            # 训练每个模型并记录预测结果
            for i, model in enumerate(self.models):
                model.fit(X_train, y_train, **kwargs)
                y_pred = model.predict(X_val, **kwargs)
                fold_predictions[i].append(y_pred)
        
        # 将列表转换为数组
        fold_predictions = [np.concatenate(preds) for preds in fold_predictions]
        fold_targets = np.concatenate(fold_targets)
        
        # 定义目标函数：最小化RMSE
        def objective(weights):
            # 归一化权重
            weights = weights / np.sum(weights)
            
            # 计算加权平均预测
            weighted_pred = np.zeros_like(fold_targets)
            for i in range(len(self.models)):
                weighted_pred += weights[i] * fold_predictions[i]
            
            # 计算RMSE
            rmse = np.sqrt(np.mean((weighted_pred - fold_targets) ** 2))
            return rmse
        
        # 使用优化算法找到最优权重
        n_models = len(self.models)
        initial_weights = np.ones(n_models) / n_models  # 初始权重相等
        bounds = [(0, 1) for _ in range(n_models)]      # 权重在0到1之间
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # 权重和为1
        
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        # 返回最优权重
        optimal_weights = result.x / np.sum(result.x)  # 再次归一化
        print(f"优化后的模型权重: {optimal_weights}")
        
        return optimal_weights
    
    def _compute_feature_importance(self):
        """计算加权特征重要性"""
        if not all(hasattr(model, 'feature_importance_') for model in self.models):
            print("警告: 部分模型不提供特征重要性，无法计算加权特征重要性")
            return
        
        # 获取特征重要性并计算加权平均
        feature_importances = []
        for i, model in enumerate(self.models):
            feature_importances.append(model.feature_importance_)
        
        feature_importances = np.array(feature_importances)
        self.feature_importance_ = np.average(feature_importances, axis=0, weights=self.weights)