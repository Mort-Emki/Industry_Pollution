# industrial_pollution_predictor/utils/evaluation.py
"""
评估工具，用于评估模型性能。
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Any, Optional


class Evaluator:
    """
    评估工具，用于评估模型性能
    """
    
    def __init__(self):
        """初始化评估工具"""
        pass
    
    def evaluate(self, model, X, y, **kwargs):
        """
        评估模型性能
        
        参数:
            model: 模型对象
            X: 特征矩阵
            y: 目标变量
            **kwargs: 其他参数
            
        返回:
            包含评估指标的字典
        """
        # 获取预测结果
        if hasattr(model, 'predict'):
            if kwargs.get('use_cv', False):
                # 使用交叉验证评估
                from sklearn.model_selection import cross_val_predict
                cv = kwargs.get('cv', 5)
                y_pred = cross_val_predict(model, X, y, cv=cv)
            else:
                # 使用模型的预测方法
                if 'graph' in kwargs:
                    y_pred = model.predict(X, graph=kwargs['graph'])
                else:
                    y_pred = model.predict(X)
        else:
            raise ValueError("模型缺少predict方法")
            
        # 计算评估指标
        results = {}
        
        # 均方根误差
        results['rmse'] = np.sqrt(mean_squared_error(y, y_pred))
        
        # 决定系数
        results['r2'] = r2_score(y, y_pred)
        
        # 平均绝对误差
        results['mae'] = mean_absolute_error(y, y_pred)
        
        # 平均绝对百分比误差
        # 避免除以零
        nonzero_mask = y != 0
        if np.any(nonzero_mask):
            results['mape'] = np.mean(np.abs((y[nonzero_mask] - y_pred[nonzero_mask]) / y[nonzero_mask])) * 100
        else:
            results['mape'] = np.nan
        
        # 计算各分位数的误差
        errors = np.abs(y - y_pred)
        results['median_error'] = np.median(errors)
        results['q90_error'] = np.percentile(errors, 90)
        results['max_error'] = np.max(errors)
        
        return results
    
    def evaluate_temporal(self, model, X, y, **kwargs):
        """
        评估时序预测性能
        
        参数:
            model: 模型对象
            X: 特征矩阵 [samples, time_steps, features]
            y: 目标变量 [samples, time_steps]
            **kwargs: 其他参数
            
        返回:
            包含评估指标的字典
        """
        # 检查输入维度
        if len(X.shape) != 3 or len(y.shape) != 2:
            raise ValueError("时序评估需要3D特征和2D目标")
            
        # 获取时间步数
        time_steps = X.shape[1]
        
        # 存储每个时间步的评估结果
        temporal_results = {}
        
        for t in range(time_steps):
            # 获取当前时间步的特征和目标
            X_t = X[:, t, :]
            y_t = y[:, t]
            
            # 获取预测结果
            if 'graph' in kwargs:
                y_pred_t = model.predict(X_t, graph=kwargs['graph'])
            else:
                y_pred_t = model.predict(X_t)
                
            # 计算评估指标
            results_t = {}
            
            # 均方根误差
            results_t['rmse'] = np.sqrt(mean_squared_error(y_t, y_pred_t))
            
            # 决定系数
            results_t['r2'] = r2_score(y_t, y_pred_t)
            
            # 平均绝对误差
            results_t['mae'] = mean_absolute_error(y_t, y_pred_t)
            
            # 存储结果
            temporal_results[f'time_step_{t}'] = results_t
        
        # 计算所有时间步的平均指标
        avg_results = {
            'temporal_avg_rmse': np.mean([results_t['rmse'] for results_t in temporal_results.values()]),
            'temporal_avg_r2': np.mean([results_t['r2'] for results_t in temporal_results.values()]),
            'temporal_avg_mae': np.mean([results_t['mae'] for results_t in temporal_results.values()]),
            'temporal_results': temporal_results
        }
        
        return avg_results