# industrial_pollution_predictor/temporal_processor.py
"""
时序数据处理模块，用于处理和分析时序监测数据。
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union


class TemporalProcessor:
    """
    时序数据处理器，用于处理和分析时序监测数据
    """
    
    def __init__(self):
        """初始化时序数据处理器"""
        self.latest_features = None
        self.start_timestamp = None
        self.time_interval = None
    
    def process(self, features, monitoring_data, time_steps=None, **kwargs):
        """
        处理时序特征
        
        参数:
            features: 特征字典，包含'X', 'y'和'monitoring_points'
            monitoring_data: 监测点位数据
            time_steps: 时间步数，如果为None则自动确定
            **kwargs: 其他参数
            
        返回:
            处理后的特征字典
        """
        # 检查监测数据是否有时间列
        time_cols = ['timestamp', 'date', 'year']
        time_col = next((col for col in time_cols if col in monitoring_data.columns), None)
        
        if time_col is None:
            raise ValueError("监测数据缺少时间列 (timestamp/date/year)")
            
        # 将时间列转换为日期时间格式
        if time_col == 'year':
            # 如果只有年份，将其转换为该年的1月1日
            dates = pd.to_datetime(monitoring_data[time_col].astype(str) + '-01-01')
        else:
            dates = pd.to_datetime(monitoring_data[time_col])
        
        # 将日期添加到监测点数据中
        monitoring_data['datetime'] = dates
        
        # 确定时间间隔
        if len(dates) > 1:
            # 计算最常见的时间间隔
            intervals = dates.sort_values().diff().dropna()
            if intervals.empty:
                self.time_interval = timedelta(days=365)  # 默认为1年
            else:
                # 获取最常见的时间间隔
                self.time_interval = intervals.mode()[0]
        else:
            self.time_interval = timedelta(days=365)  # 默认为1年
        
        # 记录开始时间
        self.start_timestamp = dates.min()
        
        # 如果未指定时间步数，根据数据确定
        if time_steps is None:
            # 计算唯一时间点的数量
            unique_dates = dates.unique()
            time_steps = len(unique_dates)
            
            # 限制最大时间步数
            time_steps = min(time_steps, 12)
        
        # 提取各时间步的特征
        temporal_X = []
        temporal_y = []
        temporal_points = []
        
        for t in range(time_steps):
            # 计算当前时间点
            current_time = self.start_timestamp + t * self.time_interval
            
            # 查找当前时间点的监测数据
            mask = (monitoring_data['datetime'] == current_time)
            
            if mask.any():
                # 获取当前时间点的特征和目标
                current_points = monitoring_data[mask]
                current_X = features['X'][mask]
                current_y = features['y'][mask]
                
                temporal_X.append(current_X)
                temporal_y.append(current_y)
                temporal_points.append(current_points)
        
        # 将时序特征转换为适合RNN/GCN的格式
        if len(temporal_X) > 0:
            # 检查所有时间步的样本数是否相同
            if not all(x.shape[0] == temporal_X[0].shape[0] for x in temporal_X):
                # 如果不同，需要对齐
                min_samples = min(x.shape[0] for x in temporal_X)
                temporal_X = [x[:min_samples] for x in temporal_X]
                temporal_y = [y[:min_samples] for y in temporal_y]
                temporal_points = [p.iloc[:min_samples] for p in temporal_points]
            
            # 转换为3D数组: [samples, time_steps, features]
            X = np.stack(temporal_X, axis=1)
            y = np.stack(temporal_y, axis=1)
            
            # 保存处理后的特征
            self.latest_features = {
                'X': X,
                'y': y,
                'monitoring_points': temporal_points
            }
            
            return self.latest_features
        else:
            # 如果没有足够的时序数据，返回原始特征
            print("警告: 没有足够的时序数据，将使用原始特征")
            return features
    
    def prepare_prediction(self, grid_features, prediction_horizon=1):
        """
        为时序预测准备特征
        
        参数:
            grid_features: 网格特征字典
            prediction_horizon: 预测时间步长
            
        返回:
            准备好的特征字典
        """
        # 添加时间编码
        if 'X' in grid_features:
            # 创建空的时序特征矩阵
            X = grid_features['X']
            batch_size = X.shape[0]
            
            # 创建时间编码
            time_encodings = []
            for t in range(prediction_horizon):
                # 可以使用正弦、余弦编码或简单的one-hot编码
                # 这里使用简单的标量值表示时间步
                time_encoding = np.full((batch_size, 1), t)
                time_encodings.append(time_encoding)
            
            # 将时间编码与特征合并
            temporal_X = []
            for t in range(prediction_horizon):
                # 将时间编码添加到特征中
                X_with_time = np.hstack([X, time_encodings[t]])
                temporal_X.append(X_with_time)
            
            # 转换为3D数组: [samples, time_steps, features]
            X_3d = np.stack(temporal_X, axis=1)
            
            # 更新特征
            grid_features['X'] = X_3d
            grid_features['prediction_horizon'] = prediction_horizon
        
        return grid_features
    
    def get_prediction_timestamps(self, prediction_horizon):
        """
        获取预测时间戳列表
        
        参数:
            prediction_horizon: 预测时间步长
            
        返回:
            时间戳列表
        """
        if self.start_timestamp is None or self.time_interval is None:
            # 如果未设置时间信息，使用当前时间和默认间隔
            self.start_timestamp = datetime.now()
            self.time_interval = timedelta(days=30)  # 默认为30天
        
        # 计算最后一个已知时间点
        last_known_time = self.start_timestamp
        
        # 生成预测时间戳
        timestamps = []
        for t in range(prediction_horizon):
            # 计算当前预测时间点
            current_time = last_known_time + (t + 1) * self.time_interval
            timestamps.append(current_time.strftime('%Y-%m-%d'))
        
        return timestamps
    
    def get_latest_features(self):
        """获取最新处理的特征"""
        return self.latest_features