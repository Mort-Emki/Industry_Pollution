# industrial_pollution_predictor/feature_extractor/traditional.py
"""
传统特征提取器，用于提取基于空间分析的特征。
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import distance
from typing import Dict, Any, List, Union, Optional


class TraditionalFeatureExtractor:
    """传统特征提取器，用于提取基于空间分析的特征"""
    
    def __init__(self):
        """初始化传统特征提取器"""
        pass
        
    def extract(self, monitoring_data, infrastructure_data, risk_data=None, **kwargs):
        """
        从监测点位和基础设施数据中提取特征
        
        参数:
            monitoring_data: 监测点位数据
            infrastructure_data: 基础设施数据
            risk_data: 风险评估数据
            **kwargs: 其他参数
            
        返回:
            包含特征的字典
        """
        # 1. 获取目标污染物数据
        target_pollutant = kwargs.get('target_pollutant', 'benzene')
        if target_pollutant not in monitoring_data.columns:
            raise ValueError(f"监测点位数据中缺少目标污染物列: {target_pollutant}")
            
        # 检查是否有缺失值
        if monitoring_data[target_pollutant].isna().any():
            print(f"警告: 目标污染物 {target_pollutant} 有缺失值，将被排除")
            monitoring_data = monitoring_data.dropna(subset=[target_pollutant])
            
        # 2. 提取特征
        features = []
        feature_names = []
        
        # 2.1 位置特征
        features.append(monitoring_data[['x', 'y']].values)
        feature_names.extend(['x', 'y'])
        
        # 2.2 与基础设施的距离特征
        for infra_type, infra_data in infrastructure_data.items():
            # 计算到最近基础设施的距离
            min_distances = self._calculate_min_distances(monitoring_data, infra_data)
            features.append(min_distances.reshape(-1, 1))
            feature_names.append(f'dist_to_{infra_type}')
            
            # 如果基础设施有类型属性，计算到各类型的距离
            if 'type' in infra_data.columns:
                for infra_subtype in infra_data['type'].unique():
                    infra_subset = infra_data[infra_data['type'] == infra_subtype]
                    min_distances = self._calculate_min_distances(monitoring_data, infra_subset)
                    features.append(min_distances.reshape(-1, 1))
                    feature_names.append(f'dist_to_{infra_type}_{infra_subtype}')
        
        # 2.3 风险评估数据（如果有）
        if risk_data is not None:
            if isinstance(risk_data, dict) and 'data' in risk_data:  # 栅格数据
                risk_values = self._extract_raster_values(
                    monitoring_data, 
                    risk_data['data'], 
                    risk_data['transform']
                )
                features.append(risk_values.reshape(-1, 1))
                feature_names.append('risk_value')
            elif isinstance(risk_data, gpd.GeoDataFrame):  # 矢量数据
                # 假设风险数据有一个'risk_level'属性
                if 'risk_level' in risk_data.columns:
                    # 空间连接以获取每个监测点的风险等级
                    joined = gpd.sjoin(monitoring_data, risk_data, how='left', predicate='within')
                    # 将风险等级转换为数值
                    risk_levels = pd.factorize(joined['risk_level'])[0]
                    features.append(risk_levels.reshape(-1, 1))
                    feature_names.append('risk_level')
        
        # 2.4 时间特征（如果有）
        if 'year' in monitoring_data.columns:
            features.append(monitoring_data['year'].values.reshape(-1, 1))
            feature_names.append('year')
        elif 'date' in monitoring_data.columns or 'timestamp' in monitoring_data.columns:
            # 转换日期/时间戳为年份
            date_col = 'date' if 'date' in monitoring_data.columns else 'timestamp'
            years = pd.to_datetime(monitoring_data[date_col]).dt.year
            features.append(years.values.reshape(-1, 1))
            feature_names.append('year')
        
        # 2.5 附加特征（如果在kwargs中指定）
        for feature_name in kwargs.get('additional_features', []):
            if feature_name in monitoring_data.columns:
                features.append(monitoring_data[feature_name].values.reshape(-1, 1))
                feature_names.append(feature_name)
        
        # 合并所有特征
        X = np.hstack(features)
        y = monitoring_data[target_pollutant].values
        
        return {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'monitoring_points': monitoring_data
        }
    
    def extract_for_grid(self, grid, infrastructure_data, risk_data=None, **kwargs):
        """
        为预测网格提取特征
        
        参数:
            grid: 预测网格
            infrastructure_data: 基础设施数据
            risk_data: 风险评估数据
            **kwargs: 其他参数
            
        返回:
            包含网格特征的字典
        """
        # 1. 提取特征
        features = []
        
        # 1.1 位置特征
        features.append(grid[['x', 'y']].values)
        
        # 1.2 与基础设施的距离特征
        for infra_type, infra_data in infrastructure_data.items():
            # 计算到最近基础设施的距离
            min_distances = self._calculate_min_distances(grid, infra_data)
            features.append(min_distances.reshape(-1, 1))
            
            # 如果基础设施有类型属性，计算到各类型的距离
            if 'type' in infra_data.columns:
                for infra_subtype in infra_data['type'].unique():
                    infra_subset = infra_data[infra_data['type'] == infra_subtype]
                    min_distances = self._calculate_min_distances(grid, infra_subset)
                    features.append(min_distances.reshape(-1, 1))
        
        # 1.3 风险评估数据（如果有）
        if risk_data is not None:
            if isinstance(risk_data, dict) and 'data' in risk_data:  # 栅格数据
                risk_values = self._extract_raster_values(
                    grid, 
                    risk_data['data'], 
                    risk_data['transform']
                )
                features.append(risk_values.reshape(-1, 1))
            elif isinstance(risk_data, gpd.GeoDataFrame):  # 矢量数据
                # 假设风险数据有一个'risk_level'属性
                if 'risk_level' in risk_data.columns:
                    # 空间连接以获取每个网格点的风险等级
                    joined = gpd.sjoin(grid, risk_data, how='left', predicate='within')
                    # 将风险等级转换为数值
                    risk_levels = pd.factorize(joined['risk_level'])[0]
                    features.append(risk_levels.reshape(-1, 1))
        
        # 1.4 时间特征（使用当前时间）
        if kwargs.get('include_year', True):
            import datetime
            current_year = datetime.datetime.now().year
            years = np.full((len(grid), 1), current_year)
            features.append(years)
        
        # 1.5 附加特征（如果在kwargs中指定）
        for feature_name in kwargs.get('additional_features', []):
            if feature_name in grid.columns:
                features.append(grid[feature_name].values.reshape(-1, 1))
        
        # 合并所有特征
        X = np.hstack(features)
        
        return {'X': X, 'grid': grid}
    
    def _calculate_min_distances(self, points_gdf, features_gdf):
        """计算点到最近要素的距离"""
        min_distances = np.full(len(points_gdf), np.inf)
        
        # 检查坐标参考系统是否一致
        if points_gdf.crs != features_gdf.crs:
            # 尝试转换features_gdf到points_gdf的坐标系
            try:
                features_gdf = features_gdf.to_crs(points_gdf.crs)
            except Exception as e:
                print(f"坐标参考系统转换失败: {e}")
                print("将使用未转换的坐标计算距离，结果可能不准确")
        
        # 提取点坐标
        points = np.vstack((points_gdf.geometry.x, points_gdf.geometry.y)).T
        
        # 计算到每个要素的距离，并取最小值
        for idx, feature in features_gdf.iterrows():
            if feature.geometry.geom_type == 'Point':
                # 点到点的距离
                feature_point = np.array([feature.geometry.x, feature.geometry.y])
                distances = distance.cdist(points, feature_point.reshape(1, -1)).flatten()
            else:
                # 点到线或面的距离
                distances = np.array([point.distance(feature.geometry) for point in points_gdf.geometry])
            
            # 更新最小距离
            min_distances = np.minimum(min_distances, distances)
        
        return min_distances
    
    def _extract_raster_values(self, points_gdf, raster_data, transform):
        """从栅格数据中提取点位置的值"""
        from rasterio.transform import rowcol
        
        # 提取点坐标
        x_coords = points_gdf.geometry.x
        y_coords = points_gdf.geometry.y
        
        # 计算点对应的栅格行列号
        rows, cols = rowcol(transform, x_coords, y_coords)
        
        # 创建掩码，确保行列号在栅格范围内
        height, width = raster_data.shape
        valid_mask = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        
        # 初始化结果数组
        values = np.full(len(points_gdf), np.nan)
        
        # 提取有效点的栅格值
        valid_rows = rows[valid_mask]
        valid_cols = cols[valid_mask]
        values[valid_mask] = raster_data[valid_rows, valid_cols]
        
        # 对无效点，使用最近有效值填充或使用平均值
        if (~valid_mask).any():
            if valid_mask.any():
                # 使用有效值的平均值填充
                values[~valid_mask] = np.nanmean(values[valid_mask])
            else:
                # 如果没有有效值，使用栅格平均值
                values[:] = np.nanmean(raster_data)
        
        return values