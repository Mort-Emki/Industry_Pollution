# industrial_pollution_predictor/utils/io.py
"""
输入输出工具，用于数据的加载、转换和输出。
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union


class IOHandler:
    """
    输入输出工具，用于数据的加载、转换和输出
    """
    
    def __init__(self):
        """初始化输入输出工具"""
        pass
    
    def create_prediction_grid(self, bounds, resolution=10.0, crs='EPSG:4326'):
        """
        创建预测网格
        
        参数:
            bounds: 边界范围 (xmin, ymin, xmax, ymax)
            resolution: 网格分辨率
            crs: 坐标参考系统
            
        返回:
            网格点GeoDataFrame
        """
        if bounds is None:
            raise ValueError("边界范围不能为空")
            
        # 解析边界范围
        xmin, ymin, xmax, ymax = bounds
        
        # 计算网格点坐标
        x = np.arange(xmin, xmax, resolution)
        y = np.arange(ymin, ymax, resolution)
        
        # 创建网格点
        xx, yy = np.meshgrid(x, y)
        points = np.vstack([xx.flatten(), yy.flatten()]).T
        
        # 创建GeoDataFrame
        import shapely.geometry
        geometry = [shapely.geometry.Point(p) for p in points]
        grid = gpd.GeoDataFrame(geometry=geometry, crs=crs)
        
        # 添加坐标列
        grid['x'] = grid.geometry.x
        grid['y'] = grid.geometry.y
        
        # 添加分辨率信息
        grid.attrs['resolution'] = resolution
        
        return grid
    
    def points_to_raster(self, x, y, values, resolution=None, bounds=None):
        """
        将点数据转换为栅格数据
        
        参数:
            x: x坐标数组
            y: y坐标数组
            values: 值数组
            resolution: 栅格分辨率
            bounds: 边界范围 (xmin, ymin, xmax, ymax)
            
        返回:
            栅格形状、栅格数据和变换的元组
        """
        # 计算边界范围
        if bounds is None:
            xmin, xmax = x.min(), x.max()
            ymin, ymax = y.min(), y.max()
        else:
            xmin, ymin, xmax, ymax = bounds
        
        # 计算栅格分辨率
        if resolution is None:
            # 使用点之间的平均距离作为分辨率
            if len(x) > 1:
                dx = np.diff(np.sort(x))
                dy = np.diff(np.sort(y))
                resolution = min(np.median(dx[dx > 0]), np.median(dy[dy > 0]))
            else:
                resolution = 10.0  # 默认分辨率
        
        # 计算栅格大小
        width = int(np.ceil((xmax - xmin) / resolution))
        height = int(np.ceil((ymax - ymin) / resolution))
        
        # 创建空栅格
        raster_data = np.zeros((height, width), dtype=np.float32)
        
        # 填充栅格
        # 将点坐标转换为栅格索引
        col_indices = ((x - xmin) / resolution).astype(int)
        row_indices = ((ymax - y) / resolution).astype(int)
        
        # 确保索引在有效范围内
        valid_mask = (col_indices >= 0) & (col_indices < width) & (row_indices >= 0) & (row_indices < height)
        
        if np.any(valid_mask):
            # 将值填充到栅格中
            # 如果多个点映射到同一个栅格单元，取平均值
            for i in range(len(values)):
                if valid_mask[i]:
                    row, col = row_indices[i], col_indices[i]
                    raster_data[row, col] = values[i]
        
        # 创建坐标变换
        transform = from_origin(xmin, ymax, resolution, resolution)
        
        return (height, width), raster_data, transform
    
    def raster_to_points(self, raster_data, transform):
        """
        将栅格数据转换为点数据
        
        参数:
            raster_data: 栅格数据
            transform: 栅格变换
            
        返回:
            x坐标、y坐标和值的元组
        """
        # 获取栅格大小
        height, width = raster_data.shape
        
        # 创建行列索引
        rows, cols = np.mgrid[0:height, 0:width]
        
        # 将行列索引转换为坐标
        from rasterio.transform import xy
        xs, ys = xy(transform, rows.flatten(), cols.flatten())
        
        # 获取栅格值
        values = raster_data.flatten()
        
        return np.array(xs), np.array(ys), values
    
    def save_to_geojson(self, df, output_path):
        """
        将DataFrame保存为GeoJSON文件
        
        参数:
            df: DataFrame或GeoDataFrame
            output_path: 输出文件路径
            
        返回:
            输出文件路径
        """
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 如果是普通DataFrame，先转换为GeoDataFrame
        if not isinstance(df, gpd.GeoDataFrame):
            if 'x' in df.columns and 'y' in df.columns:
                import shapely.geometry
                geometry = [shapely.geometry.Point(x, y) for x, y in zip(df['x'], df['y'])]
                gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
            else:
                raise ValueError("DataFrame缺少坐标列")
        else:
            gdf = df.copy()
        
        # 保存为GeoJSON
        gdf.to_file(output_path, driver='GeoJSON')
        
        return output_path
    
    def save_to_csv(self, df, output_path):
        """
        将DataFrame保存为CSV文件
        
        参数:
            df: DataFrame
            output_path: 输出文件路径
            
        返回:
            输出文件路径
        """
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存为CSV
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def save_to_raster(self, grid, values, output_path, resolution=None):
        """
        将网格数据保存为GeoTIFF文件
        
        参数:
            grid: 网格点GeoDataFrame
            values: 值数组
            output_path: 输出文件路径
            resolution: 栅格分辨率
            
        返回:
            输出文件路径
        """
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 获取分辨率
        if resolution is None:
            resolution = grid.attrs.get('resolution', 10.0)
        
        # 获取坐标
        x = grid.geometry.x.values
        y = grid.geometry.y.values
        
        # 转换为栅格
        grid_shape, raster_data, transform = self.points_to_raster(x, y, values, resolution)
        
        # 保存为GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=grid_shape[0],
            width=grid_shape[1],
            count=1,
            dtype=raster_data.dtype,
            crs=grid.crs,
            transform=transform
        ) as dst:
            dst.write(raster_data, 1)
        
        return output_path