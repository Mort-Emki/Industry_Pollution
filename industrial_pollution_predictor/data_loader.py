# industrial_pollution_predictor/data_loader.py
"""
数据加载模块，负责加载和预处理各类输入数据。
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from pathlib import Path
from typing import List, Dict, Union, Optional, Any


class DataLoader:
    """
    数据加载器，用于加载和预处理各类输入数据。
    """
    
    def __init__(self):
        """初始化数据加载器"""
        self.rs_data = None
        self.monitoring_data = None
        self.infrastructure_data = None
        self.risk_assessment = None
        self.study_area = None
        self.study_area_bounds = None
        
    def load_remote_sensing_data(self, rs_data_paths: List[str]) -> Dict[str, Any]:
        """
        加载遥感数据
        
        参数:
            rs_data_paths: 遥感图像路径列表
            
        返回:
            包含遥感数据的字典
        """
        if not rs_data_paths:
            raise ValueError("遥感数据路径列表不能为空")
            
        rs_data_dict = {}
        for path in rs_data_paths:
            # 检查文件是否存在
            if not os.path.exists(path):
                raise FileNotFoundError(f"遥感数据文件不存在: {path}")
                
            # 获取文件名作为键（不含扩展名）
            key = os.path.splitext(os.path.basename(path))[0]
            
            # 加载GeoTIFF文件
            with rasterio.open(path) as src:
                # 读取数据和元数据
                rs_data_dict[key] = {
                    'data': src.read(),
                    'meta': src.meta,
                    'bounds': src.bounds,
                    'transform': src.transform,
                    'crs': src.crs,
                    'path': path
                }
                
                # 更新研究区域边界
                if self.study_area_bounds is None:
                    self.study_area_bounds = src.bounds
                else:
                    # 扩展边界以包含所有遥感数据
                    self.study_area_bounds = (
                        min(self.study_area_bounds[0], src.bounds.left),
                        min(self.study_area_bounds[1], src.bounds.bottom),
                        max(self.study_area_bounds[2], src.bounds.right),
                        max(self.study_area_bounds[3], src.bounds.top)
                    )
        
        print(f"已加载 {len(rs_data_dict)} 个遥感数据文件")
        self.rs_data = rs_data_dict
        return rs_data_dict
    
    def load_monitoring_data(self, file_path: str, target_pollutant: str = None) -> pd.DataFrame:
        """
        加载监测点位数据
        
        参数:
            file_path: 监测点位数据CSV文件路径
            target_pollutant: 目标污染物名称
            
        返回:
            监测点位数据DataFrame
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"监测点位数据文件不存在: {file_path}")
            
        # 加载CSV文件
        df = pd.read_csv(file_path)
        
        # 检查必需的列
        required_cols = ['id', 'x', 'y']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"监测点位数据缺少必需列: {required_cols}")
            
        # 检查目标污染物列是否存在
        if target_pollutant and target_pollutant not in df.columns:
            raise ValueError(f"监测点位数据缺少目标污染物列: {target_pollutant}")
            
        # 检查时间列
        time_cols = ['year', 'date', 'timestamp']
        if not any(col in df.columns for col in time_cols):
            print("警告: 监测点位数据缺少时间列 (year/date/timestamp)")
            
        # 添加几何列，转换为GeoDataFrame
        geometry = gpd.points_from_xy(df['x'], df['y'])
        gdf = gpd.GeoDataFrame(df, geometry=geometry)
        
        # 检查数据量
        if len(gdf) == 0:
            print("警告: 监测点位数据为空")
        else:
            print(f"已加载 {len(gdf)} 个监测点位数据")
            
        self.monitoring_data = gdf
        return gdf
    
    def load_infrastructure_data(self, infra_data: Dict[str, str]) -> Dict[str, gpd.GeoDataFrame]:
        """
        加载基础设施数据
        
        参数:
            infra_data: 包含'pipelines'和'tanks'等路径的字典
            
        返回:
            包含基础设施数据的字典
        """
        if not isinstance(infra_data, dict):
            raise ValueError("基础设施数据必须是字典类型")
            
        infra_dict = {}
        for key, path in infra_data.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"基础设施数据文件不存在: {path}")
                
            # 加载Shapefile或GeoJSON
            if path.endswith('.shp') or path.endswith('.geojson'):
                try:
                    gdf = gpd.read_file(path)
                    infra_dict[key] = gdf
                    print(f"已加载 {len(gdf)} 个 {key} 要素")
                except Exception as e:
                    print(f"加载 {key} 基础设施数据时出错: {e}")
            else:
                print(f"不支持的基础设施数据格式: {path}")
        
        self.infrastructure_data = infra_dict
        return infra_dict
    
    def load_risk_assessment(self, file_path: str) -> Union[gpd.GeoDataFrame, Dict[str, Any]]:
        """
        加载风险评估数据
        
        参数:
            file_path: 风险评估数据文件路径
            
        返回:
            风险评估数据
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"风险评估数据文件不存在: {file_path}")
            
        # 根据文件扩展名加载不同格式的数据
        if file_path.endswith('.tif'):
            # 加载GeoTIFF
            with rasterio.open(file_path) as src:
                risk_data = {
                    'data': src.read(1),  # 假设只有一个波段
                    'meta': src.meta,
                    'bounds': src.bounds,
                    'transform': src.transform,
                    'crs': src.crs,
                    'path': file_path
                }
                print(f"已加载风险评估栅格数据: {file_path}")
        elif file_path.endswith('.shp') or file_path.endswith('.geojson'):
            # 加载Shapefile或GeoJSON
            risk_data = gpd.read_file(file_path)
            print(f"已加载 {len(risk_data)} 个风险评估矢量数据")
        else:
            raise ValueError(f"不支持的风险评估数据格式: {file_path}")
            
        self.risk_assessment = risk_data
        return risk_data
    
    def load_study_area(self, file_path: str) -> gpd.GeoDataFrame:
        """
        加载研究区域边界
        
        参数:
            file_path: 研究区域边界文件路径
            
        返回:
            研究区域边界GeoDataFrame
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"研究区域边界文件不存在: {file_path}")
            
        # 加载Shapefile或GeoJSON
        if file_path.endswith('.shp') or file_path.endswith('.geojson'):
            study_area = gpd.read_file(file_path)
            print(f"已加载研究区域边界: {file_path}")
            
            # 更新研究区域边界
            bounds = study_area.total_bounds
            self.study_area_bounds = (bounds[0], bounds[1], bounds[2], bounds[3])
            
            self.study_area = study_area
            return study_area
        else:
            raise ValueError(f"不支持的研究区域边界格式: {file_path}")
    
    def get_remote_sensing_data(self) -> Dict[str, Any]:
        """获取已加载的遥感数据"""
        return self.rs_data
    
    def get_monitoring_data(self) -> gpd.GeoDataFrame:
        """获取已加载的监测点位数据"""
        return self.monitoring_data
    
    def get_infrastructure_data(self) -> Dict[str, gpd.GeoDataFrame]:
        """获取已加载的基础设施数据"""
        return self.infrastructure_data
    
    def get_risk_assessment(self) -> Union[gpd.GeoDataFrame, Dict[str, Any]]:
        """获取已加载的风险评估数据"""
        return self.risk_assessment
    
    def get_study_area(self) -> gpd.GeoDataFrame:
        """获取已加载的研究区域边界"""
        return self.study_area
    
    def get_study_area_bounds(self) -> tuple:
        """获取研究区域边界范围"""
        return self.study_area_bounds