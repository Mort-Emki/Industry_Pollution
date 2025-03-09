# industrial_pollution_predictor/predictor.py
"""
污染预测主接口模块，提供统一的API用于训练、预测和评估模型。
"""

import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import List, Dict, Union, Optional, Any, Tuple

from .data_loader import DataLoader
from .feature_extractor import FeatureExtractor
from .models import EnsembleModel, RandomForestModel, XGBoostModel, KrigingModel
from .models import CNNModel, GCNModel
from .graph_builder import GraphBuilder
from .temporal_processor import TemporalProcessor
from .utils.visualization import Visualizer
from .utils.evaluation import Evaluator
from .utils.io import IOHandler


class PollutionPredictor:
    """
    工业集聚区多介质复合污染预测工具
    
    参数:
        version (str): 使用的算法版本，'simplified'或'full'
        use_cnn (bool): 是否使用CNN特征提取
        use_gcn (bool): 是否使用图神经网络(仅完整版可用)
        model_path (str, optional): 预训练模型路径，如果提供则加载模型
    """
    
    def __init__(self, version='simplified', use_cnn=True, use_gcn=False, model_path=None):
        self.version = version
        self.use_cnn = use_cnn
        self.use_gcn = use_gcn and version == 'full'  # 只有完整版可用GCN
        self.data_loader = DataLoader()
        self.feature_extractor = FeatureExtractor(use_cnn=use_cnn)
        self.io_handler = IOHandler()
        self.visualizer = Visualizer()
        self.evaluator = Evaluator()
        
        # 根据版本初始化组件
        if self.version == 'full':
            self.graph_builder = GraphBuilder()
            self.temporal_processor = TemporalProcessor()
        
        self.model = None
        self.trained = False
        self.prediction_results = None
        self.evaluation_results = None
        self.target_pollutant = None
        self.metadata = {}
        
        # 如果提供了模型路径，加载预训练模型
        if model_path:
            self.load_model(model_path)
    
    def train(self, rs_data_paths, monitoring_data_path, infrastructure_data,
              risk_assessment_path=None, study_area_path=None, target_pollutant='benzene',
              temporal_sequence=False, time_steps=None, graph_connectivity='knn',
              gcn_layers=2, gcn_hidden_dim=64, **kwargs):
        """
        训练污染预测模型
        
        参数:
            rs_data_paths (list): 遥感图像路径列表
            monitoring_data_path (str): 监测点位数据CSV文件路径
            infrastructure_data (dict): 基础设施数据，包含'pipelines'和'tanks'路径
            risk_assessment_path (str, optional): 风险评估数据路径
            study_area_path (str, optional): 研究区域边界路径
            target_pollutant (str): 目标污染物名称
            temporal_sequence (bool): 是否使用时序序列数据(完整版)
            time_steps (int, optional): 时序步长(完整版)
            graph_connectivity (str): 图连接方式，'knn'、'radius'或'delaunay'(完整版)
            gcn_layers (int): GCN层数(完整版)
            gcn_hidden_dim (int): GCN隐藏层维度(完整版)
            **kwargs: 其他扩展参数
            
        返回:
            self: 返回实例本身
        """
        print(f"开始训练模型 - 使用{self.version}版本")
        self.target_pollutant = target_pollutant
        
        # 保存元数据信息
        self.metadata = {
            'version': self.version,
            'use_cnn': self.use_cnn,
            'use_gcn': self.use_gcn,
            'target_pollutant': target_pollutant,
            'temporal_sequence': temporal_sequence,
            'time_steps': time_steps,
            'graph_connectivity': graph_connectivity,
            'gcn_layers': gcn_layers,
            'gcn_hidden_dim': gcn_hidden_dim,
            **kwargs
        }
        
        # 1. 加载数据
        print("加载数据...")
        rs_data = self.data_loader.load_remote_sensing_data(rs_data_paths)
        monitoring_data = self.data_loader.load_monitoring_data(
            monitoring_data_path, 
            target_pollutant=target_pollutant
        )
        infra_data = self.data_loader.load_infrastructure_data(infrastructure_data)
        
        if risk_assessment_path:
            risk_data = self.data_loader.load_risk_assessment(risk_assessment_path)
        else:
            risk_data = None
            
        if study_area_path:
            study_area = self.data_loader.load_study_area(study_area_path)
        else:
            study_area = None
        
        # 2. 特征提取
        print("提取特征...")
        features = self.feature_extractor.extract_features(
            rs_data=rs_data,
            monitoring_data=monitoring_data,
            infrastructure_data=infra_data,
            risk_data=risk_data,
            study_area=study_area,
            **kwargs
        )
        
        # 3. 时序处理（完整版）
        if self.version == 'full' and temporal_sequence:
            print("处理时序数据...")
            features = self.temporal_processor.process(
                features,
                monitoring_data,
                time_steps=time_steps,
                **kwargs
            )
        
        # 4. 构建图（完整版GCN）
        if self.use_gcn:
            print("构建图结构...")
            graph = self.graph_builder.build_graph(
                features,
                connectivity=graph_connectivity,
                **kwargs
            )
        else:
            graph = None
            
        # 5. 创建并训练模型
        print("训练模型...")
        X, y = features['X'], features['y']
        
        # 选择模型
        if self.version == 'simplified':
            # 简化版使用集成模型
            random_forest = RandomForestModel(**kwargs)
            xgboost = XGBoostModel(**kwargs)
            kriging = KrigingModel(**kwargs)
            
            # 在简化版下，如果use_cnn=True，则加入CNN模型
            models = [random_forest, xgboost, kriging]
            if self.use_cnn:
                cnn = CNNModel(**kwargs)
                models.append(cnn)
                
            self.model = EnsembleModel(models=models, **kwargs)
            
        else:  # 完整版
            if self.use_gcn:
                # 使用GCN模型
                self.model = GCNModel(
                    input_dim=X.shape[1], 
                    hidden_dim=gcn_hidden_dim,
                    layers=gcn_layers,
                    **kwargs
                )
            else:
                # 使用CNN模型
                self.model = CNNModel(**kwargs)
        
        # 训练模型
        self.model.fit(X, y, graph=graph if self.use_gcn else None, **kwargs)
        self.trained = True
        
        # 评估训练性能
        self.evaluation_results = self.evaluator.evaluate(self.model, X, y, **kwargs)
        print(f"模型训练完成，训练集R²: {self.evaluation_results['r2']:.4f}")
        
        return self
        
    def predict(self, output_grid_resolution=10.0, return_type='geojson', 
                temporal_prediction=False, prediction_horizon=1, **kwargs):
        """
        预测污染物分布
        
        参数:
            output_grid_resolution (float): 输出网格分辨率(米)
            return_type (str): 返回类型，'geojson', 'dataframe'或'raster'
            temporal_prediction (bool): 是否进行时序预测(完整版)
            prediction_horizon (int): 预测时间步长(完整版)
            **kwargs: 其他扩展参数
            
        返回:
            dict/GeoDataFrame/array: 根据return_type返回相应格式的预测结果
        """
        if not self.trained and not self.model:
            raise ValueError("模型尚未训练或加载，请先调用train()方法或加载预训练模型")
            
        print(f"预测污染物分布 - 网格分辨率: {output_grid_resolution}米")
        
        # 创建预测网格
        grid = self.io_handler.create_prediction_grid(
            self.data_loader.get_study_area_bounds(),
            resolution=output_grid_resolution
        )
        
        # 提取网格特征
        grid_features = self.feature_extractor.extract_grid_features(
            grid=grid,
            rs_data=self.data_loader.get_remote_sensing_data(),
            infrastructure_data=self.data_loader.get_infrastructure_data(),
            risk_data=self.data_loader.get_risk_assessment(),
            **kwargs
        )
        
        # 时序预测处理（完整版）
        if self.version == 'full' and temporal_prediction:
            grid_features = self.temporal_processor.prepare_prediction(
                grid_features, 
                prediction_horizon=prediction_horizon
            )
            
        # 构建预测图（如果使用GCN）
        if self.use_gcn:
            graph = self.graph_builder.build_prediction_graph(
                grid_features,
                **kwargs
            )
        else:
            graph = None
        
        # 执行预测
        predictions = self.model.predict(
            grid_features, 
            graph=graph if self.use_gcn else None,
            **kwargs
        )
        
        # 计算不确定性（如果支持）
        if hasattr(self.model, 'predict_uncertainty'):
            uncertainties = self.model.predict_uncertainty(
                grid_features,
                graph=graph if self.use_gcn else None,
                **kwargs
            )
        else:
            uncertainties = np.zeros_like(predictions)
        
        # 使用专题1风险评级标准对预测结果进行分级
        risk_levels = self._classify_risk_levels(predictions)
        
        # 准备返回结果
        if self.version == 'full' and temporal_prediction:
            # 时序预测结果
            self.prediction_results = self._format_temporal_predictions(
                grid, predictions, uncertainties, risk_levels, 
                prediction_horizon, return_type
            )
        else:
            # 常规预测结果
            self.prediction_results = self._format_predictions(
                grid, predictions, uncertainties, risk_levels, return_type
            )
        
        print(f"预测完成，返回{return_type}格式结果")
        return self.prediction_results
    
    def _classify_risk_levels(self, predictions):
        """根据专题1的风险分级标准将预测值分级"""
        # 这里使用简单的分级规则，实际应用中应根据专题1提供的标准
        risk_levels = np.empty(len(predictions), dtype=object)
        for i, pred in enumerate(predictions):
            if pred < 0.5:
                risk_levels[i] = "Low"
            elif pred < 2.0:
                risk_levels[i] = "Medium"
            else:
                risk_levels[i] = "High"
        return risk_levels
    
    def _format_predictions(self, grid, predictions, uncertainties, risk_levels, return_type):
        """格式化预测结果"""
        if return_type == 'geojson':
            # 创建GeoJSON格式输出
            features = []
            for i in range(len(grid)):
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(grid.iloc[i]['x']), float(grid.iloc[i]['y'])]
                    },
                    "properties": {
                        "predicted_value": float(predictions[i]),
                        "uncertainty": float(uncertainties[i]),
                        "risk_level": risk_levels[i]
                    }
                }
                features.append(feature)
                
            return {"type": "FeatureCollection", "features": features}
            
        elif return_type == 'dataframe':
            # 创建DataFrame格式输出
            result_df = grid.copy()
            result_df['predicted_value'] = predictions
            result_df['uncertainty'] = uncertainties
            result_df['risk_level'] = risk_levels
            return result_df
            
        elif return_type == 'raster':
            # 创建栅格格式输出
            # 需要将预测数据组织成规则网格
            grid_shape, raster_data, transform = self.io_handler.points_to_raster(
                grid['x'].values, grid['y'].values, 
                predictions, resolution=grid.attrs.get('resolution', 10.0)
            )
            
            # 返回栅格数据和元数据
            return {
                'data': raster_data,
                'transform': transform,
                'grid_shape': grid_shape,
                'crs': grid.attrs.get('crs', 'EPSG:4326')
            }
        
        else:
            raise ValueError(f"不支持的返回类型: {return_type}")
    
    def _format_temporal_predictions(self, grid, predictions, uncertainties, 
                                    risk_levels, prediction_horizon, return_type):
        """格式化时序预测结果"""
        # 此处假设predictions是三维数组：[样本数, 时间步数, 特征数]
        # 创建返回结果集合
        results = []
        
        # 获取时间戳列表
        timestamps = self.temporal_processor.get_prediction_timestamps(prediction_horizon)
        
        # 对每个时间步进行格式化
        for t in range(prediction_horizon):
            # 获取当前时间步的预测、不确定性和风险等级
            current_preds = predictions[:, t] if len(predictions.shape) > 1 else predictions
            current_uncs = uncertainties[:, t] if len(uncertainties.shape) > 1 else uncertainties
            current_risks = risk_levels
            
            # 格式化当前时间步的结果
            current_result = self._format_predictions(
                grid, current_preds, current_uncs, current_risks, return_type
            )
            
            # 添加时间戳信息
            if return_type == 'geojson':
                for feature in current_result['features']:
                    feature['properties']['timestamp'] = timestamps[t]
                    
            elif return_type == 'dataframe':
                current_result['timestamp'] = timestamps[t]
                
            elif return_type == 'raster':
                current_result['timestamp'] = timestamps[t]
            
            results.append(current_result)
        
        # 如果只有一个时间步，直接返回结果而不是列表
        if len(results) == 1:
            return results[0]
        return results
    
    def save_model(self, model_path):
        """
        保存训练好的模型
        
        参数:
            model_path (str): 保存模型的路径
        """
        if not self.trained or not self.model:
            raise ValueError("模型尚未训练，无法保存")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 创建要保存的数据字典
        save_data = {
            'model': self.model,
            'metadata': self.metadata,
            'feature_extractor': self.feature_extractor,
            'version': self.version,
            'use_cnn': self.use_cnn,
            'use_gcn': self.use_gcn,
            'target_pollutant': self.target_pollutant
        }
        
        # 如果是完整版，还需保存图构建器和时序处理器
        if self.version == 'full':
            save_data['graph_builder'] = self.graph_builder
            save_data['temporal_processor'] = self.temporal_processor
        
        # 保存模型和元数据
        with open(model_path, 'wb') as f:
            pickle.dump(save_data, f)
            
        print(f"模型已保存至 {model_path}")
        
    def load_model(self, model_path):
        """
        加载预训练模型
        
        参数:
            model_path (str): 模型路径
            
        返回:
            self: 返回实例本身
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
        # 加载模型和元数据
        with open(model_path, 'rb') as f:
            save_data = pickle.load(f)
        
        # 恢复模型和元数据
        self.model = save_data['model']
        self.metadata = save_data['metadata']
        self.feature_extractor = save_data['feature_extractor']
        self.version = save_data['version']
        self.use_cnn = save_data['use_cnn']
        self.use_gcn = save_data['use_gcn']
        self.target_pollutant = save_data['target_pollutant']
        
        # 如果是完整版，还需恢复图构建器和时序处理器
        if self.version == 'full':
            self.graph_builder = save_data.get('graph_builder', GraphBuilder())
            self.temporal_processor = save_data.get('temporal_processor', TemporalProcessor())
        
        self.trained = True
        print(f"已加载模型 (版本: {self.version}, 目标污染物: {self.target_pollutant})")
        
        return self
        
    def visualize(self, output_folder, visualize_uncertainty=True,
                 visualize_feature_importance=True, visualize_temporal=False, **kwargs):
        """
        生成可视化结果
        
        参数:
            output_folder (str): 输出文件夹路径
            visualize_uncertainty (bool): 是否可视化不确定性
            visualize_feature_importance (bool): 是否可视化特征重要性
            visualize_temporal (bool): 是否可视化时序变化(完整版)
            **kwargs: 其他扩展参数
            
        返回:
            dict: 生成的图像文件路径字典
        """
        if not self.prediction_results:
            raise ValueError("尚未生成预测结果，请先调用predict()方法")
            
        os.makedirs(output_folder, exist_ok=True)
        
        # 将预测结果转换为栅格格式（如果不是）
        if isinstance(self.prediction_results, dict) and 'type' in self.prediction_results:
            # GeoJSON格式
            predictions_df = self._geojson_to_dataframe(self.prediction_results)
        elif isinstance(self.prediction_results, pd.DataFrame):
            # DataFrame格式
            predictions_df = self.prediction_results
        else:
            # 栅格格式
            predictions_df = self._raster_to_dataframe(self.prediction_results)
        
        # 生成污染物分布图
        print("生成污染物分布可视化...")
        viz_outputs = {}
        viz_outputs['pollution_map'] = self.visualizer.plot_pollution_map(
            predictions_df, 
            output_path=os.path.join(output_folder, f"{self.target_pollutant}_distribution.png"),
            title=f"{self.target_pollutant} 分布图",
            **kwargs
        )
        
        # 生成不确定性图
        if visualize_uncertainty and 'uncertainty' in predictions_df.columns:
            print("生成不确定性可视化...")
            viz_outputs['uncertainty_map'] = self.visualizer.plot_uncertainty_map(
                predictions_df,
                output_path=os.path.join(output_folder, "uncertainty_map.png"),
                title="预测不确定性分布图",
                **kwargs
            )
        
        # 生成特征重要性图
        if visualize_feature_importance and hasattr(self.model, 'feature_importance_'):
            print("生成特征重要性可视化...")
            viz_outputs['feature_importance'] = self.visualizer.plot_feature_importance(
                self.model.feature_importance_,
                self.feature_extractor.get_feature_names(),
                output_path=os.path.join(output_folder, "feature_importance.png"),
                title="特征重要性",
                **kwargs
            )
        
        # 生成时序变化图（完整版）
        if visualize_temporal and self.version == 'full' and isinstance(self.prediction_results, list):
            print("生成时序变化可视化...")
            temporal_dfs = []
            for i, result in enumerate(self.prediction_results):
                if isinstance(result, dict) and 'type' in result:
                    df = self._geojson_to_dataframe(result)
                elif isinstance(result, pd.DataFrame):
                    df = result
                else:
                    df = self._raster_to_dataframe(result)
                df['time_step'] = i
                temporal_dfs.append(df)
            
            temporal_df = pd.concat(temporal_dfs)
            viz_outputs['temporal_change'] = self.visualizer.plot_temporal_change(
                temporal_df,
                output_path=os.path.join(output_folder, "temporal_change.png"),
                title=f"{self.target_pollutant} 时序变化",
                **kwargs
            )
        
        print(f"可视化结果已保存至 {output_folder}")
        return viz_outputs
        
    def _geojson_to_dataframe(self, geojson):
        """将GeoJSON格式转换为DataFrame"""
        data = []
        for feature in geojson['features']:
            x, y = feature['geometry']['coordinates']
            props = feature['properties']
            data.append({
                'x': x,
                'y': y,
                'predicted_value': props['predicted_value'],
                'uncertainty': props.get('uncertainty', 0),
                'risk_level': props.get('risk_level', 'Unknown'),
                'timestamp': props.get('timestamp', None)
            })
        return pd.DataFrame(data)
    
    def _raster_to_dataframe(self, raster):
        """将栅格格式转换为DataFrame"""
        # 栅格数据转换回点数据
        x, y, values = self.io_handler.raster_to_points(
            raster['data'], 
            raster['transform']
        )
        
        # 创建DataFrame
        data = {
            'x': x,
            'y': y,
            'predicted_value': values
        }
        
        # 添加额外信息（如果有）
        if 'uncertainty' in raster:
            _, _, uncertainty = self.io_handler.raster_to_points(
                raster['uncertainty'], 
                raster['transform']
            )
            data['uncertainty'] = uncertainty
            
        if 'risk_level' in raster:
            _, _, risk_levels = self.io_handler.raster_to_points(
                raster['risk_level'], 
                raster['transform']
            )
            data['risk_level'] = risk_levels
            
        if 'timestamp' in raster:
            data['timestamp'] = raster['timestamp']
            
        return pd.DataFrame(data)
        
    def evaluate(self, temporal_evaluation=False):
        """
        评估模型性能
        
        参数:
            temporal_evaluation (bool): 是否评估时序预测性能(完整版)
            
        返回:
            dict: 包含'rmse', 'r2'等性能指标的字典
        """
        if not self.trained or not self.model:
            raise ValueError("模型尚未训练，无法评估")
            
        if self.evaluation_results is None:
            # 如果尚未评估，使用最新的训练数据进行评估
            features = self.feature_extractor.get_latest_features()
            X, y = features['X'], features['y']
            self.evaluation_results = self.evaluator.evaluate(self.model, X, y)
        
        # 时序评估（完整版）
        if temporal_evaluation and self.version == 'full':
            temporal_features = self.temporal_processor.get_latest_features()
            if temporal_features:
                temporal_X, temporal_y = temporal_features['X'], temporal_features['y']
                temporal_results = self.evaluator.evaluate_temporal(
                    self.model, temporal_X, temporal_y
                )
                # 合并结果
                self.evaluation_results.update(temporal_results)
        
        return self.evaluation_results