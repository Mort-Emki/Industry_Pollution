# industrial_pollution_predictor/feature_extractor/cnn.py
"""
CNN特征提取器，用于从遥感图像中提取深度特征。
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from typing import Dict, Any, List, Union, Optional

# 尝试导入深度学习库
try:
    import tensorflow as tf
    from tensorflow.keras.applications import VGG16, ResNet50
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("警告: TensorFlow未安装，CNN特征提取将不可用")

try:
    import torch
    import torchvision.models as models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch未安装，将尝试使用TensorFlow作为备选")


class CNNFeatureExtractor:
    """CNN特征提取器，用于从遥感图像中提取深度特征"""
    
    def __init__(self, model_type='resnet50', use_pretrained=True, framework='auto'):
        """
        初始化CNN特征提取器
        
        参数:
            model_type (str): 模型类型，'resnet50'或'vgg16'
            use_pretrained (bool): 是否使用预训练模型
            framework (str): 深度学习框架，'tensorflow', 'pytorch'或'auto'
        """
        self.model_type = model_type
        self.use_pretrained = use_pretrained
        
        # 自动选择框架或使用指定框架
        if framework == 'auto':
            if TORCH_AVAILABLE:
                self.framework = 'pytorch'
            elif TF_AVAILABLE:
                self.framework = 'tensorflow'
            else:
                raise ImportError("未检测到可用的深度学习框架，请安装PyTorch或TensorFlow")
        else:
            self.framework = framework
            if framework == 'pytorch' and not TORCH_AVAILABLE:
                raise ImportError("指定使用PyTorch，但未安装")
            elif framework == 'tensorflow' and not TF_AVAILABLE:
                raise ImportError("指定使用TensorFlow，但未安装")
        
        # 初始化模型
        self.model = self._init_model()
        self.feature_size = self._get_feature_size()
        
    def _init_model(self):
        """初始化CNN模型"""
        if self.framework == 'pytorch':
            if self.model_type == 'resnet50':
                model = models.resnet50(pretrained=self.use_pretrained)
                # 移除最后的全连接层
                model = torch.nn.Sequential(*(list(model.children())[:-1]))
            elif self.model_type == 'vgg16':
                model = models.vgg16(pretrained=self.use_pretrained)
                # 移除最后的全连接层
                model = torch.nn.Sequential(*(list(model.children())[:-1]))
            else:
                raise ValueError(f"不支持的模型类型: {self.model_type}")
                
            # 设置为评估模式
            model.eval()
            return model
            
        elif self.framework == 'tensorflow':
            if self.model_type == 'resnet50':
                base_model = ResNet50(weights='imagenet' if self.use_pretrained else None, 
                                     include_top=False, pooling='avg')
            elif self.model_type == 'vgg16':
                base_model = VGG16(weights='imagenet' if self.use_pretrained else None, 
                                  include_top=False, pooling='avg')
            else:
                raise ValueError(f"不支持的模型类型: {self.model_type}")
                
            # 设置为非训练模式
            base_model.trainable = False
            return base_model
            
    def _get_feature_size(self):
        """获取CNN特征大小"""
        if self.framework == 'pytorch':
            if self.model_type == 'resnet50':
                return 2048
            elif self.model_type == 'vgg16':
                return 512
                
        elif self.framework == 'tensorflow':
            if self.model_type == 'resnet50':
                return 2048
            elif self.model_type == 'vgg16':
                return 512
                
        return 0
    
    def extract(self, rs_data, monitoring_data, patch_size=32, **kwargs):
        """
        从遥感图像中提取CNN特征
        
        参数:
            rs_data: 遥感数据
            monitoring_data: 监测点位数据
            patch_size: 图像块大小
            **kwargs: 其他参数
            
        返回:
            包含CNN特征的字典
        """
        if not rs_data:
            raise ValueError("遥感数据不能为空")
            
        # 提取监测点位周围的图像块
        patches = self._extract_patches(rs_data, monitoring_data, patch_size)
        
        # 使用CNN提取特征
        features = self._extract_cnn_features(patches)
        
        # 返回结果
        return {
            'X': features,
            'feature_names': [f'cnn_{i}' for i in range(features.shape[1])]
        }
    
    def extract_for_grid(self, rs_data, grid, patch_size=32, **kwargs):
        """
        为预测网格提取CNN特征
        
        参数:
            rs_data: 遥感数据
            grid: 预测网格
            patch_size: 图像块大小
            **kwargs: 其他参数
            
        返回:
            包含CNN特征的字典
        """
        if not rs_data:
            raise ValueError("遥感数据不能为空")
            
        # 提取网格点位周围的图像块
        patches = self._extract_patches(rs_data, grid, patch_size)
        
        # 使用CNN提取特征
        features = self._extract_cnn_features(patches)
        
        # 返回结果
        return {'X': features}
    
    def _extract_patches(self, rs_data, points_gdf, patch_size):
        """从遥感图像中提取点位周围的图像块"""
        # 选择第一个遥感数据作为基础
        key = next(iter(rs_data))
        rs_item = rs_data[key]
        
        # 获取遥感数据
        raster_data = rs_item['data']
        transform = rs_item['transform']
        
        # 提取点坐标
        x_coords = points_gdf.geometry.x
        y_coords = points_gdf.geometry.y
        
        # 计算点对应的栅格行列号
        rows, cols = rowcol(transform, x_coords, y_coords)
        
        # 创建掩码，确保行列号在栅格范围内
        num_bands, height, width = raster_data.shape
        valid_mask = (rows >= patch_size//2) & (rows < height - patch_size//2) & \
                     (cols >= patch_size//2) & (cols < width - patch_size//2)
        
        # 初始化图像块数组
        n_valid = np.sum(valid_mask)
        patches = np.zeros((n_valid, patch_size, patch_size, num_bands), dtype=np.float32)
        
        # 提取图像块
        valid_idx = 0
        for i in range(len(points_gdf)):
            if valid_mask[i]:
                row, col = rows[i], cols[i]
                # 提取各个波段的数据
                for band in range(num_bands):
                    patches[valid_idx, :, :, band] = raster_data[band, 
                                                              row - patch_size//2:row + patch_size//2,
                                                              col - patch_size//2:col + patch_size//2]
                valid_idx += 1
        
        # 标准化
        patches = patches / 255.0
        
        # 处理无效点：为剩余点生成空图像块
        if np.sum(~valid_mask) > 0:
            # 创建平均图像块
            if n_valid > 0:
                avg_patch = np.mean(patches, axis=0)
            else:
                avg_patch = np.zeros((patch_size, patch_size, num_bands), dtype=np.float32)
                
            # 为无效点分配平均图像块
            for i in range(len(points_gdf)):
                if not valid_mask[i]:
                    patches = np.vstack([patches, avg_patch.reshape(1, patch_size, patch_size, num_bands)])
        
        return patches
    
    def _extract_cnn_features(self, patches):
        """使用CNN模型提取特征"""
        if self.framework == 'pytorch':
            # 转换为PyTorch张量
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            
            # 转换为PyTorch张量并调整通道顺序(N,H,W,C) -> (N,C,H,W)
            x = torch.from_numpy(patches.transpose(0, 3, 1, 2)).float()
            
            # 创建数据加载器
            dataset = TensorDataset(x)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            # 提取特征
            features = []
            with torch.no_grad():
                for batch in dataloader:
                    batch_x = batch[0]
                    batch_features = self.model(batch_x)
                    # 展平特征
                    batch_features = batch_features.reshape(batch_features.size(0), -1)
                    features.append(batch_features.cpu().numpy())
            
            # 合并特征
            features = np.vstack(features)
            
        elif self.framework == 'tensorflow':
            # 使用TensorFlow进行预测
            import tensorflow as tf
            
            # 转换为TensorFlow张量
            x = tf.convert_to_tensor(patches, dtype=tf.float32)
            
            # 提取特征
            features = self.model.predict(x)
            
        return features