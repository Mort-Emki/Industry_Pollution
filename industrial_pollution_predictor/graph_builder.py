# industrial_pollution_predictor/graph_builder.py
"""
图构建模块，用于构建图卷积神经网络的图结构。
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, Any, Optional

# 尝试导入PyTorch Geometric
try:
    import torch
    import torch_geometric
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


class GraphBuilder:
    """
    图构建器，用于构建图卷积神经网络的图结构
    """
    
    def __init__(self):
        """初始化图构建器"""
        pass
    
    def build_graph(self, features, connectivity='knn', k=10, radius=None, **kwargs):
        """
        构建图结构
        
        参数:
            features: 特征字典，包含'X'和'monitoring_points'
            connectivity: 图连接方式，'knn'、'radius'或'delaunay'
            k: k近邻算法的k值
            radius: 半径算法的半径
            **kwargs: 其他参数
            
        返回:
            图结构字典
        """
        if 'monitoring_points' not in features:
            raise ValueError("特征字典中缺少'monitoring_points'键")
        
        # 获取监测点坐标
        points = features['monitoring_points']
        coords = np.vstack((points.geometry.x, points.geometry.y)).T
        
        # 构建图
        if connectivity == 'knn':
            graph = self._build_knn_graph(coords, k, **kwargs)
        elif connectivity == 'radius':
            if radius is None:
                # 估计合适的半径
                from sklearn.neighbors import NearestNeighbors
                nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
                distances, _ = nbrs.kneighbors(coords)
                radius = np.mean(distances[:, 1:]) * 2  # 使用平均最近邻距离的2倍作为半径
            
            graph = self._build_radius_graph(coords, radius, **kwargs)
        elif connectivity == 'delaunay':
            graph = self._build_delaunay_graph(coords, **kwargs)
        else:
            raise ValueError(f"不支持的图连接方式: {connectivity}")
        
        return graph
    
    def build_prediction_graph(self, grid_features, connectivity='knn', k=10, radius=None, **kwargs):
        """
        为预测网格构建图结构
        
        参数:
            grid_features: 网格特征字典，包含'X'和'grid'
            connectivity: 图连接方式，'knn'、'radius'或'delaunay'
            k: k近邻算法的k值
            radius: 半径算法的半径
            **kwargs: 其他参数
            
        返回:
            图结构字典
        """
        if 'grid' not in grid_features:
            raise ValueError("网格特征字典中缺少'grid'键")
        
        # 获取网格点坐标
        grid = grid_features['grid']
        coords = np.vstack((grid.geometry.x, grid.geometry.y)).T
        
        # 构建图
        if connectivity == 'knn':
            graph = self._build_knn_graph(coords, k, **kwargs)
        elif connectivity == 'radius':
            if radius is None:
                # 估计合适的半径
                from sklearn.neighbors import NearestNeighbors
                nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
                distances, _ = nbrs.kneighbors(coords)
                radius = np.mean(distances[:, 1:]) * 2  # 使用平均最近邻距离的2倍作为半径
            
            graph = self._build_radius_graph(coords, radius, **kwargs)
        elif connectivity == 'delaunay':
            graph = self._build_delaunay_graph(coords, **kwargs)
        else:
            raise ValueError(f"不支持的图连接方式: {connectivity}")
        
        return graph
    
    def _build_knn_graph(self, coords, k, **kwargs):
        """
        构建k近邻图
        
        参数:
            coords: 坐标矩阵
            k: 近邻数量
            **kwargs: 其他参数
            
        返回:
            图结构字典
        """
        from sklearn.neighbors import kneighbors_graph
        
        # 构建k近邻图，包括自环
        A = kneighbors_graph(coords, k, mode='connectivity', include_self=True)
        
        # 确保图是对称的（无向图）
        A = 0.5 * (A + A.T)
        A.data = np.ones_like(A.data)  # 将所有非零元素设为1
        
        # 计算边权重（可选）
        if kwargs.get('weighted', True):
            # 计算欧氏距离
            from scipy.spatial.distance import pdist, squareform
            dist_matrix = squareform(pdist(coords))
            
            # 根据距离计算权重
            weight_method = kwargs.get('edge_weight_method', 'inverse_distance')
            if weight_method == 'inverse_distance':
                # 反距离权重
                W = 1.0 / (dist_matrix + 1e-10)  # 避免除以零
            elif weight_method == 'gaussian':
                # 高斯权重
                sigma = kwargs.get('sigma', np.mean(dist_matrix) / 2)
                W = np.exp(-dist_matrix**2 / (2 * sigma**2))
            else:
                W = np.ones_like(dist_matrix)
                
            # 只保留k近邻的权重
            W = W * A.toarray()
        else:
            W = A.toarray()
        
        # 创建PyTorch Geometric格式的图（如果可用）
        if TORCH_GEOMETRIC_AVAILABLE:
            edge_index, edge_weight = self._dense_to_sparse(W)
            return {
                'edge_index': edge_index,
                'edge_weight': edge_weight,
                'adj_matrix': W  # 也返回邻接矩阵，以便自定义GCN使用
            }
        else:
            # 返回邻接矩阵
            return {'adj_matrix': W}
    
    def _build_radius_graph(self, coords, radius, **kwargs):
        """
        构建半径图
        
        参数:
            coords: 坐标矩阵
            radius: 搜索半径
            **kwargs: 其他参数
            
        返回:
            图结构字典
        """
        from sklearn.neighbors import radius_neighbors_graph
        
        # 构建半径图，包括自环
        A = radius_neighbors_graph(coords, radius, mode='connectivity', include_self=True)
        
        # 确保图是对称的（无向图）
        A = 0.5 * (A + A.T)
        A.data = np.ones_like(A.data)  # 将所有非零元素设为1
        
        # 计算边权重（可选）
        if kwargs.get('weighted', True):
            # 计算欧氏距离
            from scipy.spatial.distance import pdist, squareform
            dist_matrix = squareform(pdist(coords))
            
            # 根据距离计算权重
            weight_method = kwargs.get('edge_weight_method', 'inverse_distance')
            if weight_method == 'inverse_distance':
                # 反距离权重
                W = 1.0 / (dist_matrix + 1e-10)  # 避免除以零
            elif weight_method == 'gaussian':
                # 高斯权重
                sigma = kwargs.get('sigma', radius / 2)
                W = np.exp(-dist_matrix**2 / (2 * sigma**2))
            else:
                W = np.ones_like(dist_matrix)
                
            # 只保留半径内的权重
            W = W * A.toarray()
        else:
            W = A.toarray()
        
        # 创建PyTorch Geometric格式的图（如果可用）
        if TORCH_GEOMETRIC_AVAILABLE:
            edge_index, edge_weight = self._dense_to_sparse(W)
            return {
                'edge_index': edge_index,
                'edge_weight': edge_weight,
                'adj_matrix': W  # 也返回邻接矩阵，以便自定义GCN使用
            }
        else:
            # 返回邻接矩阵
            return {'adj_matrix': W}
    
    def _build_delaunay_graph(self, coords, **kwargs):
        """
        构建Delaunay三角剖分图
        
        参数:
            coords: 坐标矩阵
            **kwargs: 其他参数
            
        返回:
            图结构字典
        """
        from scipy.spatial import Delaunay
        
        # 构建Delaunay三角剖分
        tri = Delaunay(coords)
        
        # 创建邻接矩阵
        n = coords.shape[0]
        A = sp.lil_matrix((n, n), dtype=np.float32)
        
        # 填充邻接矩阵
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        A[simplex[i], simplex[j]] = 1
        
        # 确保图是对称的（无向图）
        A = sp.csr_matrix(A)
        A = 0.5 * (A + A.T)
        A.data = np.ones_like(A.data)  # 将所有非零元素设为1
        
        # 添加自环
        A = A + sp.eye(n, dtype=np.float32)
        
        # 计算边权重（可选）
        if kwargs.get('weighted', True):
            # 计算欧氏距离
            from scipy.spatial.distance import pdist, squareform
            dist_matrix = squareform(pdist(coords))
            
            # 根据距离计算权重
            weight_method = kwargs.get('edge_weight_method', 'inverse_distance')
            if weight_method == 'inverse_distance':
                # 反距离权重
                W = 1.0 / (dist_matrix + 1e-10)  # 避免除以零
            elif weight_method == 'gaussian':
                # 高斯权重
                # 使用平均边长作为sigma
                edge_lengths = []
                for simplex in tri.simplices:
                    for i in range(3):
                        for j in range(i+1, 3):
                            length = np.linalg.norm(coords[simplex[i]] - coords[simplex[j]])
                            edge_lengths.append(length)
                sigma = np.mean(edge_lengths)
                W = np.exp(-dist_matrix**2 / (2 * sigma**2))
            else:
                W = np.ones_like(dist_matrix)
                
            # 只保留Delaunay三角剖分中的边的权重
            W = W * A.toarray()
        else:
            W = A.toarray()
        
        # 创建PyTorch Geometric格式的图（如果可用）
        if TORCH_GEOMETRIC_AVAILABLE:
            edge_index, edge_weight = self._dense_to_sparse(W)
            return {
                'edge_index': edge_index,
                'edge_weight': edge_weight,
                'adj_matrix': W  # 也返回邻接矩阵，以便自定义GCN使用
            }
        else:
            # 返回邻接矩阵
            return {'adj_matrix': W}
    
    def _dense_to_sparse(self, adj):
        """
        将稠密邻接矩阵转换为PyTorch Geometric格式的边索引和边权重
        
        参数:
            adj: 邻接矩阵
            
        返回:
            边索引和边权重的元组
        """
        # 获取非零元素的坐标
        rows, cols = np.where(adj > 0)
        edge_index = np.stack([rows, cols], axis=0)
        
        # 获取边权重
        edge_weight = adj[rows, cols]
        
        return edge_index, edge_weight