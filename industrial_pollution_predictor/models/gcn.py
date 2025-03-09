# industrial_pollution_predictor/models/gcn.py
"""
图卷积网络模型，用于考虑空间关系的污染物浓度预测。
"""

import numpy as np
from typing import Dict, Any, Optional

# 尝试导入PyTorch和PyTorch Geometric
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    
    try:
        import torch_geometric
        from torch_geometric.nn import GCNConv, SAGEConv
        TORCH_GEOMETRIC_AVAILABLE = True
    except ImportError:
        TORCH_GEOMETRIC_AVAILABLE = False
        print("警告: PyTorch Geometric未安装，将使用自定义GCN实现")
        
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_GEOMETRIC_AVAILABLE = False
    print("警告: PyTorch未安装，GCN模型将使用备选方法")


class GCNModel:
    """
    图卷积网络模型，用于考虑空间关系的污染物浓度预测
    """
    
    def __init__(self, input_dim=None, hidden_dim=64, layers=2, dropout_rate=0.3, 
                learning_rate=0.001, epochs=200, batch_size=32, **kwargs):
        """
        初始化GCN模型
        
        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            layers: GCN层数
            dropout_rate: Dropout比率
            learning_rate: 学习率
            epochs: 训练轮数
            batch_size: 批次大小
            **kwargs: 其他参数
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.trained = False
        self.feature_importance_ = None
        
        # 检查PyTorch是否可用
        if not TORCH_AVAILABLE:
            print("警告: PyTorch不可用，GCN模型将使用sklearn的随机森林作为备选")
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                random_state=kwargs.get('random_state', 42),
                n_jobs=kwargs.get('n_jobs', -1)
            )
    
    def fit(self, X, y, graph=None, **kwargs):
        """
        训练GCN模型
        
        参数:
            X: 特征矩阵
            y: 目标变量
            graph: 图结构
            **kwargs: 其他参数
            
        返回:
            self: 返回实例本身
        """
        if TORCH_AVAILABLE:
            self._fit_torch(X, y, graph, **kwargs)
        else:
            # 使用sklearn的随机森林作为备选
            self.model.fit(X, y)
            self.feature_importance_ = self.model.feature_importances_
        
        self.trained = True
        return self
    
    def _fit_torch(self, X, y, graph, **kwargs):
        """使用PyTorch训练GCN模型"""
        if graph is None:
            raise ValueError("GCN模型需要提供图结构")
            
        # 更新输入维度
        if self.input_dim is None:
            self.input_dim = X.shape[1]
            
        # 创建或更新模型
        if TORCH_GEOMETRIC_AVAILABLE:
            self._create_pyg_model()
        else:
            self._create_custom_gcn_model()
            
        # 转换数据为PyTorch格式
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        # 转换图结构
        if TORCH_GEOMETRIC_AVAILABLE:
            edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
            edge_weight = torch.tensor(graph['edge_weight'], dtype=torch.float32) if 'edge_weight' in graph else None
            data = torch_geometric.data.Data(x=X_tensor, y=y_tensor, edge_index=edge_index)
            if edge_weight is not None:
                data.edge_weight = edge_weight
                
            # 使用PyTorch Geometric的DataLoader
            from torch_geometric.loader import DataLoader as PyGDataLoader
            loader = PyGDataLoader([data], batch_size=1)
        else:
            # 使用邻接矩阵
            adj_matrix = torch.tensor(graph['adj_matrix'], dtype=torch.float32)
            
            # 创建数据集
            dataset = TensorDataset(X_tensor, adj_matrix, y_tensor)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 定义优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # 训练模型
        best_val_loss = float('inf')
        patience = kwargs.get('patience', 20)
        patience_counter = 0
        best_model_state = None
        
        # 分割数据为训练集和验证集
        train_size = int(0.8 * len(X_tensor))
        val_size = len(X_tensor) - train_size
        indices = torch.randperm(len(X_tensor))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        for epoch in range(self.epochs):
            # 训练模式
            self.model.train()
            train_loss = 0
            
            if TORCH_GEOMETRIC_AVAILABLE:
                for batch in loader:
                    optimizer.zero_grad()
                    
                    # 选择训练数据
                    train_mask = torch.zeros(batch.x.size(0), dtype=torch.bool)
                    train_mask[train_indices] = True
                    batch.train_mask = train_mask
                    
                    # 前向传播
                    out = self.model(batch.x, batch.edge_index, 
                                   batch.edge_weight if hasattr(batch, 'edge_weight') else None)
                    loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
                    
                    # 反向传播
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
            else:
                for batch_X, batch_adj, batch_y in loader:
                    optimizer.zero_grad()
                    
                    # 选择训练数据
                    train_batch_X = batch_X[train_indices]
                    train_batch_adj = batch_adj[:, train_indices][train_indices, :]
                    train_batch_y = batch_y[train_indices]
                    
                    # 前向传播
                    out = self.model(train_batch_X, train_batch_adj)
                    loss = criterion(out, train_batch_y)
                    
                    # 反向传播
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
            
            # 验证模式
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                if TORCH_GEOMETRIC_AVAILABLE:
                    for batch in loader:
                        # 选择验证数据
                        val_mask = torch.zeros(batch.x.size(0), dtype=torch.bool)
                        val_mask[val_indices] = True
                        batch.val_mask = val_mask
                        
                        # 前向传播
                        out = self.model(batch.x, batch.edge_index, 
                                       batch.edge_weight if hasattr(batch, 'edge_weight') else None)
                        loss = criterion(out[batch.val_mask], batch.y[batch.val_mask])
                        val_loss += loss.item()
                else:
                    for batch_X, batch_adj, batch_y in loader:
                        # 选择验证数据
                        val_batch_X = batch_X[val_indices]
                        val_batch_adj = batch_adj[:, val_indices][val_indices, :]
                        val_batch_y = batch_y[val_indices]
                        
                        # 前向传播
                        out = self.model(val_batch_X, val_batch_adj)
                        loss = criterion(out, val_batch_y)
                        val_loss += loss.item()
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], '
                      f'Train Loss: {train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}')
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # 计算特征重要性
        self.feature_importance_ = self._compute_feature_importance(X, y, graph)
    
    def _create_pyg_model(self):
        """创建PyTorch Geometric GCN模型"""
        class PyGGCN(nn.Module):
            def __init__(self, input_dim, hidden_dim, layers=2, dropout_rate=0.3):
                super().__init__()
                self.dropout_rate = dropout_rate
                self.layers = nn.ModuleList()
                
                # 第一层
                self.layers.append(GCNConv(input_dim, hidden_dim))
                
                # 中间层
                for _ in range(layers - 2):
                    self.layers.append(GCNConv(hidden_dim, hidden_dim))
                
                # 输出层
                self.layers.append(GCNConv(hidden_dim, 1))
                
            def forward(self, x, edge_index, edge_weight=None):
                for i, layer in enumerate(self.layers[:-1]):
                    x = layer(x, edge_index, edge_weight)
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout_rate, training=self.training)
                
                # 输出层
                x = self.layers[-1](x, edge_index, edge_weight)
                return x
        
        self.model = PyGGCN(self.input_dim, self.hidden_dim, self.layers, self.dropout_rate)
    
    def _create_custom_gcn_model(self):
        """创建自定义GCN模型"""
        class CustomGCN(nn.Module):
            def __init__(self, input_dim, hidden_dim, layers=2, dropout_rate=0.3):
                super().__init__()
                self.dropout_rate = dropout_rate
                self.layers = nn.ModuleList()
                
                # 第一层
                self.layers.append(CustomGCNLayer(input_dim, hidden_dim))
                
                # 中间层
                for _ in range(layers - 2):
                    self.layers.append(CustomGCNLayer(hidden_dim, hidden_dim))
                
                # 输出层
                self.layers.append(CustomGCNLayer(hidden_dim, 1))
                
            def forward(self, x, adj):
                for i, layer in enumerate(self.layers[:-1]):
                    x = layer(x, adj)
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout_rate, training=self.training)
                
                # 输出层
                x = self.layers[-1](x, adj)
                return x
        
        class CustomGCNLayer(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.linear = nn.Linear(in_features, out_features)
                
            def forward(self, x, adj):
                support = self.linear(x)
                output = torch.matmul(adj, support)
                return output
        
        self.model = CustomGCN(self.input_dim, self.hidden_dim, self.layers, self.dropout_rate)
    
    def predict(self, X, graph=None, **kwargs):
        """
        预测目标变量
        
        参数:
            X: 特征矩阵
            graph: 图结构
            **kwargs: 其他参数
            
        返回:
            预测结果
        """
        if not self.trained or self.model is None:
            raise ValueError("模型尚未训练")
            
        if not TORCH_AVAILABLE:
            # 使用sklearn的随机森林
            return self.model.predict(X)
            
        # 转换数据为PyTorch格式
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # 预测模式
        self.model.eval()
        
        with torch.no_grad():
            if TORCH_GEOMETRIC_AVAILABLE:
                if graph is None:
                    raise ValueError("GCN模型需要提供图结构")
                
                # 转换图结构
                edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
                edge_weight = torch.tensor(graph['edge_weight'], dtype=torch.float32) if 'edge_weight' in graph else None
                
                # 前向传播
                out = self.model(X_tensor, edge_index, edge_weight)
            else:
                if graph is None:
                    raise ValueError("GCN模型需要提供图结构")
                
                # 使用邻接矩阵
                adj_matrix = torch.tensor(graph['adj_matrix'], dtype=torch.float32)
                
                # 前向传播
                out = self.model(X_tensor, adj_matrix)
        
        return out.squeeze().numpy()
    
    def predict_uncertainty(self, X, graph=None, **kwargs):
        """
        预测不确定性
        
        参数:
            X: 特征矩阵
            graph: 图结构
            **kwargs: 其他参数
            
        返回:
            不确定性估计
        """
        if not self.trained or self.model is None:
            raise ValueError("模型尚未训练")
            
        if not TORCH_AVAILABLE:
            # 使用sklearn的随机森林的树间方差
            if hasattr(self.model, 'estimators_'):
                # 使用各个树的预测方差
                predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
                return np.std(predictions, axis=0)
            else:
                # 简单启发式
                predictions = self.model.predict(X)
                return 0.1 * np.abs(predictions)
            
        # 对于GCN，使用MC-Dropout计算不确定性
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # 启用Dropout并进行多次预测
        self.model.train()  # 设置为训练模式以启用Dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(10):  # 进行10次预测
                if TORCH_GEOMETRIC_AVAILABLE:
                    # 转换图结构
                    edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
                    edge_weight = torch.tensor(graph['edge_weight'], dtype=torch.float32) if 'edge_weight' in graph else None
                    
                    # 前向传播
                    out = self.model(X_tensor, edge_index, edge_weight)
                else:
                    # 使用邻接矩阵
                    adj_matrix = torch.tensor(graph['adj_matrix'], dtype=torch.float32)
                    
                    # 前向传播
                    out = self.model(X_tensor, adj_matrix)
                
                predictions.append(out.squeeze().numpy())
        
        # 计算预测的标准差作为不确定性估计
        return np.std(predictions, axis=0)
    
    def _compute_feature_importance(self, X, y, graph):
        """
        计算特征重要性
        
        参数:
            X: 特征矩阵
            y: 目标变量
            graph: 图结构
            
        返回:
            特征重要性
        """
        # 对于GCN模型，特征重要性计算比较复杂
        # 这里使用简化的置换特征重要性
        from sklearn.metrics import mean_squared_error
        
        # 基线评分
        baseline_predictions = self.predict(X, graph)
        baseline_score = mean_squared_error(y, baseline_predictions)
        
        # 计算每个特征的重要性
        importances = np.zeros(X.shape[1])
        
        for i in range(X.shape[1]):
            # 创建一个X的副本
            X_permuted = X.copy()
            
            # 置换特征i
            perm_idx = np.random.permutation(X.shape[0])
            X_permuted[:, i] = X[perm_idx, i]
            
            # 计算新评分
            permuted_predictions = self.predict(X_permuted, graph)
            permuted_score = mean_squared_error(y, permuted_predictions)
            
            # 重要性 = 基线评分 - 置换后评分
            importances[i] = permuted_score - baseline_score
        
        # 归一化
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        
        return importances