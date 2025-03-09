# industrial_pollution_predictor/models/cnn_model.py
"""
CNN模型，用于基于遥感图像特征预测污染物浓度。
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional

# 尝试导入深度学习库
try:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.models import Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("警告: TensorFlow未安装，CNN模型将使用备选方法")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    if not TF_AVAILABLE:
        print("警告: PyTorch也未安装，将使用sklearn的备选模型")
        from sklearn.ensemble import RandomForestRegressor


class CNNModel:
    """
    CNN模型，用于基于遥感图像特征预测污染物浓度
    """
    
    def __init__(self, hidden_layers=[128, 64], dropout_rate=0.3, learning_rate=0.001, 
                epochs=100, batch_size=32, framework='auto', **kwargs):
        """
        初始化CNN模型
        
        参数:
            hidden_layers: 隐藏层大小列表
            dropout_rate: Dropout比率
            learning_rate: 学习率
            epochs: 训练轮数
            batch_size: 批次大小
            framework: 深度学习框架，'tensorflow', 'pytorch'或'auto'
            **kwargs: 其他参数
        """
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = StandardScaler()
        self.trained = False
        self.feature_importance_ = None
        
        # 自动选择框架或使用指定框架
        if framework == 'auto':
            if TORCH_AVAILABLE:
                self.framework = 'pytorch'
            elif TF_AVAILABLE:
                self.framework = 'tensorflow'
            else:
                self.framework = 'sklearn'
        else:
            self.framework = framework
            if framework == 'pytorch' and not TORCH_AVAILABLE:
                raise ImportError("指定使用PyTorch，但未安装")
            elif framework == 'tensorflow' and not TF_AVAILABLE:
                raise ImportError("指定使用TensorFlow，但未安装")
        
    def fit(self, X, y, graph=None, **kwargs):
        """
        训练CNN模型
        
        参数:
            X: 特征矩阵
            y: 目标变量
            graph: 图结构（对CNN无效）
            **kwargs: 其他参数
            
        返回:
            self: 返回实例本身
        """
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 根据不同框架创建并训练模型
        if self.framework == 'tensorflow' and TF_AVAILABLE:
            self._fit_tensorflow(X_scaled, y, **kwargs)
        elif self.framework == 'pytorch' and TORCH_AVAILABLE:
            self._fit_pytorch(X_scaled, y, **kwargs)
        else:
            self._fit_sklearn(X_scaled, y, **kwargs)
        
        self.trained = True
        return self
    
    def _fit_tensorflow(self, X_scaled, y, **kwargs):
        """使用TensorFlow训练模型"""
        # 定义模型
        inputs = Input(shape=(X_scaled.shape[1],))
        x = inputs
        
        # 添加隐藏层
        for units in self.hidden_layers:
            x = Dense(units, activation='relu')(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
        
        # 输出层
        outputs = Dense(1)(x)
        
        # 创建模型
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # 编译模型
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        # 训练模型
        history = self.model.fit(
            X_scaled, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=kwargs.get('patience', 10),
                    restore_best_weights=True
                )
            ]
        )
        
        # 计算特征重要性（使用置换特征重要性）
        self.feature_importance_ = self._compute_permutation_importance(X_scaled, y)
    
    def _fit_pytorch(self, X_scaled, y, **kwargs):
        """使用PyTorch训练模型"""
        # 转换为PyTorch张量
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(X_tensor, y_tensor)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 定义模型
        class MLP(nn.Module):
            def __init__(self, input_dim, hidden_layers, dropout_rate):
                super().__init__()
                layers = []
                prev_dim = input_dim
                
                # 添加隐藏层
                for units in hidden_layers:
                    layers.append(nn.Linear(prev_dim, units))
                    layers.append(nn.ReLU())
                    if dropout_rate > 0:
                        layers.append(nn.Dropout(dropout_rate))
                    prev_dim = units
                
                # 输出层
                layers.append(nn.Linear(prev_dim, 1))
                
                self.model = nn.Sequential(*layers)
                
            def forward(self, x):
                return self.model(x)
        
        # 创建模型
        self.model = MLP(X_scaled.shape[1], self.hidden_layers, self.dropout_rate)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 训练模型
        best_val_loss = float('inf')
        patience = kwargs.get('patience', 10)
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.epochs):
            # 训练模式
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # 验证模式
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], '
                      f'Train Loss: {train_loss/len(train_loader):.4f}, '
                      f'Val Loss: {val_loss/len(val_loader):.4f}')
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # 计算特征重要性（使用置换特征重要性）
        self.feature_importance_ = self._compute_permutation_importance(X_scaled, y)
    
    def _fit_sklearn(self, X_scaled, y, **kwargs):
        """使用sklearn的RandomForestRegressor作为备选"""
        self.model = RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            random_state=kwargs.get('random_state', 42),
            n_jobs=kwargs.get('n_jobs', -1)
        )
        
        self.model.fit(X_scaled, y)
        
        # 获取特征重要性
        self.feature_importance_ = self.model.feature_importances_
    
    def predict(self, X, graph=None, **kwargs):
        """
        预测目标变量
        
        参数:
            X: 特征矩阵
            graph: 图结构（对CNN无效）
            **kwargs: 其他参数
            
        返回:
            预测结果
        """
        if not self.trained or self.model is None:
            raise ValueError("模型尚未训练")
            
        # 数据标准化
        X_scaled = self.scaler.transform(X)
        
        # 根据不同框架进行预测
        if self.framework == 'tensorflow' and TF_AVAILABLE:
            predictions = self.model.predict(X_scaled).flatten()
        elif self.framework == 'pytorch' and TORCH_AVAILABLE:
            # 转换为PyTorch张量
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            
            # 预测模式
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_tensor).numpy().flatten()
        else:
            # sklearn模型
            predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_uncertainty(self, X, graph=None, **kwargs):
        """
        预测不确定性
        
        参数:
            X: 特征矩阵
            graph: 图结构（对CNN无效）
            **kwargs: 其他参数
            
        返回:
            不确定性估计
        """
        if not self.trained or self.model is None:
            raise ValueError("模型尚未训练")
            
        # 数据标准化
        X_scaled = self.scaler.transform(X)
        
        # 对于深度学习模型，可以使用MC-Dropout或简单启发式方法
        if self.framework == 'tensorflow' and TF_AVAILABLE:
            # 如果模型有Dropout层，可以使用MC-Dropout
            if any(isinstance(layer, tf.keras.layers.Dropout) for layer in self.model.layers):
                # 启用训练模式，但不更新权重
                predictions = []
                for _ in range(10):  # 运行10次得到样本
                    pred = self.model(X_scaled, training=True).numpy().flatten()
                    predictions.append(pred)
                
                # 计算标准差
                uncertainty = np.std(predictions, axis=0)
            else:
                # 简单启发式：使用预测值的10%作为不确定性估计
                predictions = self.model.predict(X_scaled).flatten()
                uncertainty = 0.1 * np.abs(predictions)
                
        elif self.framework == 'pytorch' and TORCH_AVAILABLE:
            # 如果模型有Dropout层，可以使用MC-Dropout
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            
            # 检查模型是否有Dropout层
            has_dropout = any(isinstance(module, nn.Dropout) for module in self.model.modules())
            
            if has_dropout:
                # 启用训练模式，但不更新权重
                self.model.train()
                predictions = []
                with torch.no_grad():
                    for _ in range(10):  # 运行10次得到样本
                        pred = self.model(X_tensor).numpy().flatten()
                        predictions.append(pred)
                
                # 计算标准差
                uncertainty = np.std(predictions, axis=0)
            else:
                # 简单启发式：使用预测值的10%作为不确定性估计
                self.model.eval()
                with torch.no_grad():
                    predictions = self.model(X_tensor).numpy().flatten()
                uncertainty = 0.1 * np.abs(predictions)
                
        else:
            # sklearn模型（如随机森林）
            if hasattr(self.model, 'estimators_'):
                # 使用各个树的预测方差
                predictions = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])
                uncertainty = np.std(predictions, axis=0)
            else:
                # 简单启发式
                predictions = self.model.predict(X_scaled)
                uncertainty = 0.1 * np.abs(predictions)
        
        return uncertainty
    
    def _compute_permutation_importance(self, X, y, n_repeats=10):
        """
        计算置换特征重要性
        
        参数:
            X: 特征矩阵
            y: 目标变量
            n_repeats: 重复次数
            
        返回:
            特征重要性
        """
        from sklearn.metrics import mean_squared_error
        
        # 基线评分
        baseline_predictions = self.predict(X)
        baseline_score = mean_squared_error(y, baseline_predictions)
        
        # 计算每个特征的重要性
        importances = np.zeros(X.shape[1])
        
        for i in range(X.shape[1]):
            # 对特征i进行多次置换
            feature_importance = 0
            for _ in range(n_repeats):
                # 创建一个X的副本
                X_permuted = X.copy()
                
                # 置换特征i
                perm_idx = np.random.permutation(X.shape[0])
                X_permuted[:, i] = X[perm_idx, i]
                
                # 计算新评分
                permuted_predictions = self.predict(X_permuted)
                permuted_score = mean_squared_error(y, permuted_predictions)
                
                # 重要性 = 基线评分 - 置换后评分
                importance = permuted_score - baseline_score
                feature_importance += importance
            
            # 平均重要性
            importances[i] = feature_importance / n_repeats
        
        # 归一化
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        
        return importances