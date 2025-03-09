# industrial_pollution_predictor/models/__init__.py
"""
模型模块，包含各种预测模型。
"""

from .ensemble import EnsembleModel
from .random_forest import RandomForestModel
from .xgboost_model import XGBoostModel
from .kriging import KrigingModel
from .cnn_model import CNNModel
from .gcn import GCNModel

__all__ = [
    'EnsembleModel',
    'RandomForestModel',
    'XGBoostModel',
    'KrigingModel',
    'CNNModel',
    'GCNModel',
]