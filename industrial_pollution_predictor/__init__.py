# industrial_pollution_predictor/__init__.py
"""
工业集聚区多介质复合污染预测算法包

提供简化版和完整版两种模式的工业集聚区污染物预测功能。
简化版基于集成学习的空间预测算法，适用于有限数据情况。
完整版利用监测点位的完整时空序列，结合图神经网络(GCN)进行深度学习建模。
"""

from .predictor import PollutionPredictor

__version__ = '1.0.0'
__author__ = '专题2研究团队'