# industrial_pollution_predictor/utils/visualization.py
"""
可视化工具，用于生成污染物分布和模型结果的可视化。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable
import seaborn as sns
from typing import Dict, Any, List, Optional, Union


class Visualizer:
    """
    可视化工具，用于生成污染物分布和模型结果的可视化
    """
    
    def __init__(self):
        """初始化可视化工具"""
        pass
    
    def plot_pollution_map(self, predictions_df, output_path=None, title=None, 
                          colormap='viridis', **kwargs):
        """
        绘制污染物分布图
        
        参数:
            predictions_df: 预测结果DataFrame
            output_path: 输出图像路径
            title: 图像标题
            colormap: 颜色映射
            **kwargs: 其他参数
            
        返回:
            图像路径或Matplotlib图像对象
        """
        # 创建图像
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 设置散点图参数
        point_size = kwargs.get('point_size', 30)
        alpha = kwargs.get('alpha', 0.7)
        
        # 绘制散点图
        sc = ax.scatter(
            predictions_df['x'], 
            predictions_df['y'],
            c=predictions_df['predicted_value'],
            cmap=colormap,
            s=point_size,
            alpha=alpha,
            edgecolors='k',
            linewidths=0.5
        )
        
        # 添加颜色条
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Predicted Value')
        
        # 设置标题和轴标签
        if title:
            ax.set_title(title, fontsize=14)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        
        # 设置网格线
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return output_path
        else:
            return fig
    
    def plot_uncertainty_map(self, predictions_df, output_path=None, title=None, 
                           colormap='YlOrRd', **kwargs):
        """
        绘制不确定性分布图
        
        参数:
            predictions_df: 预测结果DataFrame
            output_path: 输出图像路径
            title: 图像标题
            colormap: 颜色映射
            **kwargs: 其他参数
            
        返回:
            图像路径或Matplotlib图像对象
        """
        if 'uncertainty' not in predictions_df.columns:
            raise ValueError("预测结果中缺少不确定性信息")
            
        # 创建图像
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 设置散点图参数
        point_size = kwargs.get('point_size', 30)
        alpha = kwargs.get('alpha', 0.7)
        
        # 绘制散点图
        sc = ax.scatter(
            predictions_df['x'], 
            predictions_df['y'],
            c=predictions_df['uncertainty'],
            cmap=colormap,
            s=point_size,
            alpha=alpha,
            edgecolors='k',
            linewidths=0.5
        )
        
        # 添加颜色条
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Uncertainty')
        
        # 设置标题和轴标签
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('Prediction Uncertainty', fontsize=14)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        
        # 设置网格线
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return output_path
        else:
            return fig
    
    def plot_feature_importance(self, importances, feature_names=None, output_path=None, 
                               title=None, **kwargs):
        """
        绘制特征重要性图
        
        参数:
            importances: 特征重要性数组
            feature_names: 特征名称列表
            output_path: 输出图像路径
            title: 图像标题
            **kwargs: 其他参数
            
        返回:
            图像路径或Matplotlib图像对象
        """
        # 检查特征名称
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importances))]
        else:
            # 确保特征名称和重要性数量一致
            if len(feature_names) != len(importances):
                feature_names = feature_names[:len(importances)]
                if len(feature_names) < len(importances):
                    feature_names.extend([f'Feature {i+len(feature_names)}' 
                                        for i in range(len(importances) - len(feature_names))])
        
        # 创建特征重要性DataFrame
        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # 按重要性排序
        feature_df = feature_df.sort_values('Importance', ascending=False)
        
        # 限制显示的特征数量
        top_n = kwargs.get('top_n', 20)
        if len(feature_df) > top_n:
            feature_df = feature_df.head(top_n)
        
        # 创建图像
        fig, ax = plt.subplots(figsize=(10, max(6, len(feature_df) * 0.3)))
        
        # 绘制水平条形图
        sns.barplot(
            x='Importance',
            y='Feature',
            data=feature_df,
            ax=ax,
            palette=kwargs.get('palette', 'viridis')
        )
        
        # 设置标题和轴标签
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('Feature Importance', fontsize=14)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return output_path
        else:
            return fig
    
    def plot_temporal_change(self, temporal_df, output_path=None, title=None, **kwargs):
        """
        绘制时序变化图
        
        参数:
            temporal_df: 时序预测结果DataFrame
            output_path: 输出图像路径
            title: 图像标题
            **kwargs: 其他参数
            
        返回:
            图像路径或Matplotlib图像对象
        """
        if 'time_step' not in temporal_df.columns:
            raise ValueError("时序预测结果中缺少时间步信息")
            
        # 创建图像
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 计算每个时间步的平均值、最小值和最大值
        summary = temporal_df.groupby('time_step')['predicted_value'].agg(['mean', 'min', 'max'])
        
        # 绘制线图和区间
        ax.plot(summary.index, summary['mean'], 'o-', lw=2, label='Mean')
        ax.fill_between(summary.index, summary['min'], summary['max'], alpha=0.3, label='Range')
        
        # 设置标题和轴标签
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('Temporal Change of Predicted Values', fontsize=14)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Predicted Value', fontsize=12)
        
        # 设置x轴刻度
        ax.set_xticks(summary.index)
        
        # 添加图例
        ax.legend()
        
        # 设置网格线
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 调整布局
        plt.tight_layout()
        
        # 如果有时间戳信息，创建第二张图
        if 'timestamp' in temporal_df.columns:
            # 创建热力图
            fig2, ax2 = plt.subplots(figsize=(12, 8))
            
            # 按位置和时间步分组计算均值
            pivot_df = temporal_df.pivot_table(
                index='time_step',
                columns=['y', 'x'],  # 使用坐标作为列索引
                values='predicted_value',
                aggfunc='mean'
            )
            
            # 绘制热力图
            sns.heatmap(pivot_df, ax=ax2, cmap='viridis', cbar_kws={'label': 'Predicted Value'})
            
            # 设置标题和轴标签
            ax2.set_title('Spatial-Temporal Distribution', fontsize=14)
            ax2.set_ylabel('Time Step', fontsize=12)
            ax2.set_xlabel('Grid Position', fontsize=12)
            
            # 如果提供了输出路径，保存两张图
            if output_path:
                # 保存第一张图
                plt.figure(fig.number)
                base_path = output_path.rsplit('.', 1)[0]
                ext = output_path.rsplit('.', 1)[1] if len(output_path.rsplit('.', 1)) > 1 else 'png'
                path1 = f"{base_path}_line.{ext}"
                plt.savefig(path1, dpi=300, bbox_inches='tight')
                
                # 保存第二张图
                plt.figure(fig2.number)
                path2 = f"{base_path}_heatmap.{ext}"
                plt.savefig(path2, dpi=300, bbox_inches='tight')
                
                return [path1, path2]
            else:
                return [fig, fig2]
        
        # 保存图像
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return output_path
        else:
            return fig