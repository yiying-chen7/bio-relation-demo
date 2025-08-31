#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识图谱可视化模块
提供多种可视化方式展示生物医学知识图谱
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging
from collections import Counter
import pandas as pd
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)

class KGVisualizer:
    """知识图谱可视化器"""
    
    def __init__(self, kg=None):
        self.kg = kg
        self.color_map = {
            'microbe': '#FF6B6B',     # 红色 - 微生物
            'metabolite': '#4ECDC4',  # 蓝绿色 - 代谢物
            'unknown': '#95A5A6'      # 灰色 - 未知
        }
        self.predicate_colors = {
            'produces': '#E74C3C',
            'metabolizes': '#3498DB', 
            'affects': '#F39C12',
            'associates_with': '#9B59B6',
            'degrades': '#E67E22',
            'converts': '#1ABC9C',
            'utilizes': '#34495E'
        }
    
    def plot_network_graph(self, output_path: str = "network_graph.png", 
                          figsize: Tuple[int, int] = (12, 8),
                          max_nodes: int = 50) -> None:
        """绘制网络图"""
        if not self.kg or self.kg.graph.number_of_nodes() == 0:
            logger.warning("知识图谱为空，无法绘制网络图")
            return
        
        plt.figure(figsize=figsize)
        
        # 如果节点太多，只显示最重要的节点
        if self.kg.graph.number_of_nodes() > max_nodes:
            top_entities = self.kg.get_top_entities(limit=max_nodes, by="degree")
            node_names = [entity[0] for entity in top_entities]
            subgraph = self.kg.get_subgraph(node_names, expand_neighbors=False)
            title_suffix = f" (Top {max_nodes} entities by degree)"
        else:
            subgraph = self.kg.graph
            title_suffix = ""
        
        # 设置布局
        try:
            pos = nx.spring_layout(subgraph, k=3, iterations=50, seed=42)
        except:
            pos = nx.random_layout(subgraph, seed=42)
        
        # 准备节点颜色
        node_colors = []
        for node in subgraph.nodes():
            entity_type = subgraph.nodes[node].get('entity_type', 'unknown')
            node_colors.append(self.color_map.get(entity_type, self.color_map['unknown']))
        
        # 准备边颜色
        edge_colors = []
        edge_weights = []
        for u, v, data in subgraph.edges(data=True):
            predicate = data.get('predicate', 'unknown')
            edge_colors.append(self.predicate_colors.get(predicate, '#BDC3C7'))
            confidence = data.get('confidence', 0.5)
            edge_weights.append(confidence * 3)  # 置信度影响边宽度
        
        # 绘制图
        nx.draw_networkx_nodes(subgraph, pos, 
                             node_color=node_colors,
                             node_size=300,
                             alpha=0.8)
        
        nx.draw_networkx_edges(subgraph, pos,
                             edge_color=edge_colors,
                             width=edge_weights,
                             alpha=0.6,
                             arrows=True,
                             arrowsize=20,
                             arrowstyle='->')
        
        # 添加标签（只显示较短的名称）
        labels = {}
        for node in subgraph.nodes():
            if len(node) > 15:
                labels[node] = node[:12] + "..."
            else:
                labels[node] = node
        
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)
        
        # 图例
        legend_elements = []
        for entity_type, color in self.color_map.items():
            if entity_type != 'unknown':
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=color, markersize=10, 
                                                label=entity_type.capitalize()))
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title(f"Biomedical Knowledge Graph{title_suffix}")
        plt.axis('off')
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"网络图已保存到: {output_path}")
    
    def plot_entity_statistics(self, output_path: str = "entity_stats.png",
                             figsize: Tuple[int, int] = (10, 6)) -> None:
        """绘制实体统计图"""
        if not self.kg or not self.kg.statistics:
            logger.warning("无统计数据可绘制")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 实体类型分布饼图
        entity_types = self.kg.statistics.get('entity_types', {})
        if entity_types:
            colors = [self.color_map.get(t, '#95A5A6') for t in entity_types.keys()]
            ax1.pie(entity_types.values(), labels=entity_types.keys(), autopct='%1.1f%%',
                   colors=colors, startangle=90)
            ax1.set_title('Entity Type Distribution')
        
        # 关系类型分布条形图
        predicate_types = self.kg.statistics.get('predicate_types', {})
        if predicate_types:
            predicates = list(predicate_types.keys())
            counts = list(predicate_types.values())
            colors = [self.predicate_colors.get(p, '#BDC3C7') for p in predicates]
            
            ax2.bar(range(len(predicates)), counts, color=colors)
            ax2.set_xticks(range(len(predicates)))
            ax2.set_xticklabels(predicates, rotation=45, ha='right')
            ax2.set_title('Relation Type Distribution')
            ax2.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"统计图已保存到: {output_path}")
    
    def plot_top_entities(self, output_path: str = "top_entities.png",
                         limit: int = 15, figsize: Tuple[int, int] = (10, 6)) -> None:
        """绘制最重要实体图"""
        if not self.kg:
            logger.warning("知识图谱为空")
            return
        
        # 获取top实体（按频次）
        top_entities = self.kg.get_top_entities(limit=limit, by="frequency")
        
        if not top_entities:
            logger.warning("无实体数据可绘制")
            return
        
        entities, frequencies = zip(*top_entities)
        
        # 获取实体类型用于着色
        colors = []
        for entity in entities:
            entity_info = self.kg.get_entity_details(entity)
            entity_type = entity_info.get('type', 'unknown') if entity_info else 'unknown'
            colors.append(self.color_map.get(entity_type, '#95A5A6'))
        
        plt.figure(figsize=figsize)
        bars = plt.barh(range(len(entities)), frequencies, color=colors, alpha=0.7)
        
        # 添加数值标签
        for i, (bar, freq) in enumerate(zip(bars, frequencies)):
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    str(freq), ha='left', va='center')
        
        plt.yticks(range(len(entities)), [e[:20] + '...' if len(e) > 20 else e for e in entities])
        plt.xlabel('Frequency')
        plt.title(f'Top {limit} Entities by Frequency')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Top实体图已保存到: {output_path}")
    
    def plot_relation_confidence(self, output_path: str = "relation_confidence.png",
                               figsize: Tuple[int, int] = (10, 6)) -> None:
        """绘制关系置信度分布"""
        if not self.kg or not self.kg.relations:
            logger.warning("无关系数据可绘制")
            return
        
        confidences = [rel.get('confidence', 0.5) for rel in self.kg.relations]
        predicates = [rel.get('predicate', 'unknown') for rel in self.kg.relations]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 置信度分布直方图
        ax1.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Count')
        ax1.set_title('Relation Confidence Distribution')
        ax1.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidences):.3f}')
        ax1.legend()
        
        # 按谓词类型的置信度箱线图
        df = pd.DataFrame({'predicate': predicates, 'confidence': confidences})
        predicate_counts = Counter(predicates)
        # 只显示出现次数较多的谓词
        frequent_predicates = [p for p, c in predicate_counts.most_common(5)]
        df_filtered = df[df['predicate'].isin(frequent_predicates)]
        
        if not df_filtered.empty:
            sns.boxplot(data=df_filtered, x='predicate', y='confidence', ax=ax2)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
            ax2.set_title('Confidence by Relation Type')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"置信度分布图已保存到: {output_path}")
    
    def plot_adjacency_matrix(self, output_path: str = "adjacency_matrix.png",
                            max_entities: int = 20, figsize: Tuple[int, int] = (10, 8)) -> None:
        """绘制邻接矩阵热图"""
        if not self.kg or self.kg.graph.number_of_nodes() == 0:
            logger.warning("知识图谱为空")
            return
        
        # 获取最重要的实体
        top_entities = self.kg.get_top_entities(limit=max_entities, by="degree")
        entity_names = [entity[0] for entity in top_entities]
        
        # 创建邻接矩阵
        subgraph = self.kg.get_subgraph(entity_names, expand_neighbors=False)
        adj_matrix = nx.adjacency_matrix(subgraph, nodelist=entity_names)
        
        # 绘制热图
        plt.figure(figsize=figsize)
        
        # 简化实体名称
        simplified_names = [name[:15] + '...' if len(name) > 15 else name 
                          for name in entity_names]
        
        sns.heatmap(adj_matrix.toarray(), 
                   xticklabels=simplified_names, 
                   yticklabels=simplified_names,
                   cmap='Blues', 
                   cbar_kws={'label': 'Connection'},
                   square=True)
        
        plt.title(f'Entity Adjacency Matrix (Top {len(entity_names)} entities)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"邻接矩阵热图已保存到: {output_path}")
    
    def generate_all_visualizations(self, output_dir: str = "visualizations") -> None:
        """生成所有可视化图表"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        logger.info(f"开始生成所有可视化图表，输出目录: {output_dir}")
        
        try:
            # 1. 网络图
            self.plot_network_graph(os.path.join(output_dir, "network_graph.png"))
            
            # 2. 统计图
            self.plot_entity_statistics(os.path.join(output_dir, "entity_stats.png"))
            
            # 3. Top实体图
            self.plot_top_entities(os.path.join(output_dir, "top_entities.png"))
            
            # 4. 置信度分布
            self.plot_relation_confidence(os.path.join(output_dir, "relation_confidence.png"))
            
            # 5. 邻接矩阵
            self.plot_adjacency_matrix(os.path.join(output_dir, "adjacency_matrix.png"))
            
            logger.info(f"所有可视化图表已生成完毕，保存在: {output_dir}")
            
        except Exception as e:
            logger.error(f"生成可视化图表时出错: {e}")
    
    def create_html_report(self, output_path: str = "kg_report.html") -> None:
        """创建HTML可视化报告"""
        if not self.kg:
            logger.warning("知识图谱为空，无法创建报告")
            return
        
        stats = self.kg.statistics
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Biomedical Knowledge Graph Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; color: #2C3E50; }}
                .section {{ margin: 30px 0; }}
                .stats-box {{ background: #ECF0F1; padding: 15px; border-radius: 5px; }}
                .entity-list {{ display: flex; flex-wrap: wrap; gap: 10px; }}
                .entity-tag {{ background: #3498DB; color: white; padding: 5px 10px; 
                              border-radius: 15px; font-size: 12px; }}
                .microbe {{ background: #E74C3C; }}
                .metabolite {{ background: #27AE60; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #BDC3C7; padding: 8px; text-align: left; }}
                th {{ background-color: #34495E; color: white; }}
            </style>
        </head>
        <body>
            <h1 class="header">Biomedical Knowledge Graph Report</h1>
            
            <div class="section">
                <h2>Overview Statistics</h2>
                <div class="stats-box">
                    <p><strong>Total Entities:</strong> {stats.get('total_entities', 0)}</p>
                    <p><strong>Total Relations:</strong> {stats.get('total_relations', 0)}</p>
                    <p><strong>Graph Density:</strong> {stats.get('graph_density', 0)}</p>
                    <p><strong>Average Degree:</strong> {stats.get('avg_degree', 0)}</p>
                    <p><strong>Generated:</strong> {stats.get('timestamp', 'Unknown')}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Entity Type Distribution</h2>
                <div class="stats-box">
        """
        
        # 实体类型分布
        for entity_type, count in stats.get('entity_types', {}).items():
            html_content += f"<p><strong>{entity_type.capitalize()}:</strong> {count}</p>"
        
        html_content += """
                </div>
            </div>
            
            <div class="section">
                <h2>Top Entities</h2>
                <div class="entity-list">
        """
        
        # Top实体
        top_entities = self.kg.get_top_entities(limit=20)
        for entity_name, freq in top_entities:
            entity_info = self.kg.get_entity_details(entity_name)
            entity_type = entity_info.get('type', 'unknown') if entity_info else 'unknown'
            html_content += f'<span class="entity-tag {entity_type}">{entity_name} ({freq})</span>'
        
        html_content += """
                </div>
            </div>
            
            <div class="section">
                <h2>Relation Types</h2>
                <table>
                    <tr><th>Predicate</th><th>Count</th></tr>
        """
        
        # 关系类型
        for predicate, count in self.kg.get_top_predicates(limit=10):
            html_content += f"<tr><td>{predicate}</td><td>{count}</td></tr>"
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML报告已生成: {output_path}")


if __name__ == "__main__":
    # 简单测试
    from build import BiomeKnowledgeGraph
    
    kg = BiomeKnowledgeGraph()
    
    # 添加测试数据
    test_data = {
        "pmid": "test001",
        "entities": {
            "microbes": [
                {"name": "Oscillibacter", "entity_type": "microbe", "confidence_score": 0.8}
            ],
            "metabolites": [
                {"name": "cholesterol", "entity_type": "metabolite", "confidence_score": 0.9}
            ]
        },
        "relations": [
            {
                "subject": "Oscillibacter",
                "predicate": "metabolizes", 
                "object": "cholesterol",
                "evidence": "test evidence",
                "confidence": 0.8
            }
        ]
    }
    
    kg.add_extraction_result(test_data)
    
    # 创建可视化
    visualizer = KGVisualizer(kg)
    visualizer.generate_all_visualizations("test_viz")
