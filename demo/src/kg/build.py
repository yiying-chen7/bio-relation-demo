#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识图谱构建模块
基于提取的实体和关系构建生物医学知识图谱，支持可视化和查询
"""

import json
import logging
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class BiomeKnowledgeGraph:
    """生物医学知识图谱构建器"""
    
    def __init__(self):
        self.graph = nx.DiGraph()  # 有向图
        self.entities = {}  # 实体详细信息
        self.relations = []  # 关系列表
        self.statistics = {}  # 图谱统计信息
        
    def add_extraction_result(self, extraction_result: Dict) -> None:
        """添加提取结果到知识图谱"""
        pmid = extraction_result.get("pmid", "unknown")
        logger.info(f"添加PMID={pmid}的提取结果到知识图谱")
        
        # 添加实体
        self._add_entities(extraction_result.get("entities", {}), pmid)
        
        # 添加关系
        self._add_relations(extraction_result.get("relations", []), pmid)
        
        # 更新统计信息
        self._update_statistics()
    
    def _add_entities(self, entities_data: Dict, pmid: str) -> None:
        """添加实体到图谱"""
        for entity_type, entity_list in entities_data.items():
            if entity_type in ["microbes", "metabolites"]:
                for entity in entity_list:
                    entity_name = entity.get("name", "")
                    if entity_name:
                        # 添加到NetworkX图 - 避免重复的entity_type参数
                        node_attrs = entity.copy()
                        node_attrs["graph_entity_type"] = entity_type[:-1]  # microbes -> microbe
                        self.graph.add_node(entity_name, **node_attrs)
                        
                        # 保存详细信息
                        if entity_name not in self.entities:
                            self.entities[entity_name] = {
                                "name": entity_name,
                                "type": entity_type[:-1],
                                "properties": entity,
                                "pmids": [pmid],
                                "frequency": 1
                            }
                        else:
                            # 更新频次和来源
                            if pmid not in self.entities[entity_name]["pmids"]:
                                self.entities[entity_name]["pmids"].append(pmid)
                                self.entities[entity_name]["frequency"] += 1
    
    def _add_relations(self, relations_data: List[Dict], pmid: str) -> None:
        """添加关系到图谱"""
        for relation in relations_data:
            subject = relation.get("subject", "")
            predicate = relation.get("predicate", "")
            obj = relation.get("object", "")
            evidence = relation.get("evidence", "")
            confidence = relation.get("confidence", 0.0)
            
            if subject and predicate and obj:
                # 添加到NetworkX图（如果实体存在）
                if subject in self.graph.nodes and obj in self.graph.nodes:
                    self.graph.add_edge(subject, obj, 
                                      predicate=predicate,
                                      evidence=evidence,
                                      confidence=confidence,
                                      pmid=pmid)
                
                # 保存关系信息
                relation_info = {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "evidence": evidence,
                    "confidence": confidence,
                    "pmid": pmid
                }
                self.relations.append(relation_info)
    
    def _update_statistics(self) -> None:
        """更新图谱统计信息"""
        total_nodes = self.graph.number_of_nodes()
        total_edges = self.graph.number_of_edges()
        
        # 按类型统计实体
        entity_types = Counter([data.get("entity_type", "unknown") 
                              for _, data in self.graph.nodes(data=True)])
        
        # 按谓词统计关系
        predicates = Counter([data.get("predicate", "unknown") 
                            for _, _, data in self.graph.edges(data=True)])
        
        # 计算图的基本指标
        try:
            density = nx.density(self.graph)
            if total_nodes > 0:
                avg_degree = sum(dict(self.graph.degree()).values()) / total_nodes
            else:
                avg_degree = 0
        except:
            density = 0
            avg_degree = 0
        
        self.statistics = {
            "total_entities": total_nodes,
            "total_relations": total_edges,
            "entity_types": dict(entity_types),
            "predicate_types": dict(predicates),
            "graph_density": round(density, 4),
            "avg_degree": round(avg_degree, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"图谱统计更新: {total_nodes}个实体, {total_edges}个关系")
    
    def get_entity_details(self, entity_name: str) -> Optional[Dict]:
        """获取实体详细信息"""
        return self.entities.get(entity_name)
    
    def get_relations_by_entity(self, entity_name: str) -> List[Dict]:
        """获取与指定实体相关的所有关系"""
        related_relations = []
        for relation in self.relations:
            if relation["subject"] == entity_name or relation["object"] == entity_name:
                related_relations.append(relation)
        return related_relations
    
    def get_top_entities(self, limit: int = 10, by: str = "frequency") -> List[Tuple[str, int]]:
        """获取最重要的实体"""
        if by == "frequency":
            # 按出现频次排序
            sorted_entities = sorted(self.entities.items(), 
                                   key=lambda x: x[1]["frequency"], reverse=True)
        elif by == "degree":
            # 按图中度数排序
            degrees = dict(self.graph.degree())
            sorted_entities = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
            sorted_entities = [(name, degree) for name, degree in sorted_entities]
        else:
            sorted_entities = list(self.entities.items())[:limit]
        
        return sorted_entities[:limit]
    
    def get_top_predicates(self, limit: int = 10) -> List[Tuple[str, int]]:
        """获取最常见的关系类型"""
        predicate_counts = Counter([rel["predicate"] for rel in self.relations])
        return predicate_counts.most_common(limit)
    
    def find_paths(self, source: str, target: str, max_length: int = 3) -> List[List[str]]:
        """查找两个实体间的路径"""
        try:
            if source in self.graph.nodes and target in self.graph.nodes:
                paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=max_length))
                return paths[:10]  # 限制返回数量
        except nx.NetworkXNoPath:
            pass
        return []
    
    def get_subgraph(self, entities: List[str], expand_neighbors: bool = True) -> nx.DiGraph:
        """提取子图"""
        nodes_to_include = set(entities)
        
        # 是否包含邻居节点
        if expand_neighbors:
            for entity in entities:
                if entity in self.graph.nodes:
                    # 添加直接邻居
                    neighbors = list(self.graph.neighbors(entity)) + list(self.graph.predecessors(entity))
                    nodes_to_include.update(neighbors)
        
        # 只保留存在的节点
        existing_nodes = [node for node in nodes_to_include if node in self.graph.nodes]
        
        return self.graph.subgraph(existing_nodes)
    
    def export_to_dict(self) -> Dict:
        """导出图谱为字典格式"""
        return {
            "entities": self.entities,
            "relations": self.relations,
            "statistics": self.statistics,
            "export_time": datetime.now().isoformat()
        }
    
    def export_to_json(self, filepath: str) -> None:
        """导出图谱为JSON文件"""
        data = self.export_to_dict()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"知识图谱已导出到: {filepath}")
    
    def load_from_json(self, filepath: str) -> None:
        """从JSON文件加载图谱"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.entities = data.get("entities", {})
        self.relations = data.get("relations", [])
        self.statistics = data.get("statistics", {})
        
        # 重建NetworkX图
        self._rebuild_graph()
        logger.info(f"知识图谱已从 {filepath} 加载")
    
    def _rebuild_graph(self) -> None:
        """从实体和关系数据重建NetworkX图"""
        self.graph.clear()
        
        # 添加节点
        for entity_name, entity_info in self.entities.items():
            self.graph.add_node(entity_name, **entity_info["properties"])
        
        # 添加边
        for relation in self.relations:
            subject = relation["subject"]
            obj = relation["object"]
            if subject in self.graph.nodes and obj in self.graph.nodes:
                self.graph.add_edge(subject, obj, **relation)
    
    def print_summary(self) -> None:
        """打印图谱摘要信息"""
        print("\n" + "="*50)
        print("知识图谱摘要")
        print("="*50)
        
        stats = self.statistics
        print(f"实体总数: {stats.get('total_entities', 0)}")
        print(f"关系总数: {stats.get('total_relations', 0)}")
        print(f"图密度: {stats.get('graph_density', 0)}")
        print(f"平均度数: {stats.get('avg_degree', 0)}")
        
        print("\n实体类型分布:")
        for entity_type, count in stats.get('entity_types', {}).items():
            print(f"  {entity_type}: {count}")
        
        print("\n关系类型分布:")
        for predicate, count in stats.get('predicate_types', {}).items():
            print(f"  {predicate}: {count}")
        
        print("\n最重要的实体 (按频次):")
        top_entities = self.get_top_entities(limit=5)
        for entity_name, freq in top_entities:
            entity_type = self.entities[entity_name]["type"]
            print(f"  {entity_name} ({entity_type}): {freq}次")
        
        print("\n最常见的关系:")
        top_predicates = self.get_top_predicates(limit=5)
        for predicate, count in top_predicates:
            print(f"  {predicate}: {count}次")
        
        print("="*50)


def build_kg_from_jsonl(jsonl_file: str, output_file: str = None) -> BiomeKnowledgeGraph:
    """从JSONL文件构建知识图谱"""
    logger.info(f"从 {jsonl_file} 构建知识图谱")
    
    kg = BiomeKnowledgeGraph()
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        kg.add_extraction_result(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"第{line_num}行JSON解析失败: {e}")
                        continue
    
    except FileNotFoundError:
        logger.error(f"文件不存在: {jsonl_file}")
        return kg
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        return kg
    
    # 输出摘要
    kg.print_summary()
    
    # 保存结果
    if output_file:
        kg.export_to_json(output_file)
    
    return kg


if __name__ == "__main__":
    # 测试代码
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
    
    kg = BiomeKnowledgeGraph()
    kg.add_extraction_result(test_data)
    kg.print_summary()
