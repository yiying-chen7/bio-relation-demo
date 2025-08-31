#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测模块
提供实体关系提取和知识图谱构建的评测指标
"""

import json
import logging
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class EntityEvaluator:
    """实体提取评测器"""
    
    def __init__(self):
        self.precision_scores = []
        self.recall_scores = []
        self.f1_scores = []
    
    def evaluate_entities(self, predicted_entities: List[str], 
                         gold_entities: List[str]) -> Dict[str, float]:
        """评测实体提取结果"""
        predicted_set = set([e.lower().strip() for e in predicted_entities])
        gold_set = set([e.lower().strip() for e in gold_entities])
        
        # 计算指标
        true_positives = len(predicted_set & gold_set)
        predicted_count = len(predicted_set)
        gold_count = len(gold_set)
        
        precision = true_positives / predicted_count if predicted_count > 0 else 0.0
        recall = true_positives / gold_count if gold_count > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        self.precision_scores.append(precision)
        self.recall_scores.append(recall)
        self.f1_scores.append(f1)
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "true_positives": true_positives,
            "predicted_count": predicted_count,
            "gold_count": gold_count
        }
    
    def get_average_scores(self) -> Dict[str, float]:
        """获取平均分数"""
        if not self.precision_scores:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        return {
            "avg_precision": round(sum(self.precision_scores) / len(self.precision_scores), 4),
            "avg_recall": round(sum(self.recall_scores) / len(self.recall_scores), 4),
            "avg_f1": round(sum(self.f1_scores) / len(self.f1_scores), 4)
        }

class RelationEvaluator:
    """关系提取评测器"""
    
    def __init__(self):
        self.precision_scores = []
        self.recall_scores = []
        self.f1_scores = []
    
    def evaluate_relations(self, predicted_relations: List[Dict], 
                          gold_relations: List[Dict],
                          exact_match: bool = False) -> Dict[str, float]:
        """评测关系提取结果"""
        
        # 转换为元组格式便于比较
        def relation_to_tuple(rel, exact=False):
            if exact:
                return (rel["subject"].lower().strip(), 
                       rel["predicate"].lower().strip(), 
                       rel["object"].lower().strip())
            else:
                # 允许谓词的近义词匹配
                predicate = self._normalize_predicate(rel["predicate"].lower().strip())
                return (rel["subject"].lower().strip(), 
                       predicate, 
                       rel["object"].lower().strip())
        
        predicted_set = set([relation_to_tuple(rel, exact_match) for rel in predicted_relations])
        gold_set = set([relation_to_tuple(rel, exact_match) for rel in gold_relations])
        
        # 计算指标
        true_positives = len(predicted_set & gold_set)
        predicted_count = len(predicted_set)
        gold_count = len(gold_set)
        
        precision = true_positives / predicted_count if predicted_count > 0 else 0.0
        recall = true_positives / gold_count if gold_count > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        self.precision_scores.append(precision)
        self.recall_scores.append(recall)
        self.f1_scores.append(f1)
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "true_positives": true_positives,
            "predicted_count": predicted_count,
            "gold_count": gold_count,
            "exact_match": exact_match
        }
    
    def _normalize_predicate(self, predicate: str) -> str:
        """标准化谓词，处理近义词"""
        predicate_map = {
            "produces": ["produces", "generates", "creates", "synthesizes"],
            "metabolizes": ["metabolizes", "processes", "breaks down"],
            "affects": ["affects", "influences", "impacts", "modulates"],
            "associates_with": ["associates with", "correlates with", "relates to"],
            "degrades": ["degrades", "decomposes", "breaks down"],
            "converts": ["converts", "transforms", "changes"],
            "utilizes": ["utilizes", "uses", "consumes"]
        }
        
        for standard, synonyms in predicate_map.items():
            if predicate in synonyms:
                return standard
        
        return predicate
    
    def get_average_scores(self) -> Dict[str, float]:
        """获取平均分数"""
        if not self.precision_scores:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        return {
            "avg_precision": round(sum(self.precision_scores) / len(self.precision_scores), 4),
            "avg_recall": round(sum(self.recall_scores) / len(self.recall_scores), 4),
            "avg_f1": round(sum(self.f1_scores) / len(self.f1_scores), 4)
        }

class KnowledgeGraphEvaluator:
    """知识图谱整体评测器"""
    
    def __init__(self):
        self.entity_evaluator = EntityEvaluator()
        self.relation_evaluator = RelationEvaluator()
        self.results = []
    
    def evaluate_extraction_result(self, predicted_result: Dict, 
                                 gold_result: Dict) -> Dict[str, any]:
        """评测单个提取结果"""
        pmid = predicted_result.get("pmid", "unknown")
        logger.info(f"评测PMID={pmid}的提取结果")
        
        # 准备实体列表
        pred_entities = self._extract_entity_names(predicted_result.get("entities", {}))
        gold_entities = self._extract_entity_names(gold_result.get("entities", {}))
        
        # 评测实体
        entity_metrics = self.entity_evaluator.evaluate_entities(pred_entities, gold_entities)
        
        # 评测关系
        pred_relations = predicted_result.get("relations", [])
        gold_relations = gold_result.get("relations", [])
        relation_metrics = self.relation_evaluator.evaluate_relations(pred_relations, gold_relations)
        
        # 计算整体质量分数
        overall_score = self._calculate_overall_score(entity_metrics, relation_metrics)
        
        result = {
            "pmid": pmid,
            "entity_metrics": entity_metrics,
            "relation_metrics": relation_metrics,
            "overall_score": overall_score,
            "timestamp": datetime.now().isoformat()
        }
        
        self.results.append(result)
        return result
    
    def _extract_entity_names(self, entities_data: Dict) -> List[str]:
        """从实体数据中提取实体名称列表"""
        names = []
        for entity_type, entity_list in entities_data.items():
            if entity_type in ["microbes", "metabolites"]:
                for entity in entity_list:
                    if isinstance(entity, dict) and "name" in entity:
                        names.append(entity["name"])
                    elif isinstance(entity, str):
                        names.append(entity)
        return names
    
    def _calculate_overall_score(self, entity_metrics: Dict, relation_metrics: Dict) -> float:
        """计算整体质量分数"""
        entity_f1 = entity_metrics.get("f1", 0.0)
        relation_f1 = relation_metrics.get("f1", 0.0)
        
        # 实体和关系F1的加权平均，实体权重0.4，关系权重0.6
        overall = 0.4 * entity_f1 + 0.6 * relation_f1
        return round(overall, 4)
    
    def get_summary_report(self) -> Dict[str, any]:
        """获取评测总结报告"""
        if not self.results:
            return {"message": "No evaluation results available"}
        
        # 实体评测汇总
        entity_avg = self.entity_evaluator.get_average_scores()
        
        # 关系评测汇总
        relation_avg = self.relation_evaluator.get_average_scores()
        
        # 整体分数统计
        overall_scores = [r["overall_score"] for r in self.results]
        avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0
        
        # 按PMID的详细结果
        detailed_results = []
        for result in self.results:
            detailed_results.append({
                "pmid": result["pmid"],
                "entity_f1": result["entity_metrics"]["f1"],
                "relation_f1": result["relation_metrics"]["f1"],
                "overall_score": result["overall_score"]
            })
        
        return {
            "summary": {
                "total_evaluated": len(self.results),
                "entity_metrics": entity_avg,
                "relation_metrics": relation_avg,
                "overall_avg_score": round(avg_overall, 4)
            },
            "detailed_results": detailed_results,
            "generated_at": datetime.now().isoformat()
        }
    
    def export_report(self, filepath: str) -> None:
        """导出评测报告"""
        report = self.get_summary_report()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"评测报告已导出到: {filepath}")
    
    def print_summary(self) -> None:
        """打印评测摘要"""
        report = self.get_summary_report()
        
        if "message" in report:
            print(report["message"])
            return
        
        summary = report["summary"]
        
        print("\n" + "="*60)
        print("知识图谱评测报告")
        print("="*60)
        print(f"评测样本数: {summary['total_evaluated']}")
        print()
        
        print("实体提取性能:")
        entity_metrics = summary["entity_metrics"]
        print(f"  平均精确率: {entity_metrics['avg_precision']:.4f}")
        print(f"  平均召回率: {entity_metrics['avg_recall']:.4f}")
        print(f"  平均F1分数: {entity_metrics['avg_f1']:.4f}")
        print()
        
        print("关系提取性能:")
        relation_metrics = summary["relation_metrics"]
        print(f"  平均精确率: {relation_metrics['avg_precision']:.4f}")
        print(f"  平均召回率: {relation_metrics['avg_recall']:.4f}")
        print(f"  平均F1分数: {relation_metrics['avg_f1']:.4f}")
        print()
        
        print(f"整体平均分数: {summary['overall_avg_score']:.4f}")
        print()
        
        # 显示前5个详细结果
        print("详细结果 (前5个):")
        for i, result in enumerate(report["detailed_results"][:5]):
            print(f"  {i+1}. PMID={result['pmid']}: "
                  f"实体F1={result['entity_f1']:.3f}, "
                  f"关系F1={result['relation_f1']:.3f}, "
                  f"整体={result['overall_score']:.3f}")
        
        if len(report["detailed_results"]) > 5:
            print(f"  ... (还有{len(report['detailed_results']) - 5}个结果)")
        
        print("="*60)

def evaluate_against_gold_standard(predicted_file: str, gold_file: str, 
                                 output_file: str = None) -> KnowledgeGraphEvaluator:
    """与金标准数据比较评测"""
    logger.info(f"开始评测: 预测文件={predicted_file}, 金标准={gold_file}")
    
    evaluator = KnowledgeGraphEvaluator()
    
    # 读取预测结果
    predicted_data = {}
    try:
        with open(predicted_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    pmid = data.get("pmid", "unknown")
                    predicted_data[pmid] = data
    except Exception as e:
        logger.error(f"读取预测文件失败: {e}")
        return evaluator
    
    # 读取金标准数据
    gold_data = {}
    try:
        with open(gold_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    pmid = data.get("pmid", "unknown")
                    gold_data[pmid] = data
    except Exception as e:
        logger.error(f"读取金标准文件失败: {e}")
        return evaluator
    
    # 进行评测
    common_pmids = set(predicted_data.keys()) & set(gold_data.keys())
    logger.info(f"找到{len(common_pmids)}个共同的PMID进行评测")
    
    for pmid in common_pmids:
        evaluator.evaluate_extraction_result(predicted_data[pmid], gold_data[pmid])
    
    # 输出报告
    evaluator.print_summary()
    
    if output_file:
        evaluator.export_report(output_file)
    
    return evaluator

if __name__ == "__main__":
    # 测试代码
    predicted = {
        "pmid": "test001",
        "entities": {
            "microbes": [{"name": "Oscillibacter"}],
            "metabolites": [{"name": "cholesterol"}]
        },
        "relations": [
            {"subject": "Oscillibacter", "predicate": "metabolizes", "object": "cholesterol"}
        ]
    }
    
    gold = {
        "pmid": "test001", 
        "entities": {
            "microbes": [{"name": "Oscillibacter"}],
            "metabolites": [{"name": "cholesterol"}, {"name": "ATP"}]
        },
        "relations": [
            {"subject": "Oscillibacter", "predicate": "metabolizes", "object": "cholesterol"}
        ]
    }
    
    evaluator = KnowledgeGraphEvaluator()
    result = evaluator.evaluate_extraction_result(predicted, gold)
    print("评测结果:", json.dumps(result, indent=2, ensure_ascii=False))
