#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生物医学知识图谱构建Demo
完整展示：实体提取 -> 关系提取 -> 一致性检查 -> 知识图谱构建 -> 可视化 -> 评测
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict

# 添加src路径
demo_root = Path(__file__).parent.parent
sys.path.insert(0, str(demo_root / "src"))

from extractors.rule_based import BiomeRuleExtractor
from kg.build import BiomeKnowledgeGraph, build_kg_from_jsonl
from kg.visualize import KGVisualizer
from eval.metrics import KnowledgeGraphEvaluator, evaluate_against_gold_standard

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(demo_root / "demo_run.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_input_text(input_file: str) -> List[Dict[str, str]]:
    """解析输入文本文件，提取标题和摘要"""
    documents = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按空行分割文档
    docs = content.strip().split('\n\n')
    
    for i, doc in enumerate(docs):
        if doc.strip():
            lines = doc.strip().split('\n')
            title = ""
            abstract = ""
            
            for line in lines:
                if line.startswith("Title:"):
                    title = line[6:].strip()
                elif line.startswith("Abstract:"):
                    abstract = line[9:].strip()
            
            if title and abstract:
                documents.append({
                    "pmid": f"demo_{i+1:03d}",
                    "title": title,
                    "abstract": abstract
                })
    
    return documents

def run_extraction_pipeline(input_file: str, output_dir: str) -> str:
    """运行完整的提取流水线"""
    logger.info("="*60)
    logger.info("开始生物医学知识提取流水线")
    logger.info("="*60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 解析输入文档
    logger.info("📄 解析输入文档...")
    documents = parse_input_text(input_file)
    logger.info(f"共解析到 {len(documents)} 个文档")
    
    # 初始化提取器
    logger.info("🔧 初始化规则提取器...")
    extractor = BiomeRuleExtractor()
    
    # 提取结果存储
    results = []
    extraction_file = os.path.join(output_dir, "extraction_results.jsonl")
    
    # 逐个文档提取
    logger.info("🔍 开始实体关系提取...")
    for doc in documents:
        pmid = doc["pmid"]
        title = doc["title"]
        abstract = doc["abstract"]
        
        logger.info(f"处理文档 {pmid}: {title[:50]}...")
        
        # 合并标题和摘要
        full_text = f"{title}. {abstract}"
        
        # 执行两步提取
        result = extractor.extract_complete(full_text, pmid)
        results.append(result)
        
        # 保存到文件
        with open(extraction_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # 输出摘要
        entity_count = (len(result["entities"]["microbes"]) + 
                       len(result["entities"]["metabolites"]))
        relation_count = len(result["relations"])
        logger.info(f"  ✅ {pmid}: {entity_count}个实体, {relation_count}个关系, "
                   f"质量分数: {result['quality_score']}")
    
    logger.info(f"🎉 提取完成！结果保存到: {extraction_file}")
    return extraction_file

def build_knowledge_graph(extraction_file: str, output_dir: str) -> BiomeKnowledgeGraph:
    """构建知识图谱"""
    logger.info("🏗️ 构建知识图谱...")
    
    kg = build_kg_from_jsonl(extraction_file)
    
    # 保存知识图谱
    kg_file = os.path.join(output_dir, "knowledge_graph.json")
    kg.export_to_json(kg_file)
    
    logger.info(f"📊 知识图谱已保存到: {kg_file}")
    return kg

def create_visualizations(kg: BiomeKnowledgeGraph, output_dir: str) -> None:
    """创建可视化"""
    logger.info("🎨 生成可视化图表...")
    
    visualizer = KGVisualizer(kg)
    viz_dir = os.path.join(output_dir, "visualizations")
    
    visualizer.generate_all_visualizations(viz_dir)
    
    # 创建HTML报告
    html_file = os.path.join(output_dir, "kg_report.html")
    visualizer.create_html_report(html_file)
    
    logger.info(f"🖼️ 可视化图表已生成: {viz_dir}")
    logger.info(f"📄 HTML报告: {html_file}")

def run_evaluation(extraction_file: str, gold_file: str, output_dir: str) -> None:
    """运行评测"""
    logger.info("📊 开始性能评测...")
    
    eval_file = os.path.join(output_dir, "evaluation_report.json")
    evaluator = evaluate_against_gold_standard(extraction_file, gold_file, eval_file)
    
    logger.info(f"📈 评测报告已保存到: {eval_file}")

def main():
    """主函数"""
    print("🧬 生物医学知识图谱构建Demo")
    print("="*60)
    print("功能展示：先实体后关系 + 一致性检查 + 评测 + 可视化")
    print("="*60)
    
    # 文件路径
    demo_root = Path(__file__).parent.parent
    input_file = demo_root / "data" / "demo_input.txt"
    gold_file = demo_root / "data" / "demo_gold.jsonl"
    output_dir = demo_root / "results"
    
    try:
        # 1. 实体关系提取
        extraction_file = run_extraction_pipeline(str(input_file), str(output_dir))
        
        # 2. 构建知识图谱
        kg = build_knowledge_graph(extraction_file, str(output_dir))
        
        # 3. 创建可视化
        create_visualizations(kg, str(output_dir))
        
        # 4. 运行评测
        run_evaluation(extraction_file, str(gold_file), str(output_dir))
        
        # 5. 输出最终总结
        logger.info("🎊 Demo运行完成！")
        logger.info("📁 结果文件:")
        logger.info(f"   - 提取结果: {extraction_file}")
        logger.info(f"   - 知识图谱: {output_dir}/knowledge_graph.json")
        logger.info(f"   - 可视化: {output_dir}/visualizations/")
        logger.info(f"   - HTML报告: {output_dir}/kg_report.html")
        logger.info(f"   - 评测报告: {output_dir}/evaluation_report.json")
        
        print("\n" + "="*60)
        print("🎉 Demo运行成功完成！")
        print("请查看results文件夹中的输出文件")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Demo运行失败: {e}")
        print(f"❌ 运行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
