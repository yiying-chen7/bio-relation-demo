#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于规则的生物医学实体关系提取器
离线Demo版本 - 不依赖任何API或密钥
模拟原项目的两步提取策略：先实体后关系
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """实体数据结构"""
    name: str
    entity_type: str  # microbe, metabolite
    taxonomy: Dict[str, str] = None
    source: Dict[str, str] = None
    chemical_properties: Dict[str, str] = None
    confidence_score: float = 0.8

@dataclass
class Relation:
    """关系数据结构"""
    subject: str
    predicate: str
    object: str
    evidence: str
    confidence: float = 0.8
    extraction_type: str = "rule_based"

class BiomeRuleExtractor:
    """生物医学规则提取器 - 基于原项目逻辑"""
    
    def __init__(self):
        self.microbe_patterns = self._init_microbe_patterns()
        self.metabolite_patterns = self._init_metabolite_patterns()
        self.relation_patterns = self._init_relation_patterns()
        self.extracted_entities = {}
        
    def _init_microbe_patterns(self) -> List[Dict]:
        """微生物识别规则 - 基于原项目的分类学体系"""
        return [
            {"pattern": r'\b([A-Z][a-z]+)\s+([a-z]+)\b', "type": "genus_species", "confidence": 0.9},
            {"pattern": r'\b(Clostridium|Bacteroides|Oscillibacter|Prochlorococcus|Bifidobacterium)\b', "type": "genus", "confidence": 0.8},
            {"pattern": r'\b(gut microbiota|gut microbiome|intestinal microbiota)\b', "type": "community", "confidence": 0.7},
            {"pattern": r'\b([A-Z][a-z]+)\s+(strain|spp\.)\b', "type": "strain", "confidence": 0.8},
            {"pattern": r'\b(bacteria|microorganisms|microbes)\b', "type": "general", "confidence": 0.6}
        ]
    
    def _init_metabolite_patterns(self) -> List[Dict]:
        """代谢物识别规则 - 基于原项目的4类分类"""
        return [
            # traditional_metabolite
            {"pattern": r'\b(cholesterol|ATP|glucose|amino acids?|tryptophan|phenylalanine|tyrosine)\b', 
             "category": "traditional_metabolite", "confidence": 0.9},
            # functional_biomolecule 
            {"pattern": r'\b([a-z]+ase|enzyme|protein|synthetase|dehydrogenase)\b', 
             "category": "functional_biomolecule", "confidence": 0.8},
            # bioactive_compound
            {"pattern": r'\b(flavonoid|phenylacetylglutamine|phenylacetic acid|lantipeptides?)\b', 
             "category": "bioactive_compound", "confidence": 0.8},
            # Chemical compound patterns
            {"pattern": r'\b([a-z]+-[a-z]+|[A-Z][0-9]+[A-Z][0-9]+[A-Z])\b', 
             "category": "unknown_bioactive", "confidence": 0.6}
        ]
    
    def _init_relation_patterns(self) -> List[Dict]:
        """关系识别规则 - 基于原项目的生物学关系动词"""
        return [
            {"pattern": r'(.+?)\s+(produces?|produce|producing)\s+(.+)', "predicate": "produces", "confidence": 0.8},
            {"pattern": r'(.+?)\s+(metabolizes?|metabolize|metabolizing)\s+(.+)', "predicate": "metabolizes", "confidence": 0.9},
            {"pattern": r'(.+?)\s+(affects?|affect|affecting)\s+(.+)', "predicate": "affects", "confidence": 0.7},
            {"pattern": r'(.+?)\s+(associated with|correlates? with)\s+(.+)', "predicate": "associates_with", "confidence": 0.7},
            {"pattern": r'(.+?)\s+(degrades?|degrade|degrading)\s+(.+)', "predicate": "degrades", "confidence": 0.8},
            {"pattern": r'(.+?)\s+(converts?|convert|converting)\s+(.+)', "predicate": "converts", "confidence": 0.8},
            {"pattern": r'(.+?)\s+(utilizes?|utilize|utilizing)\s+(.+)', "predicate": "utilizes", "confidence": 0.8}
        ]
    
    def extract_entities_step(self, text: str, pmid: str = "demo") -> Dict:
        """第一步：实体提取 - 基于原项目的两步策略"""
        logger.info(f"PMID={pmid} 开始实体提取...")
        
        microbes = self._extract_microbes(text)
        metabolites = self._extract_metabolites(text)
        
        # 保存提取的实体供关系提取使用
        self.extracted_entities[pmid] = {
            "microbes": [e.name for e in microbes],
            "metabolites": [e.name for e in metabolites]
        }
        
        result = {
            "entities": {
                "microbes": [asdict(e) for e in microbes],
                "metabolites": [asdict(e) for e in metabolites],
                "experimental_conditions": self._extract_experimental_conditions(text)
            }
        }
        
        total_entities = len(microbes) + len(metabolites)
        logger.info(f"PMID={pmid} 实体提取完成：{total_entities}个实体 ({len(microbes)}微生物, {len(metabolites)}代谢物)")
        
        return result
    
    def extract_relations_step(self, text: str, entities_result: Dict, pmid: str = "demo") -> List[Dict]:
        """第二步：关系提取 - 只在已提取实体间建立关系"""
        logger.info(f"PMID={pmid} 开始关系提取...")
        
        if pmid not in self.extracted_entities:
            logger.warning(f"PMID={pmid} 未找到预提取的实体，跳过关系提取")
            return []
        
        entity_names = (self.extracted_entities[pmid]["microbes"] + 
                       self.extracted_entities[pmid]["metabolites"])
        
        relations = []
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # 跳过太短的句子
                continue
                
            for pattern_info in self.relation_patterns:
                matches = re.finditer(pattern_info["pattern"], sentence, re.IGNORECASE)
                
                for match in matches:
                    subject_text = match.group(1).strip()
                    object_text = match.group(3).strip() if len(match.groups()) >= 3 else ""
                    
                    # 实体一致性检查 - 关键改进点
                    subject_entity = self._find_matching_entity(subject_text, entity_names)
                    object_entity = self._find_matching_entity(object_text, entity_names)
                    
                    if subject_entity and object_entity and subject_entity != object_entity:
                        relation = Relation(
                            subject=subject_entity,
                            predicate=pattern_info["predicate"],
                            object=object_entity,
                            evidence=sentence,
                            confidence=pattern_info["confidence"],
                            extraction_type="rule_based"
                        )
                        relations.append(asdict(relation))
        
        # 去重
        relations = self._deduplicate_relations(relations)
        
        logger.info(f"PMID={pmid} 关系提取完成：{len(relations)}个关系")
        return relations
    
    def _extract_microbes(self, text: str) -> List[Entity]:
        """提取微生物实体"""
        microbes = []
        found_names = set()
        
        for pattern_info in self.microbe_patterns:
            matches = re.finditer(pattern_info["pattern"], text, re.IGNORECASE)
            
            for match in matches:
                if pattern_info["type"] == "genus_species":
                    name = f"{match.group(1)} {match.group(2)}"
                    taxonomy = {"genus": match.group(1), "species": match.group(2)}
                else:
                    name = match.group(1) if len(match.groups()) >= 1 else match.group(0)
                    taxonomy = {"genus": name.split()[0] if " " in name else name}
                
                if name.lower() not in found_names:
                    microbes.append(Entity(
                        name=name,
                        entity_type="microbe",
                        taxonomy=taxonomy,
                        source={"body_part": "gut" if "gut" in text.lower() else "not_specified"},
                        confidence_score=pattern_info["confidence"]
                    ))
                    found_names.add(name.lower())
        
        return microbes
    
    def _extract_metabolites(self, text: str) -> List[Entity]:
        """提取代谢物实体"""
        metabolites = []
        found_names = set()
        
        for pattern_info in self.metabolite_patterns:
            matches = re.finditer(pattern_info["pattern"], text, re.IGNORECASE)
            
            for match in matches:
                name = match.group(1) if len(match.groups()) >= 1 else match.group(0)
                
                if name.lower() not in found_names and len(name) > 2:
                    metabolites.append(Entity(
                        name=name,
                        entity_type="metabolite",
                        chemical_properties={"category": pattern_info["category"]},
                        confidence_score=pattern_info["confidence"]
                    ))
                    found_names.add(name.lower())
        
        return metabolites
    
    def _extract_experimental_conditions(self, text: str) -> Dict:
        """提取实验条件 - 基于原项目逻辑"""
        methods = []
        equipment = []
        
        method_patterns = [
            r'\b(PCR|Western blot|chromatography|spectroscopy|metabolomics|metagenomics)\b',
            r'\b(in vitro|in vivo|genome analysis|functional prediction)\b'
        ]
        
        for pattern in method_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            methods.extend([m for m in matches if m not in methods])
        
        return {
            "methods": methods,
            "equipment": equipment,
            "treatments": [],
            "physical_parameters": {}
        }
    
    def _find_matching_entity(self, text: str, entity_names: List[str]) -> Optional[str]:
        """在已提取实体中查找匹配 - 实体一致性检查"""
        text_clean = text.lower().strip()
        
        # 完全匹配
        for entity in entity_names:
            if entity.lower() == text_clean:
                return entity
        
        # 部分匹配
        for entity in entity_names:
            if entity.lower() in text_clean or text_clean in entity.lower():
                return entity
        
        return None
    
    def _deduplicate_relations(self, relations: List[Dict]) -> List[Dict]:
        """关系去重"""
        seen = set()
        unique_relations = []
        
        for rel in relations:
            key = (rel["subject"], rel["predicate"], rel["object"])
            if key not in seen:
                seen.add(key)
                unique_relations.append(rel)
        
        return unique_relations
    
    def extract_complete(self, text: str, pmid: str = "demo") -> Dict:
        """完整的两步提取流程 - 主要API"""
        start_time = datetime.now()
        
        # 第一步：实体提取
        entities_result = self.extract_entities_step(text, pmid)
        
        # 第二步：关系提取
        relations = self.extract_relations_step(text, entities_result, pmid)
        
        # 一致性验证
        valid_relations = self._validate_relations_consistency(relations, entities_result, pmid)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        result = {
            "pmid": pmid,
            "status": "success",
            "entities": entities_result["entities"],
            "relations": valid_relations,
            "timestamp": end_time.isoformat(),
            "processing_time": processing_time,
            "quality_score": self._calculate_quality_score(entities_result, valid_relations)
        }
        
        logger.info(f"PMID={pmid} 完整提取完成，用时{processing_time:.2f}秒")
        return result
    
    def _validate_relations_consistency(self, relations: List[Dict], entities_result: Dict, pmid: str) -> List[Dict]:
        """关系一致性验证 - 基于原项目的质量控制逻辑"""
        all_entity_names = []
        
        for microbe in entities_result["entities"]["microbes"]:
            all_entity_names.append(microbe["name"])
        
        for metabolite in entities_result["entities"]["metabolites"]:
            all_entity_names.append(metabolite["name"])
        
        valid_relations = []
        for relation in relations:
            subject_valid = relation["subject"] in all_entity_names
            object_valid = relation["object"] in all_entity_names
            
            if subject_valid and object_valid:
                valid_relations.append(relation)
            else:
                logger.debug(f"PMID={pmid} 跳过无效关系: {relation['subject']} -> {relation['object']}")
        
        logger.info(f"PMID={pmid} 关系一致性验证：{len(valid_relations)}/{len(relations)} 通过")
        return valid_relations
    
    def _calculate_quality_score(self, entities_result: Dict, relations: List[Dict]) -> float:
        """质量分数计算 - 简化版本"""
        entity_count = (len(entities_result["entities"]["microbes"]) + 
                       len(entities_result["entities"]["metabolites"]))
        relation_count = len(relations)
        
        # 基础分数
        base_score = 0.5
        
        # 实体奖励
        entity_bonus = min(entity_count * 0.05, 0.3)
        
        # 关系奖励  
        relation_bonus = min(relation_count * 0.1, 0.4)
        
        # 多样性奖励
        diversity_bonus = 0.1 if entity_count > 0 and relation_count > 0 else 0
        
        total_score = min(base_score + entity_bonus + relation_bonus + diversity_bonus, 1.0)
        return round(total_score, 2)


if __name__ == "__main__":
    # 简单测试
    extractor = BiomeRuleExtractor()
    
    test_text = """
    The gut microbiota plays crucial roles in cholesterol metabolism. 
    Species from the Oscillibacter genus were associated with decreased cholesterol levels.
    Clostridium sporogenes produces aromatic amino acid metabolites including tryptophan.
    """
    
    result = extractor.extract_complete(test_text, "test_001")
    print("提取结果：")
    print(json.dumps(result, indent=2, ensure_ascii=False))
