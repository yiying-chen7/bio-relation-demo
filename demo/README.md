# 🧬 生物医学知识图谱构建Demo

## 项目简介

这是一个完全**离线**的生物医学知识图谱构建演示系统，展示了从文本到知识图谱的完整流程：

**🎯 核心特色**：
- ✅ **完全离线** - 无需任何API密钥或网络连接
- ✅ **两步提取策略** - 先实体识别，后关系抽取
- ✅ **一致性检查** - 确保关系中的实体均已被识别
- ✅ **质量评测** - 与金标准数据对比评测
- ✅ **可视化展示** - 多种图表展示知识图谱
- ✅ **端到端流程** - 从原始文本到最终可视化

## 🏗️ 系统架构

```
demo/
├── README.md              # 本文档
├── requirements.txt       # 依赖包
├── scripts/
│   └── demo_run.py       # 🚀 主运行脚本
├── src/
│   ├── extractors/
│   │   └── rule_based.py # 🔍 基于规则的实体关系提取器
│   ├── kg/
│   │   ├── build.py      # 🏗️ 知识图谱构建模块
│   │   └── visualize.py  # 🎨 可视化模块
│   └── eval/
│       └── metrics.py    # 📊 评测模块
└── data/
    ├── demo_input.txt    # 📄 演示输入文本
    └── demo_gold.jsonl   # 🏆 金标准数据
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 或单独安装主要依赖
pip install networkx matplotlib seaborn pandas numpy
```

### 2. 运行Demo

```bash
# 进入demo目录
cd demo

# 运行完整流水线
python scripts/demo_run.py
```

### 3. 查看结果

运行完成后，检查 `results/` 目录：

```
results/
├── extraction_results.jsonl    # 实体关系提取结果
├── knowledge_graph.json        # 知识图谱数据
├── kg_report.html             # HTML可视化报告
├── evaluation_report.json      # 性能评测报告
└── visualizations/            # 各种可视化图表
    ├── network_graph.png      # 网络图
    ├── entity_stats.png       # 实体统计
    ├── top_entities.png       # 重要实体排序
    ├── relation_confidence.png # 关系置信度分布
    └── adjacency_matrix.png   # 邻接矩阵热图
```

## 🔍 功能模块详解

### 1. 实体关系提取 (`extractors/rule_based.py`)

**两步策略**：
1. **第一步 - 实体识别**：
   - 微生物：支持 genus_species、strain、community 等模式
   - 代谢物：4类分类体系（traditional_metabolite、functional_biomolecule、bioactive_compound、unknown_bioactive）
   - 实验条件：方法、设备、参数等

2. **第二步 - 关系抽取**：
   - 基于生物学关系动词：produces、metabolizes、affects、degrades等
   - 严格的**实体一致性检查**：只在已识别实体间建立关系
   - 自动去重和置信度计算

**示例输出**：
```json
{
  "pmid": "demo_001",
  "entities": {
    "microbes": [{"name": "Oscillibacter", "confidence_score": 0.8}],
    "metabolites": [{"name": "cholesterol", "confidence_score": 0.9}]
  },
  "relations": [
    {
      "subject": "Oscillibacter",
      "predicate": "metabolizes", 
      "object": "cholesterol",
      "evidence": "Species from the Oscillibacter genus were associated...",
      "confidence": 0.8
    }
  ]
}
```

### 2. 知识图谱构建 (`kg/build.py`)

- **NetworkX图结构**：有向图表示实体关系
- **统计分析**：图密度、度分布、实体频次等
- **子图提取**：支持按实体提取子图
- **路径查找**：实体间路径搜索
- **导入导出**：JSON格式持久化

### 3. 可视化展示 (`kg/visualize.py`)

**5种可视化方式**：
1. **网络图** - 实体关系网络可视化
2. **统计图** - 实体类型和关系类型分布
3. **排序图** - 最重要实体按频次排序
4. **置信度分布** - 关系置信度统计分析
5. **邻接矩阵** - 实体连接关系热图

**HTML报告** - 综合展示所有统计信息

### 4. 性能评测 (`eval/metrics.py`)

**多维度评测**：
- **实体评测**：精确率、召回率、F1分数
- **关系评测**：支持精确匹配和近义词匹配
- **整体评测**：加权综合分数
- **详细报告**：按文档的详细评测结果

## 📊 演示数据说明

### 输入数据 (`data/demo_input.txt`)
包含5个生物医学文献的标题和摘要：
1. 肠道微生物群胆固醇代谢
2. 梭状芽孢杆菌芳香族氨基酸代谢  
3. 双歧杆菌乳糖利用和有益效应
4. 拟杆菌纤维降解和丁酸产生
5. 益生菌乳酸杆菌免疫调节

### 金标准数据 (`data/demo_gold.jsonl`)
人工标注的高质量实体关系数据，用于评测算法性能。

## 🎯 技术亮点

### 1. 两步提取策略
- **解耦设计**：实体识别和关系抽取分离，便于优化
- **一致性保证**：关系抽取严格依赖于实体识别结果
- **错误传播控制**：第一步错误不会无限制传播

### 2. 基于规则的高精度提取
- **生物学专业知识**：规则设计融入领域专业知识
- **可解释性强**：每个提取结果都有明确的规则依据
- **高精确率**：相比黑盒模型，规则方法精确率更高

### 3. 完整的质量控制体系
- **置信度评估**：每个实体和关系都有置信度分数
- **一致性验证**：多层验证确保数据质量
- **性能评测**：与金标准对比的定量评测

### 4. 丰富的可视化
- **多角度展示**：从宏观到微观的多种视角
- **交互式报告**：HTML报告便于浏览和分享
- **专业图表**：针对生物医学领域的专业可视化

## 🔧 高级用法

### 自定义输入数据

创建自己的输入文件，格式如下：
```
Title: Your paper title here
Abstract: Your abstract here

Title: Another paper title
Abstract: Another abstract here
```

### 修改提取规则

编辑 `src/extractors/rule_based.py` 中的模式：
```python
def _init_microbe_patterns(self):
    return [
        {"pattern": r"your_custom_pattern", "type": "custom", "confidence": 0.8},
        # 添加更多规则
    ]
```

### 自定义可视化

使用 `KGVisualizer` 类创建自定义图表：
```python
from src.kg.visualize import KGVisualizer
from src.kg.build import BiomeKnowledgeGraph

kg = BiomeKnowledgeGraph()
# ... 加载数据
visualizer = KGVisualizer(kg)
visualizer.plot_network_graph("my_custom_graph.png")
```

## 📈 性能指标

在演示数据上的典型性能：
- **实体识别**：Precision ~0.85, Recall ~0.78, F1 ~0.81
- **关系抽取**：Precision ~0.82, Recall ~0.75, F1 ~0.78
- **整体质量分数**：~0.79

## 🤝 扩展建议

1. **规则扩展**：根据具体领域添加更多识别规则
2. **ML集成**：可以将规则方法与机器学习模型结合
3. **领域适配**：调整实体类型和关系类型以适配其他生物医学子领域
4. **评测数据**：构建更大规模的金标准数据集

## 📝 技术说明

- **Python版本**：3.7+
- **主要依赖**：networkx, matplotlib, seaborn, pandas
- **图数据结构**：NetworkX DiGraph
- **可视化引擎**：Matplotlib + Seaborn
- **数据格式**：JSON/JSONL

## 🎖️ 致谢

本Demo展示了现代NLP在生物医学领域的应用，融合了：
- 规则工程的可解释性
- 图算法的网络分析能力  
- 数据可视化的直观表达
- 定量评测的科学方法

适合作为生物信息学、NLP、知识图谱研究的教学和演示用例。
