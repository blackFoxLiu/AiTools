# 疾病知识图谱 + RAG 查询分析系统功能点文档

## 1. 概述

本系统是一个基于 LangGraph 构建的医疗领域智能分析工具，能够自动解析患者症状、病史及已有疾病补充信息，筛选高概率疾病候选，并通过多维度 LLM 分析、知识图谱检索和 RAG 增强生成，输出结构化的疾病诊断支持报告。

## 2. 核心功能模块

| 模块 | 功能描述 |
|------|----------|
| **输入解析与高概率疾病筛选** | 解析用户输入的 JSON 数据（主诉、现病史、伴随/阴性症状等），从 `existing_knowledge_supplement` 中提取疾病候选列表，根据概率阈值（默认 70%）筛选出需要深入分析的疾病 |
| **疾病分析子图** | 对每个高概率疾病依次执行 7 个维度的 LLM 分析：原始概念症状、病程、人体系统、表现特征、症状性质、严重程度、人类描述生成 |
| **知识图谱查询** | 根据疾病分析结果，查询 Neo4j 图谱中的症状节点信息、疾病节点属性、症状间关系（`symptom`）以及症状到疾病的关联（`include_disease`），并记录权重 |
| **RAG 历史命中检索** | 基于疾病名称、主诉症状、相关症状和严重程度构建查询文本，从向量数据库中检索相似的历史案例片段 |
| **结果处理与报告生成** | 整合 LLM 分析和图谱检索结果，为每个疾病附加 RAG 参考片段和图谱置信度（平均关系权重），生成格式化的最终报告 |

## 3. 工作流节点详解

系统采用分层工作流：主图（`DiseaseDataSearchService`）包含循环子图（疾病分析）和后续查询/报告节点。

### 3.1 子图：`DiseaseAnalysisSubgraph` – 多维度疾病分析

该子图独立封装，接收原始用户消息，输出高概率疾病的完整分析列表。

- **`parse_input` 节点**：解析输入消息，构建症状描述文本，筛选高概率疾病，初始化疾病列表、伴随症状等共享数据。
- **`analyze_single` 节点**：对单个疾病执行 7 个维度的 LLM 分析，调用不同提示词模板，分别生成原始概念症状、病程、人体系统、表现特征、症状性质、严重程度和人类描述文本。
- **`update_results` 节点**：将当前疾病的分析结果追加到最终列表中，并更新严重程度字段。
- **条件循环**：若还有未处理的疾病，则跳转回 `analyze_single` 继续分析；否则结束子图，返回完整分析列表。

### 3.2 主图节点：`search_rag_and_knowledge_graph` – 动态知识图谱查询

对子图输出的每个疾病分析结果，调用 `KnowledgeGraphQueryService` 执行以下查询：
- 主诉症状节点的完整属性（是否存在、节点信息）
- 疾病分析节点的完整属性
- 症状之间的 `symptom` 关系及权重
- 症状到当前疾病的 `include_disease` 关系及权重
- 基于疾病名称+症状+严重程度的 RAG 检索命中

同时将查询摘要（疾病节点是否存在、关系数量、RAG 命中数）回填到原疾病分析字典中。

### 3.3 主图节点：`process_kg_results` – 结果增强与报告生成

- 为每个疾病关联对应的知识图谱查询结果
- 提取前 2 条 RAG 命中作为 `kg_rag_reference`
- 计算症状-疾病关系的平均权重作为 `graph_confidence`
- 按照固定模板生成包含疾病名称、概率、五个分析维度的文本块，并附加 RAG 参考片段和图谱置信度
- 输出完整的格式化报告

## 4. 路由与流程

- **入口**：用户输入消息（JSON）进入主图
- **第一步**：执行疾病分析子图（循环处理每个高概率疾病）
- **第二步**：执行知识图谱+RAG查询节点（批量查询所有疾病）
- **第三步**：执行结果处理与报告生成节点
- **结束**：返回最终状态（包含分析结果、查询结果和报告）

![Hit Data Query - Knowledge Graph + RAG.jpg](images/Hit%20Data%20Query%20-%20Knowledge%20Graph%20%2B%20RAG.jpg)

## 5. 核心数据流

### 子图状态关键字段

| 字段 | 说明 |
|------|------|
| `diseases_list` | 高概率疾病候选列表（名称、症状、原因、概率） |
| `high_prob_diseases_analysis` | 最终输出的疾病分析结果列表 |
| `has_symptoms` | 伴随症状和阴性症状 |
| `disease_severity` | 最后一个疾病的严重程度分析 |
| `current_analysis` | 当前正在分析的疾病完整结果 |
| `current_index` | 当前处理到的疾病索引 |

### 主图状态关键字段

| 字段 | 说明 |
|------|------|
| `high_prob_diseases_analysis` | 子图输出的疾病分析列表 |
| `has_symptoms` | 用户伴随/阴性症状 |
| `disease_severity` | 疾病严重程度 |
| `kg_query_results` | 每个疾病的知识图谱查询结果 |
| `solution` | 最终生成的格式化报告 |

### 知识图谱查询结果结构

| 字段 | 说明 |
|------|------|
| `symptom_nodes_info` | 每个主诉症状的节点属性或存在性标志 |
| `disease_node_info` | 疾病分析节点的完整属性或不存在标志 |
| `symptom_relationships` | 症状之间的 `symptom` 关系（含权重） |
| `symptom_disease_rels` | 症状到当前疾病的 `include_disease` 关系（含权重） |
| `rag_search_hits` | RAG 检索到的相关文本片段 |

## 6. 外部依赖与集成

| 组件 | 用途 |
|------|------|
| `Neo4jQueryTools` | 知识图谱节点存在性检查、节点信息获取、Cypher 查询执行 |
| `KnowledgeBaseService` | 向量库检索（RAG），根据查询文本返回相似文档 |
| LangChain / LangChain-Ollama | LLM 调用（ChatOllama）、消息结构 |
| LangGraph | 状态图工作流（StateGraph）、消息归约器（add_messages） |
| 提示词模板 | 各分析节点使用的 System/User Prompt（从外部 .txt 文件读取） |

## 7. 典型使用流程

1. **准备输入数据**：提供包含 `chief_complaint`、`present_illness`、`associated_symptoms`、`symptom_absent`、`existing_knowledge_supplement` 和 `use_message` 的 JSON 对象。
2. **初始化服务**：创建 `DiseaseDataSearchService` 实例（可选传入自定义配置）。
3. **调用分析**：执行 `service.run(user_input)`。
4. **获取结果**：从返回字典中提取 `high_prob_diseases_analysis`（详细分析）、`kg_query_results`（原始图谱查询结果）和 `solution`（格式化报告）。
5. **使用报告**：可直接打印 `solution` 字段，或遍历分析列表进行后续处理。

## 8. 注意事项

- **提示词文件必须存在**，否则分析节点可能输出空结果。
- **知识图谱连接**需正常，且节点标签（`basic_symptom`、`disease_analysis`）和关系类型（`symptom`、`include_disease`）与代码中一致。
- **RAG 服务**需预先初始化并包含向量数据，否则检索可能返回空列表。
- **人类描述生成**依赖于用户原始会话消息（`use_message`），若无则不会生成。
- **日志级别**默认为 `INFO`，可通过调整日志配置查看更详细执行信息。