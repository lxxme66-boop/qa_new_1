# Prompt调用流程图

```mermaid
graph TD
    Start[开始] --> Phase1[第一阶段: 文本预处理 + 质量评估]
    
    %% 第一阶段
    Phase1 --> Step1_1[步骤1.1: 文本分块<br/>extract_text_chunks]
    Step1_1 --> Step1_2[步骤1.2: AI文本处理<br/>Prompt ID: 9]
    Step1_2 --> Step1_3[步骤1.3: 文本质量评估<br/>score_template]
    
    Step1_3 --> Decision1{文本通过评估?}
    Decision1 -->|否| End1[结束流程]
    Decision1 -->|是| Phase2[第二阶段: QA生成]
    
    %% 第二阶段
    Phase2 --> Step2_1[步骤2.1: 分类问题生成<br/>prompt_template + 分类增强]
    Step2_1 --> Step2_2[步骤2.2: 问题格式转换<br/>convert_questionlist_li_data]
    Step2_2 --> Step2_3[步骤2.3: 问题质量评估<br/>evaluator_template]
    Step2_3 --> Step2_4[步骤2.4: 答案生成<br/>COT答案模板]
    
    Step2_4 --> Phase3[第三阶段: 数据增强]
    
    %% 第三阶段
    Phase3 --> Step3_1[数据改写与增强<br/>build_prompt]
    Step3_1 --> End2[完成]
    
    %% 样式定义
    classDef phaseStyle fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    classDef promptStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef decisionStyle fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    
    class Phase1,Phase2,Phase3 phaseStyle
    class Step1_2,Step1_3,Step2_1,Step2_3,Step2_4,Step3_1 promptStyle
    class Decision1 decisionStyle
```

## Prompt调用详细说明

### 第一阶段 Prompts

| 步骤 | Prompt名称 | 调用位置 | 主要功能 |
|------|-----------|---------|----------|
| 1.2 | Prompt ID 9 | `input_text_process()` | 处理文本并生成图片相关的QA数据 |
| 1.3 | score_template | `judge_processed_texts()` | 评估文本质量，判断是否适合生成推理问题 |

### 第二阶段 Prompts

| 步骤 | Prompt名称 | 调用位置 | 主要功能 |
|------|-----------|---------|----------|
| 2.1 | prompt_template + 分类增强 | `generate_classified_questions()` | 生成不同类型的问题 |
| 2.3 | evaluator_template | `judge_question_data()` | 评估问题质量 |
| 2.4 | COT答案模板 | `generate_answers()` | 为问题生成详细答案 |

### 第三阶段 Prompts

| 步骤 | Prompt名称 | 调用位置 | 主要功能 |
|------|-----------|---------|----------|
| 3.1 | build_prompt | `enhance_qa_data()` | 改写和增强QA数据 |

### 特殊用途 Prompts

| ID | 用途 | 使用场景 |
|----|------|----------|
| 343 | 多模态图片问题生成 | 处理包含图片的文档 |
| 36 | 推理有效性检查 | 质量控制 |
| 37 | 问题清晰度检查 | 质量控制 |
| 39 | 答案正确性检查 | 质量控制 |
| 40 | 难度适中性检查 | 质量控制 |

## 问题类型分布

```mermaid
pie title 问题类型分布
    "推理型 (reasoning)" : 50
    "开放型 (open)" : 20
    "事实型 (factual)" : 15
    "比较型 (comparative)" : 15
```