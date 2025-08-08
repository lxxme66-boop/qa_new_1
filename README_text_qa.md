# 文本QA数据集生成系统

## 概述

这是一个专门用于处理**纯文本内容**的QA数据集生成系统，针对半导体显示技术领域进行了优化。系统能够自动从文本中生成高质量的问答对，支持四种不同类型的问题。

## 🆕 改进的1.2阶段功能

### 主要改进

1. **专用文本QA处理器** (`TextQAProcessor`)
   - 专门处理纯文本内容
   - 并发生成多种类型的问答对
   - 内置质量验证机制

2. **四种问题类型支持**
   - **事实型** (15%): 询问具体数值、参数、指标
   - **比较型** (15%): 比较不同材料、方法、结构
   - **推理型** (50%): 探究原理、机制、原因
   - **开放型** (20%): 优化建议、改进方法

3. **智能质量评估**
   - 基础质量检查 (长度、格式、完整性)
   - AI驱动的多维度评估
   - 自动过滤低质量问答对

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 启动vLLM服务器
python start_vllm_server.py
```

### 2. 运行文本QA生成

```bash
# 使用默认配置
python run_semiconductor_qa_text.py \
    --input-dir data/texts \
    --output-dir data/text_qa_results \
    --config config_vllm_http.json

# 自定义参数
python run_semiconductor_qa_text.py \
    --input-dir /path/to/your/texts \
    --output-dir /path/to/output \
    --batch-size 4 \
    --quality-threshold 0.8
```

### 3. 参数说明

- `--input-dir`: 输入文本目录 (默认: `data/texts`)
- `--output-dir`: 输出结果目录 (默认: `data/text_qa_results`)
- `--config`: 配置文件路径 (默认: `config_vllm_http.json`)
- `--batch-size`: 批处理大小 (默认: 2)
- `--quality-threshold`: 质量阈值 0-1 (默认: 0.7)

## 📁 输出结构

```
output/
├── chunks/
│   └── text_qa_generated.json          # 原始QA生成结果
├── qa_results/
│   ├── all_qa_with_quality.json        # 所有问答对+质量评估
│   └── high_quality_qa_dataset.json    # 高质量问答对数据集
└── processing_statistics.json          # 处理统计报告
```

## 🔧 核心组件

### 1. TextQAProcessor

专门的文本QA处理器，负责：
- 解析和分块文本内容
- 并发生成不同类型的问答对
- 验证和清理生成的数据

```python
from TextGeneration.text_qa_processor import TextQAProcessor

# 初始化处理器
processor = TextQAProcessor(config)

# 处理文本块
result = await processor.process_text_chunk(
    text_content="你的文本内容",
    source_file="文件名.txt",
    chunk_index=0
)
```

### 2. QualityAssessment

智能质量评估器，提供：
- 基础质量检查 (格式、长度等)
- AI驱动的多维度评估
- 综合质量评分和过滤

```python
from TextQA.quality_assessment import QualityAssessment

# 初始化评估器
assessor = QualityAssessment(config)

# 评估问答对质量
quality_result = await assessor.assess_qa_quality(qa_pair)
```

## 📊 质量评估维度

| 维度 | 权重 | 说明 |
|------|------|------|
| 问题清晰度 | 25% | 问题表述是否清楚、准确、易理解 |
| 答案准确性 | 30% | 答案是否正确、完整、有逻辑 |
| 内容相关性 | 25% | 是否与半导体显示技术相关 |
| 难度适宜性 | 20% | 问题难度是否适合目标受众 |

## 🎯 问题类型示例

### 事实型问题 (15%)
```json
{
    "question": "IGZO TFT的典型阈值电压范围是多少？",
    "answer": "IGZO TFT的典型阈值电压范围通常在1-3V之间...",
    "question_type": "factual",
    "difficulty": "basic"
}
```

### 比较型问题 (15%)
```json
{
    "question": "顶栅结构与底栅结构IGZO TFT在寄生电容方面有什么差异？",
    "answer": "顶栅结构的IGZO TFT相比底栅结构具有更低的寄生电容...",
    "question_type": "comparative",
    "difficulty": "intermediate"
}
```

### 推理型问题 (50%)
```json
{
    "question": "为什么氧空位增加会导致IGZO薄膜迁移率的变化？",
    "answer": "氧空位的增加会影响IGZO薄膜的载流子浓度和散射机制...",
    "question_type": "reasoning",
    "difficulty": "advanced"
}
```

### 开放型问题 (20%)
```json
{
    "question": "如何改善氧化物TFT的阈值电压漂移问题？",
    "answer": "改善阈值电压漂移可以从以下几个方面入手：1)优化界面质量...",
    "question_type": "open_ended",
    "difficulty": "advanced"
}
```

## ⚙️ 配置说明

### config_vllm_http.json 示例

```json
{
  "api": {
    "use_vllm_http": true,
    "vllm_server_url": "http://localhost:8000/v1",
    "api_key": "EMPTY"
  },
  "models": {
    "qa_generator_model": {
      "name": "qwen-vllm",
      "temperature": 0.7,
      "max_tokens": 2048,
      "timeout": 120.0
    }
  }
}
```

## 📈 性能统计

系统会自动生成详细的处理统计报告：

```json
{
  "processing_summary": {
    "input_files": 10,
    "text_chunks": 45,
    "generated_qa_pairs": 180,
    "high_quality_qa_pairs": 144,
    "quality_pass_rate": 0.8
  },
  "by_type": {
    "factual": {"total": 27, "passed": 24},
    "comparative": {"total": 27, "passed": 22},
    "reasoning": {"total": 90, "passed": 72},
    "open_ended": {"total": 36, "passed": 26}
  }
}
```

## 🔍 故障排查

### 常见问题

1. **vLLM服务连接失败**
   ```bash
   # 检查服务是否启动
   curl http://localhost:8000/health
   
   # 重启vLLM服务
   python start_vllm_server.py
   ```

2. **生成的问答对质量不高**
   - 调整 `--quality-threshold` 参数 (建议0.6-0.8)
   - 检查输入文本质量
   - 增加batch处理时间间隔

3. **内存不足**
   - 减小 `--batch-size` 参数
   - 分批处理大量文本文件

## 📝 日志记录

系统会生成详细的日志文件：
- `text_qa_generation.log`: 完整的处理日志
- 控制台输出: 实时处理状态

## 🚀 高级功能

### 自定义问题类型

修改 `TextQAProcessor` 中的 `question_types` 配置：

```python
self.question_types = {
    "factual": {"ratio": 0.20, ...},
    "comparative": {"ratio": 0.20, ...},
    "reasoning": {"ratio": 0.40, ...},
    "open_ended": {"ratio": 0.20, ...}
}
```

### 自定义质量评估标准

修改 `QualityAssessment` 中的 `quality_criteria` 权重：

```python
self.quality_criteria = {
    "question_clarity": {"weight": 0.30},
    "answer_accuracy": {"weight": 0.40},
    "content_relevance": {"weight": 0.20},
    "difficulty_level": {"weight": 0.10}
}
```

## 📞 技术支持

如有问题或建议，请查看：
1. 日志文件中的错误信息
2. 配置文件是否正确
3. vLLM服务是否正常运行

---

**注意**: 此系统专门针对文本QA生成进行了优化，如需处理图像相关内容，请使用原始的 `run_semiconductor_qa.py` 系统。