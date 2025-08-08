# AI文本处理 - 大模型调用流程文档

## 概述
本文档详细说明了AI文本处理系统中大模型的调用位置、prompt使用方式和整体流程。

## 1. 主要调用位置

### 1.1 核心调用函数
大模型调用的核心位置在 `TextGeneration/Datageneration.py` 文件中：

- **函数名称**: `input_text_process` (第320-500行)
- **功能**: 处理文本内容并调用大模型生成结果
- **支持模式**: 
  - API模式（OpenAI兼容接口）
  - 本地模型模式（通过LocalModelManager）

### 1.2 调用链路
```
text_processor.py (入口)
  └── process_single_txt() 
      └── parse_txt() (TextGeneration/Datageneration.py)
          └── input_text_process() 
              └── 大模型调用（API或本地）
```

## 2. 大模型调用方式

### 2.1 API调用方式
```python
# 使用AsyncOpenAI客户端
client = AsyncOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=120.0
)

response = await client.chat.completions.create(
    model="qwen-vllm",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": formatted_prompt}
    ],
    temperature=0.8,
    max_tokens=4096,
    top_p=0.9,
)
```

### 2.2 本地模型调用方式
```python
# 通过LocalModelManager
local_model_manager = LocalModelManager(config)
content = await local_model_manager.generate(
    prompt=formatted_prompt,
    system_prompt=system_prompt,
    temperature=0.8,
    max_tokens=4096,
    top_p=0.9
)
```

## 3. Prompt模板系统

### 3.1 系统Prompt
定义在 `TextGeneration/prompts_conf.py`:
```python
system_prompt = """
你是一位显示半导体专家。现在我需要你的协助。
"""
```

### 3.2 用户Prompt模板
系统包含40+个预定义的prompt模板，主要类别：

#### 图片分析类 (1-9)
- prompt 1-4: 图片知识提取和分析
- prompt 5-7: 图片描述和解释
- prompt 8-9: 基于图片的问答生成

#### 文本处理类
- `text_qa_basic`: 基础问答对生成
- `text_qa_advanced`: 研究生水平问答生成
- `text_multimodal_prep`: 多模态场景准备

#### 质量检查类 (36-40)
- prompt 36: 推理有效性检查
- prompt 37: 问题清晰度检查
- prompt 38: 图片质量检查
- prompt 39: 答案正确性检查
- prompt 40: 难度评估

### 3.3 Prompt使用示例
```python
# 选择prompt模板
user_prompt = user_prompts[prompt_index]

# 格式化prompt
formatted_prompt = user_prompt.format(
    text_content=text_content,
    source_file=source_file,
    chunk_info=f"(Chunk {chunk_index + 1}/{total_chunks})"
)
```

## 4. 配置管理

### 4.1 主要配置项 (config.json)
```json
{
  "api": {
    "use_local_models": true,
    "use_vllm_http": true,
    "vllm_server_url": "http://localhost:8000/v1",
    "api_key": "EMPTY"
  },
  "models": {
    "qa_generator_model": {
      "name": "qwen-vllm",
      "temperature": 0.7,
      "max_tokens": 2048,
      "max_model_len": 32768
    }
  }
}
```

### 4.2 模型参数
- **温度 (temperature)**: 0.7-0.8 (控制生成的随机性)
- **最大令牌数 (max_tokens)**: 2048-4096
- **Top-p**: 0.9-0.95
- **重复惩罚 (repetition_penalty)**: 1.1

## 5. 处理流程详解

### 5.1 文本预处理
1. 读取文本文件
2. 文本分块（chunk_size=2000, overlap=200）
3. 为每个块创建处理任务

### 5.2 模型调用流程
1. 选择合适的prompt模板（根据index参数）
2. 格式化prompt（插入文本内容、文件信息等）
3. 根据配置选择调用方式（API或本地）
4. 发送请求并等待响应
5. 处理和返回结果

### 5.3 错误处理
- 超时重试机制（最多5次）
- 连接失败自动降级
- 详细的错误日志记录

## 6. 使用示例

### 6.1 命令行调用
```bash
# 处理单个文本文件
python text_processor.py --input data/texts/sample.txt --index 9

# 处理文件夹
python text_processor.py --input data/texts --batch-size 100
```

### 6.2 代码调用
```python
# 创建处理器
processor = TextProcessor("config.json")

# 处理文本
results = await processor.process_single_txt("sample.txt", index=9)
```

## 7. 性能优化

### 7.1 并发处理
- 支持批量处理多个文件
- 异步调用提高效率
- 可配置并发数量

### 7.2 缓存机制
- 启用缓存避免重复处理
- 缓存目录可配置

## 8. 注意事项

1. **模型选择**: 根据任务需求选择合适的prompt模板
2. **参数调整**: 根据生成质量调整temperature等参数
3. **错误处理**: 注意处理API调用失败的情况
4. **资源管理**: 大模型调用消耗较多资源，注意控制并发数

## 9. 扩展开发

### 9.1 添加新的Prompt模板
在 `prompts_conf.py` 中添加新的模板：
```python
user_prompts[新编号] = """
您的prompt模板内容
"""
```

### 9.2 支持新的模型后端
在 `LocalModels/` 目录下实现新的客户端类，并在 `LocalModelManager` 中注册。

## 10. 故障排查

### 常见问题
1. **连接失败**: 检查vLLM服务是否运行
2. **超时错误**: 增加timeout配置或减少max_tokens
3. **内存不足**: 减少batch_size或chunk_size
4. **生成质量差**: 调整temperature或更换prompt模板