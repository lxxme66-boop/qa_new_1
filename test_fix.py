#!/usr/bin/env python3
"""测试修复是否有效"""
import os
import sys
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置环境变量
os.environ['USE_VLLM_HTTP'] = 'true'
os.environ['VLLM_SERVER_URL'] = 'http://localhost:8000/v1'

logger.info(f"设置环境变量: USE_VLLM_HTTP={os.environ.get('USE_VLLM_HTTP')}")

# 导入并初始化SemiconductorQAGenerator
from semiconductor_qa_generator import SemiconductorQAGenerator

logger.info("初始化SemiconductorQAGenerator...")
generator = SemiconductorQAGenerator(batch_size=1, gpu_devices="0")

# 检查状态
logger.info(f"generator.use_vllm_http = {generator.use_vllm_http}")
logger.info(f"generator.llm = {generator.llm}")
logger.info(f"generator.local_model_manager = {generator.local_model_manager}")

# 测试生成方法
test_prompts = ["测试提示"]
logger.info("测试_generate方法...")
try:
    results = generator._generate(test_prompts)
    logger.info(f"生成成功: {len(results)} 个结果")
    for i, result in enumerate(results):
        logger.info(f"结果 {i}: {result.outputs[0].text[:50]}...")
except Exception as e:
    logger.error(f"生成失败: {e}")
    import traceback
    traceback.print_exc()

logger.info("测试完成")