#!/usr/bin/env python3
"""Test vLLM HTTP client directly"""

import os
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables
os.environ['USE_VLLM_HTTP'] = 'true'
os.environ['VLLM_SERVER_URL'] = 'http://localhost:8000/v1'
os.environ['USE_LOCAL_MODELS'] = 'true'

from LocalModels.local_model_manager import LocalModelManager

def test_vllm_http():
    """Test vLLM HTTP client directly"""
    
    # Create config
    config = {
        'api': {
            'use_local_models': True,
            'use_vllm_http': True,
            'vllm_server_url': 'http://localhost:8000/v1',
            'default_backend': 'vllm_http'
        },
        'models': {
            'local_models': {
                'default_backend': 'vllm_http',
                'vllm_http': {
                    'base_url': 'http://localhost:8000/v1',
                    'api_key': 'dummy-key',
                    'model_name': 'qwen-vllm',
                    'temperature': 0.6,
                    'max_tokens': 512
                }
            }
        }
    }
    
    # Initialize LocalModelManager
    manager = LocalModelManager(config)
    
    # Test simple generation
    test_prompt = """请判断以下文本是否适合用于生成逻辑推理问题，只回答【是】或【否】：

    文本内容：IGZO是一种新型的透明氧化物半导体材料，在薄膜晶体管领域得到了广泛的关注。
    
    回答："""
    
    logger.info("Testing vLLM HTTP generation...")
    response = manager.generate(test_prompt)
    logger.info(f"Response: {response}")
    
    return response

if __name__ == "__main__":
    test_vllm_http()