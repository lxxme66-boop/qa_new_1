#!/usr/bin/env python3
"""Test script to debug quality evaluation issue"""

import os
import json
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables for vLLM HTTP mode
os.environ['USE_VLLM_HTTP'] = 'true'
os.environ['VLLM_SERVER_URL'] = 'http://localhost:8000/v1'
os.environ['USE_LOCAL_MODELS'] = 'true'

from semiconductor_qa_generator import SemiconductorQAGenerator

async def test_quality_evaluation():
    """Test the quality evaluation with a sample text"""
    
    # Initialize the generator
    generator = SemiconductorQAGenerator(batch_size=1, gpu_devices="0")
    
    # Create a sample text for testing
    sample_text = {
        'paper_name': 'test_document.txt',
        'md_content': """
# InGaZnO靶材和薄膜的研究进展

## 摘要
对In-Ga-Zn-O (IGZO)材料推广应用过程中可能的技术阻碍进行了分析，包括IGZO的成分分析、IGZO靶材制备技术分析、IGZO-TFT (IGZO薄膜晶体管)稳定性分析等。通过调节IGZO中氧化物的成分比例，可以调节IGZO的光电性能；IGZO靶材的制备选取1400℃以上的烧结温度可以得到高密度，成分均匀的靶材；通过增加遮光层、保护层、采用双栅结构、设计补偿电路等措施，可以提高a-IGZO TFT的稳定性。

## 引言
近年来，IGZO作为一种新型的透明氧化物半导体材料，在薄膜晶体管领域得到了广泛的关注。本文将详细讨论IGZO材料的制备工艺和性能优化方法。

## 技术分析
1. IGZO靶材的制备需要精确控制烧结温度和气氛
2. 薄膜沉积过程中的氧分压对薄膜性能有重要影响
3. 后处理工艺可以显著改善薄膜的电学性能

## 结论
通过优化制备工艺，可以获得高性能的IGZO薄膜，满足新一代显示技术的需求。
""",
        'source_info': {'file': 'test_document.txt'}
    }
    
    # Test the quality evaluation
    logger.info("Testing quality evaluation...")
    results = await generator.judge_processed_texts([sample_text])
    
    # Print results
    for result in results:
        logger.info(f"Document: {result['paper_name']}")
        logger.info(f"Suitable for QA: {result['judgment']['suitable_for_qa']}")
        logger.info(f"Reason: {result['judgment']['reason']}")
        logger.info(f"Score text: {result['judgment']['score_text']}")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_quality_evaluation())