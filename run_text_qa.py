#!/usr/bin/env python3
"""
纯文本QA生成运行脚本
可以独立运行，也可以集成到现有流程中
"""

import asyncio
import json
import logging
import os
import argparse
from typing import Dict, Any, List

from text_qa_generator import TextQAGenerator, run_text_qa_generation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def process_text_chunks_for_qa(
    text_chunks: List[Dict[str, Any]], 
    config: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """
    处理文本块生成QA（用于替代原有的1.2阶段）
    
    Args:
        text_chunks: 文本块列表，每个包含 content, file_path, chunk_index
        config: 配置字典
        output_dir: 输出目录
        
    Returns:
        处理结果统计
    """
    generator = TextQAGenerator(config)
    
    # 转换输入格式
    formatted_chunks = []
    for chunk in text_chunks:
        formatted_chunks.append({
            'content': chunk['content'],
            'source_file': os.path.basename(chunk['file_path']),
            'chunk_index': chunk.get('chunk_index', 0)
        })
    
    # 批量处理
    logger.info(f"开始处理 {len(formatted_chunks)} 个文本块...")
    qa_results = await generator.batch_process_texts(formatted_chunks)
    
    # 筛选高质量QA
    high_quality_results = generator.filter_high_quality_qa(qa_results)
    
    # 保存结果
    chunks_dir = os.path.join(output_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    
    # 保存处理后的文本（兼容原有格式）
    processed_texts = []
    for result in high_quality_results:
        processed_texts.append({
            'source_file': result['source_file'],
            'chunk_index': result['chunk_index'],
            'content': result['qa_data'],  # QA数据作为content
            'type': 'text_qa',
            'timestamp': result['timestamp']
        })
    
    processed_file = os.path.join(chunks_dir, "ai_processed_texts.json")
    with open(processed_file, 'w', encoding='utf-8') as f:
        json.dump(processed_texts, f, ensure_ascii=False, indent=2)
    
    # 生成统计信息
    stats = generator.generate_statistics(high_quality_results)
    
    logger.info(f"文本QA处理完成：")
    logger.info(f"- 处理文本块: {len(formatted_chunks)}")
    logger.info(f"- 生成QA对: {stats['total_qa_pairs']}")
    logger.info(f"- 高质量结果: {len(high_quality_results)}")
    
    return {
        'processed_chunks': len(formatted_chunks),
        'successful_results': len(high_quality_results),
        'total_qa_pairs': stats['total_qa_pairs'],
        'statistics': stats
    }


def main():
    parser = argparse.ArgumentParser(description='运行纯文本QA生成')
    parser.add_argument('--input-dir', required=True, help='输入文本目录')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    parser.add_argument('--config', default='config_vllm_http.json', help='配置文件路径')
    parser.add_argument('--batch-size', type=int, help='批处理大小（覆盖配置文件）')
    parser.add_argument('--quality-threshold', type=float, help='质量阈值（覆盖配置文件）')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 覆盖配置参数
    if args.batch_size:
        config.setdefault('processing', {})['batch_size'] = args.batch_size
    if args.quality_threshold:
        config.setdefault('processing', {})['quality_threshold'] = args.quality_threshold
    
    # 运行生成流程
    asyncio.run(run_text_qa_generation(args.input_dir, args.output_dir, config))


if __name__ == "__main__":
    main()