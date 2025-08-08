#!/usr/bin/env python3
"""
半导体显示技术文本QA数据集生成系统 - 文本专用版本
专门用于处理纯文本内容，生成高质量的问答对数据集
"""

import os
import sys
import json
import asyncio
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from TextGeneration.text_qa_processor import TextQAProcessor
from TextGeneration.Datageneration import parse_txt
from TextQA.quality_assessment import QualityAssessment

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('text_qa_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def run_text_qa_pipeline(
    config: dict,
    input_dir: str = "data/texts",
    output_dir: str = "data/text_qa_results",
    batch_size: int = 2,
    quality_threshold: float = 0.7
):
    """
    运行文本QA生成流水线
    
    Args:
        config: 配置字典
        input_dir: 输入文本目录
        output_dir: 输出结果目录
        batch_size: 批处理大小
        quality_threshold: 质量阈值
    """
    logger.info("=" * 80)
    logger.info("🚀 启动半导体显示技术文本QA数据集生成系统")
    logger.info("=" * 80)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    chunks_dir = os.path.join(output_dir, "chunks")
    qa_results_dir = os.path.join(output_dir, "qa_results")
    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(qa_results_dir, exist_ok=True)
    
    text_files = []
    
    try:
        # ===== 第一阶段：文本预处理 + QA生成 =====
        logger.info("第一阶段: 文本预处理和QA生成")
        
        # 步骤1.1: 文本分块和预处理
        logger.info("步骤1.1: 文本分块和预处理...")
        all_tasks = []
        
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    text_files.append(file_path)
                    
                    # 使用现有的parse_txt函数进行文本分块
                    file_tasks = await parse_txt(file_path, index=9, config=config)
                    
                    if file_tasks:
                        logger.info(f"为文件 {file} 创建了 {len(file_tasks)} 个处理任务")
                        all_tasks.extend(file_tasks)
        
        if not all_tasks:
            logger.error("没有找到可处理的文本文件")
            return []
        
        logger.info(f"总共创建了 {len(all_tasks)} 个文本处理任务")
        
        # 步骤1.2: 文本QA生成处理 (改进版本)
        logger.info("步骤1.2: 文本QA生成处理...")
        
        # 初始化文本QA处理器
        text_qa_processor = TextQAProcessor(config)
        
        # 批量处理文本，生成问答对
        processed_results = await text_qa_processor.batch_process_texts(all_tasks, batch_size)
        
        # 保存QA生成结果
        qa_results_file = os.path.join(chunks_dir, "text_qa_generated.json")
        with open(qa_results_file, 'w', encoding='utf-8') as f:
            json.dump(processed_results, f, ensure_ascii=False, indent=2)
        
        total_qa_count = sum(result.get("total_qa_count", 0) for result in processed_results)
        logger.info(f"文本QA生成完成，总共生成了 {total_qa_count} 个问答对")
        
        if not processed_results:
            logger.error("没有QA生成结果，流程终止")
            return []
        
        # ===== 第二阶段：QA质量评估和过滤 =====
        logger.info("第二阶段: QA质量评估和过滤")
        
        # 步骤2.1: 提取所有问答对进行质量评估
        logger.info("步骤2.1: 提取和整理问答对...")
        
        all_qa_pairs = []
        qa_statistics = {
            "total_texts": len(processed_results),
            "total_qa_pairs": 0,
            "by_type": {"factual": 0, "comparative": 0, "reasoning": 0, "open_ended": 0},
            "by_difficulty": {"basic": 0, "intermediate": 0, "advanced": 0}
        }
        
        for result in processed_results:
            source_file = result.get("source_file", "unknown")
            chunk_index = result.get("chunk_index", 0)
            
            for q_type, qa_list in result.get("qa_pairs", {}).items():
                for qa in qa_list:
                    # 添加元数据
                    qa_with_meta = {
                        **qa,
                        "source_file": source_file,
                        "chunk_index": chunk_index,
                        "qa_id": f"{source_file}_{chunk_index}_{q_type}_{len(all_qa_pairs)}"
                    }
                    all_qa_pairs.append(qa_with_meta)
                    
                    # 统计信息
                    qa_statistics["total_qa_pairs"] += 1
                    qa_statistics["by_type"][q_type] = qa_statistics["by_type"].get(q_type, 0) + 1
                    difficulty = qa.get("difficulty", "intermediate")
                    qa_statistics["by_difficulty"][difficulty] = qa_statistics["by_difficulty"].get(difficulty, 0) + 1
        
        logger.info(f"提取了 {len(all_qa_pairs)} 个问答对进行质量评估")
        logger.info(f"问答对类型分布: {qa_statistics['by_type']}")
        logger.info(f"问答对难度分布: {qa_statistics['by_difficulty']}")
        
        # 步骤2.2: 质量评估
        logger.info("步骤2.2: 问答对质量评估...")
        
        # 初始化质量评估器
        quality_assessor = QualityAssessment(config)
        
        # 批量质量评估
        evaluated_qa_pairs = []
        for i in range(0, len(all_qa_pairs), batch_size):
            batch = all_qa_pairs[i:i+batch_size]
            logger.info(f"评估批次 {i//batch_size + 1}/{(len(all_qa_pairs)-1)//batch_size + 1}")
            
            batch_results = await asyncio.gather(
                *(quality_assessor.assess_qa_quality(qa) for qa in batch),
                return_exceptions=True
            )
            
            for j, result in enumerate(batch_results):
                if not isinstance(result, Exception) and result is not None:
                    qa_with_quality = {**batch[j], "quality_assessment": result}
                    evaluated_qa_pairs.append(qa_with_quality)
                else:
                    logger.error(f"质量评估失败: {result}")
        
        # 步骤2.3: 根据质量阈值过滤
        logger.info("步骤2.3: 根据质量阈值过滤问答对...")
        
        high_quality_qa_pairs = []
        quality_stats = {"passed": 0, "failed": 0, "total": len(evaluated_qa_pairs)}
        
        for qa in evaluated_qa_pairs:
            quality_score = qa.get("quality_assessment", {}).get("overall_score", 0)
            if quality_score >= quality_threshold:
                high_quality_qa_pairs.append(qa)
                quality_stats["passed"] += 1
            else:
                quality_stats["failed"] += 1
        
        logger.info(f"质量过滤完成: {quality_stats}")
        logger.info(f"高质量问答对数量: {len(high_quality_qa_pairs)}")
        
        # ===== 第三阶段：结果整理和保存 =====
        logger.info("第三阶段: 结果整理和保存")
        
        # 保存评估后的所有问答对
        all_qa_file = os.path.join(qa_results_dir, "all_qa_with_quality.json")
        with open(all_qa_file, 'w', encoding='utf-8') as f:
            json.dump(evaluated_qa_pairs, f, ensure_ascii=False, indent=2)
        
        # 保存高质量问答对
        final_qa_file = os.path.join(qa_results_dir, "high_quality_qa_dataset.json")
        with open(final_qa_file, 'w', encoding='utf-8') as f:
            json.dump(high_quality_qa_pairs, f, ensure_ascii=False, indent=2)
        
        # 生成统计报告
        final_stats = {
            **qa_statistics,
            "quality_stats": quality_stats,
            "quality_threshold": quality_threshold,
            "final_dataset_size": len(high_quality_qa_pairs),
            "processing_summary": {
                "input_files": len(text_files),
                "text_chunks": len(all_tasks),
                "generated_qa_pairs": qa_statistics["total_qa_pairs"],
                "high_quality_qa_pairs": len(high_quality_qa_pairs),
                "quality_pass_rate": quality_stats["passed"] / quality_stats["total"] if quality_stats["total"] > 0 else 0
            }
        }
        
        stats_file = os.path.join(output_dir, "processing_statistics.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(final_stats, f, ensure_ascii=False, indent=2)
        
        logger.info("=" * 80)
        logger.info("🎉 文本QA数据集生成完成!")
        logger.info(f"📊 处理统计:")
        logger.info(f"   输入文件: {len(text_files)}")
        logger.info(f"   文本块: {len(all_tasks)}")
        logger.info(f"   生成问答对: {qa_statistics['total_qa_pairs']}")
        logger.info(f"   高质量问答对: {len(high_quality_qa_pairs)}")
        logger.info(f"   质量通过率: {final_stats['processing_summary']['quality_pass_rate']:.2%}")
        logger.info(f"📁 输出目录: {output_dir}")
        logger.info("=" * 80)
        
        return high_quality_qa_pairs
        
    except Exception as e:
        logger.error(f"流水线执行失败: {e}")
        import traceback
        traceback.print_exc()
        return []

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"配置文件加载失败: {e}")
        # 返回默认配置
        return {
            "api": {
                "use_vllm_http": True,
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

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='半导体显示技术文本QA数据集生成系统')
    parser.add_argument('--input-dir', type=str, default='data/texts', help='输入文本目录')
    parser.add_argument('--output-dir', type=str, default='data/text_qa_results', help='输出结果目录')
    parser.add_argument('--config', type=str, default='config_vllm_http.json', help='配置文件路径')
    parser.add_argument('--batch-size', type=int, default=2, help='批处理大小')
    parser.add_argument('--quality-threshold', type=float, default=0.7, help='质量阈值(0-1)')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 运行流水线
    try:
        results = asyncio.run(run_text_qa_pipeline(
            config=config,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            quality_threshold=args.quality_threshold
        ))
        
        if results:
            logger.info(f"✅ 成功生成 {len(results)} 个高质量问答对")
        else:
            logger.error("❌ 未能生成任何问答对")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("用户中断程序执行")
        sys.exit(0)
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()