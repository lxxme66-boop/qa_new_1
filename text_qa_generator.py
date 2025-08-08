#!/usr/bin/env python3
"""
纯文本QA生成器
用于处理半导体显示技术相关文本，生成高质量的问答对
"""

import json
import asyncio
import logging
from typing import List, Dict, Any
import os
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from TextGeneration.Datageneration import input_text_process
    from LocalModels.local_model_manager import LocalModelManager
except ImportError:
    logger.error("无法导入必要的模块")
    raise


class TextQAGenerator:
    """纯文本QA生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('models', {}).get('qa_generator_model', {}).get('name', 'qwen-vllm')
        self.batch_size = config.get('processing', {}).get('batch_size', 32)
        self.quality_threshold = config.get('processing', {}).get('quality_threshold', 0.7)
        
    async def process_text_for_qa(self, text_content: str, source_file: str, chunk_index: int = 0) -> Dict[str, Any]:
        """
        处理文本内容，生成QA数据
        
        Args:
            text_content: 文本内容
            source_file: 源文件名
            chunk_index: 分块索引
            
        Returns:
            包含QA对和相关信息的字典
        """
        try:
            # 使用新的半导体文本QA prompt
            # 注意：这里需要修改input_text_process函数，使其支持text_prompts
            result = await input_text_process(
                text_content,
                source_file,
                chunk_index=chunk_index,
                total_chunks=1,
                prompt_index="semiconductor_text_qa",  # 使用新的prompt
                config=self.config
            )
            
            # 解析返回的JSON结果
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    logger.error(f"无法解析JSON结果: {result[:100]}...")
                    return None
                    
            return result
            
        except Exception as e:
            logger.error(f"处理文本时出错: {e}")
            return None
            
    async def batch_process_texts(self, text_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量处理文本块
        
        Args:
            text_chunks: 文本块列表，每个包含 content, source_file, chunk_index
            
        Returns:
            处理结果列表
        """
        results = []
        
        for i in range(0, len(text_chunks), self.batch_size):
            batch = text_chunks[i:i+self.batch_size]
            logger.info(f"处理批次 {i//self.batch_size + 1}/{(len(text_chunks)-1)//self.batch_size + 1}")
            
            batch_results = await asyncio.gather(
                *(self.process_text_for_qa(
                    chunk['content'],
                    chunk['source_file'],
                    chunk.get('chunk_index', 0)
                ) for chunk in batch),
                return_exceptions=True
            )
            
            for idx, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"处理失败: {result}")
                elif result is not None:
                    results.append({
                        'source_file': batch[idx]['source_file'],
                        'chunk_index': batch[idx].get('chunk_index', 0),
                        'qa_data': result,
                        'timestamp': datetime.now().isoformat()
                    })
                    
        return results
        
    def filter_high_quality_qa(self, qa_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        筛选高质量的QA对
        
        Args:
            qa_results: QA结果列表
            
        Returns:
            筛选后的高质量QA列表
        """
        high_quality_results = []
        
        for result in qa_results:
            if 'qa_data' not in result or 'qa_pairs' not in result['qa_data']:
                continue
                
            filtered_qa_pairs = []
            for qa_pair in result['qa_data']['qa_pairs']:
                # 这里可以添加更多的质量评估逻辑
                # 例如：检查问题长度、答案完整性、是否包含关键概念等
                if self._evaluate_qa_quality(qa_pair):
                    filtered_qa_pairs.append(qa_pair)
                    
            if filtered_qa_pairs:
                result['qa_data']['qa_pairs'] = filtered_qa_pairs
                high_quality_results.append(result)
                
        logger.info(f"筛选后保留 {len(high_quality_results)} 个高质量结果")
        return high_quality_results
        
    def _evaluate_qa_quality(self, qa_pair: Dict[str, Any]) -> bool:
        """
        评估单个QA对的质量
        
        Args:
            qa_pair: QA对
            
        Returns:
            是否为高质量
        """
        # 基本质量检查
        question = qa_pair.get('question', '')
        answer = qa_pair.get('answer', '')
        
        # 检查问题和答案长度
        if len(question) < 10 or len(answer) < 20:
            return False
            
        # 检查是否包含关键信息
        if 'question_type' not in qa_pair or 'difficulty' not in qa_pair:
            return False
            
        # 可以添加更多质量评估逻辑
        # 例如：关键词检查、专业术语检查等
        
        return True
        
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """
        保存处理结果
        
        Args:
            results: 结果列表
            output_path: 输出路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"结果已保存到: {output_path}")
        
    def generate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        生成统计信息
        
        Args:
            results: 结果列表
            
        Returns:
            统计信息
        """
        total_qa_pairs = 0
        question_types = {}
        difficulty_levels = {}
        
        for result in results:
            if 'qa_data' in result and 'qa_pairs' in result['qa_data']:
                for qa_pair in result['qa_data']['qa_pairs']:
                    total_qa_pairs += 1
                    
                    # 统计问题类型
                    q_type = qa_pair.get('question_type', 'unknown')
                    question_types[q_type] = question_types.get(q_type, 0) + 1
                    
                    # 统计难度级别
                    difficulty = qa_pair.get('difficulty', 'unknown')
                    difficulty_levels[difficulty] = difficulty_levels.get(difficulty, 0) + 1
                    
        return {
            'total_files': len(set(r['source_file'] for r in results)),
            'total_chunks': len(results),
            'total_qa_pairs': total_qa_pairs,
            'question_types': question_types,
            'difficulty_levels': difficulty_levels,
            'average_qa_per_chunk': total_qa_pairs / len(results) if results else 0
        }


async def run_text_qa_generation(input_dir: str, output_dir: str, config: Dict[str, Any]):
    """
    运行文本QA生成流程
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        config: 配置字典
    """
    generator = TextQAGenerator(config)
    
    # 1. 读取输入文本
    logger.info(f"从 {input_dir} 读取文本文件...")
    text_chunks = []
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.txt', '.md', '.json')):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # 这里可以添加文本分块逻辑
                    # 暂时将整个文件作为一个块
                    text_chunks.append({
                        'content': content,
                        'source_file': file,
                        'chunk_index': 0
                    })
                    
                except Exception as e:
                    logger.error(f"读取文件 {file} 失败: {e}")
                    
    logger.info(f"共读取 {len(text_chunks)} 个文本块")
    
    # 2. 批量处理文本生成QA
    logger.info("开始生成QA对...")
    qa_results = await generator.batch_process_texts(text_chunks)
    
    # 3. 筛选高质量QA
    logger.info("筛选高质量QA对...")
    high_quality_results = generator.filter_high_quality_qa(qa_results)
    
    # 4. 保存结果
    output_path = os.path.join(output_dir, "text_qa_results.json")
    generator.save_results(high_quality_results, output_path)
    
    # 5. 生成统计信息
    stats = generator.generate_statistics(high_quality_results)
    stats_path = os.path.join(output_dir, "text_qa_statistics.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
        
    logger.info(f"QA生成完成！统计信息：")
    logger.info(f"- 总文件数: {stats['total_files']}")
    logger.info(f"- 总QA对数: {stats['total_qa_pairs']}")
    logger.info(f"- 问题类型分布: {stats['question_types']}")
    logger.info(f"- 难度级别分布: {stats['difficulty_levels']}")
    
    return high_quality_results


if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python text_qa_generator.py <input_dir> <output_dir>")
        sys.exit(1)
        
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # 加载配置
    config_path = "config_vllm_http.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    # 运行生成流程
    asyncio.run(run_text_qa_generation(input_dir, output_dir, config))