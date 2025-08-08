"""
专门用于文本问答对生成和处理的模块
适用于纯文本内容的QA数据集生成
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class TextQAProcessor:
    """文本问答对处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_config = config.get('api', {})
        self.models_config = config.get('models', {})
        
        # API配置
        self.api_base = self.api_config.get('vllm_server_url', "http://localhost:8000/v1")
        self.api_key = self.api_config.get('api_key', "EMPTY")
        self.model_name = self.models_config.get('qa_generator_model', {}).get('name', "qwen-vllm")
        self.timeout = self.models_config.get('qa_generator_model', {}).get('timeout', 120.0)
        
        # QA生成配置
        self.question_types = {
            "factual": {
                "ratio": 0.15,
                "description": "事实型问题：获取指标、数值、性能参数等",
                "prompt_key": "text_qa_factual"
            },
            "comparative": {
                "ratio": 0.15,
                "description": "比较型问题：比较不同材料、结构或方案等",
                "prompt_key": "text_qa_comparative"
            },
            "reasoning": {
                "ratio": 0.50,
                "description": "推理型问题：机制原理解释，探究原因",
                "prompt_key": "text_qa_reasoning"
            },
            "open_ended": {
                "ratio": 0.20,
                "description": "开放型问题：优化建议，改进方法",
                "prompt_key": "text_qa_open_ended"
            }
        }
    
    async def process_text_chunk(self, text_content: str, source_file: str, 
                                chunk_index: int = 0, total_chunks: int = 1) -> Dict[str, Any]:
        """
        处理单个文本块，生成多种类型的问答对
        
        Args:
            text_content: 文本内容
            source_file: 源文件名
            chunk_index: 分块索引
            total_chunks: 总分块数
            
        Returns:
            处理结果，包含多种类型的问答对
        """
        try:
            logger.info(f"开始处理文本块 {chunk_index+1}/{total_chunks} - 大小: {len(text_content)}字符")
            
            # 并发生成不同类型的问答对
            qa_tasks = []
            for q_type, config in self.question_types.items():
                task = self._generate_qa_by_type(text_content, q_type, source_file, chunk_index)
                qa_tasks.append(task)
            
            # 等待所有任务完成
            qa_results = await asyncio.gather(*qa_tasks, return_exceptions=True)
            
            # 整理结果
            result = {
                "source_file": source_file,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "text_content": text_content,
                "text_length": len(text_content),
                "qa_pairs": {},
                "processing_status": "success",
                "error_messages": []
            }
            
            # 处理各类型QA结果
            for i, (q_type, q_config) in enumerate(self.question_types.items()):
                if i < len(qa_results) and not isinstance(qa_results[i], Exception):
                    result["qa_pairs"][q_type] = qa_results[i]
                else:
                    error_msg = f"{q_type}类型问答对生成失败: {qa_results[i] if i < len(qa_results) else '未知错误'}"
                    result["error_messages"].append(error_msg)
                    logger.error(error_msg)
                    result["qa_pairs"][q_type] = []
            
            # 计算总的问答对数量
            total_qa_count = sum(len(qas) for qas in result["qa_pairs"].values())
            result["total_qa_count"] = total_qa_count
            
            logger.info(f"文本块处理完成，生成了 {total_qa_count} 个问答对")
            return result
            
        except Exception as e:
            logger.error(f"文本块处理失败: {e}")
            return {
                "source_file": source_file,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "text_content": text_content,
                "text_length": len(text_content),
                "qa_pairs": {},
                "total_qa_count": 0,
                "processing_status": "failed",
                "error_messages": [str(e)]
            }
    
    async def _generate_qa_by_type(self, text_content: str, question_type: str, 
                                  source_file: str, chunk_index: int) -> List[Dict[str, Any]]:
        """
        根据指定类型生成问答对
        
        Args:
            text_content: 文本内容
            question_type: 问题类型
            source_file: 源文件名
            chunk_index: 分块索引
            
        Returns:
            该类型的问答对列表
        """
        try:
            # 获取对应类型的prompt模板
            prompt_template = self._get_prompt_template(question_type)
            
            # 格式化prompt
            formatted_prompt = prompt_template.format(
                text_content=text_content,
                source_file=source_file,
                chunk_info=f"(Chunk {chunk_index + 1})" if chunk_index > 0 else "",
                question_type=question_type
            )
            
            # 调用API生成问答对
            response = await self._call_api(formatted_prompt)
            
            # 解析响应
            qa_pairs = self._parse_qa_response(response, question_type)
            
            logger.info(f"{question_type}类型生成了 {len(qa_pairs)} 个问答对")
            return qa_pairs
            
        except Exception as e:
            logger.error(f"{question_type}类型问答对生成失败: {e}")
            return []
    
    def _get_prompt_template(self, question_type: str) -> str:
        """获取指定类型的prompt模板"""
        templates = {
            "factual": """
基于以下文本内容，生成事实型问答对。要求：
1. 问题应该询问具体的数值、参数、指标等事实信息
2. 答案应该准确、具体，直接来源于文本
3. 生成3-5个高质量的问答对

文本内容：
{text_content}

请以JSON格式返回，结构如下：
{{
    "qa_pairs": [
        {{
            "question": "问题内容",
            "answer": "答案内容",
            "question_type": "factual",
            "difficulty": "basic",
            "key_concepts": ["概念1", "概念2"],
            "source_evidence": "支撑答案的文本片段"
        }}
    ]
}}
""",
            
            "comparative": """
基于以下文本内容，生成比较型问答对。要求：
1. 问题应该比较不同的材料、方法、结构或性能等
2. 答案应该清楚地指出差异和对比要点
3. 生成3-5个高质量的问答对

文本内容：
{text_content}

请以JSON格式返回，结构如下：
{{
    "qa_pairs": [
        {{
            "question": "问题内容",
            "answer": "答案内容",
            "question_type": "comparative",
            "difficulty": "intermediate",
            "key_concepts": ["概念1", "概念2"],
            "comparison_aspects": ["对比维度1", "对比维度2"]
        }}
    ]
}}
""",
            
            "reasoning": """
基于以下文本内容，生成推理型问答对。要求：
1. 问题应该询问原理、机制、原因等需要推理的内容
2. 答案应该包含详细的推理过程和科学解释
3. 生成5-8个高质量的问答对（占比最高）

文本内容：
{text_content}

请以JSON格式返回，结构如下：
{{
    "qa_pairs": [
        {{
            "question": "问题内容",
            "answer": "答案内容",
            "question_type": "reasoning",
            "difficulty": "advanced",
            "reasoning_steps": ["推理步骤1", "推理步骤2", "推理步骤3"],
            "scientific_principles": ["相关原理1", "相关原理2"],
            "key_concepts": ["概念1", "概念2"]
        }}
    ]
}}
""",
            
            "open_ended": """
基于以下文本内容，生成开放型问答对。要求：
1. 问题应该询问优化建议、改进方法、解决方案等
2. 答案应该提供具体的建议和可行的方案
3. 生成3-4个高质量的问答对

文本内容：
{text_content}

请以JSON格式返回，结构如下：
{{
    "qa_pairs": [
        {{
            "question": "问题内容",
            "answer": "答案内容",
            "question_type": "open_ended",
            "difficulty": "advanced",
            "solution_approaches": ["方法1", "方法2"],
            "practical_considerations": ["考虑因素1", "考虑因素2"],
            "key_concepts": ["概念1", "概念2"]
        }}
    ]
}}
"""
        }
        
        return templates.get(question_type, templates["factual"])
    
    async def _call_api(self, prompt: str) -> str:
        """调用API生成响应"""
        try:
            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.timeout
            )
            
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一位半导体显示技术领域的专家，擅长生成高质量的问答对。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2048
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            raise
    
    def _parse_qa_response(self, response: str, question_type: str) -> List[Dict[str, Any]]:
        """解析API响应，提取问答对"""
        try:
            # 尝试解析JSON响应
            response_data = json.loads(response)
            qa_pairs = response_data.get("qa_pairs", [])
            
            # 验证和清理数据
            validated_pairs = []
            for qa in qa_pairs:
                if self._validate_qa_pair(qa, question_type):
                    validated_pairs.append(qa)
                else:
                    logger.warning(f"问答对验证失败: {qa.get('question', 'N/A')}")
            
            return validated_pairs
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            # 尝试从文本中提取问答对
            return self._extract_qa_from_text(response, question_type)
        except Exception as e:
            logger.error(f"响应解析失败: {e}")
            return []
    
    def _validate_qa_pair(self, qa: Dict[str, Any], question_type: str) -> bool:
        """验证问答对的质量"""
        # 基本字段检查
        if not qa.get("question") or not qa.get("answer"):
            return False
        
        # 长度检查
        if len(qa["question"]) < 10 or len(qa["answer"]) < 20:
            return False
        
        # 类型一致性检查
        if qa.get("question_type") != question_type:
            qa["question_type"] = question_type
        
        return True
    
    def _extract_qa_from_text(self, text: str, question_type: str) -> List[Dict[str, Any]]:
        """从文本中提取问答对（当JSON解析失败时的备用方案）"""
        # 简单的文本解析逻辑
        qa_pairs = []
        lines = text.split('\n')
        
        current_question = None
        current_answer = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('问题') or line.startswith('Q:') or line.startswith('问:'):
                if current_question and current_answer:
                    qa_pairs.append({
                        "question": current_question,
                        "answer": current_answer,
                        "question_type": question_type,
                        "difficulty": "intermediate",
                        "key_concepts": []
                    })
                current_question = line.split(':', 1)[-1].strip()
                current_answer = None
            elif line.startswith('答案') or line.startswith('A:') or line.startswith('答:'):
                current_answer = line.split(':', 1)[-1].strip()
        
        # 添加最后一对
        if current_question and current_answer:
            qa_pairs.append({
                "question": current_question,
                "answer": current_answer,
                "question_type": question_type,
                "difficulty": "intermediate",
                "key_concepts": []
            })
        
        return qa_pairs

    async def batch_process_texts(self, all_tasks: List[Dict[str, Any]], 
                                 batch_size: int = 2) -> List[Dict[str, Any]]:
        """
        批量处理文本任务
        
        Args:
            all_tasks: 所有文本处理任务
            batch_size: 批处理大小
            
        Returns:
            处理结果列表
        """
        processed_results = []
        
        for i in range(0, len(all_tasks), batch_size):
            batch = all_tasks[i:i+batch_size]
            logger.info(f"处理批次 {i//batch_size + 1}/{(len(all_tasks)-1)//batch_size + 1}")
            
            batch_results = await asyncio.gather(
                *(self.process_text_chunk(
                    task["content"], 
                    task["file_path"],
                    chunk_index=task["chunk_index"],
                    total_chunks=task.get("total_chunks", 1)
                ) for task in batch),
                return_exceptions=True
            )
            
            for result in batch_results:
                if not isinstance(result, Exception) and result is not None:
                    processed_results.append(result)
                else:
                    logger.error(f"批次处理失败: {result}")
        
        return processed_results