"""
文本问答对质量评估模块
用于评估生成的问答对的质量，包括准确性、完整性、难度等维度
"""

import json
import logging
from typing import Dict, List, Any, Optional
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class QualityAssessment:
    """问答对质量评估器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_config = config.get('api', {})
        self.models_config = config.get('models', {})
        
        # API配置
        self.api_base = self.api_config.get('vllm_server_url', "http://localhost:8000/v1")
        self.api_key = self.api_config.get('api_key', "EMPTY")
        self.model_name = self.models_config.get('qa_generator_model', {}).get('name', "qwen-vllm")
        self.timeout = self.models_config.get('qa_generator_model', {}).get('timeout', 120.0)
        
        # 质量评估标准
        self.quality_criteria = {
            "question_clarity": {
                "weight": 0.25,
                "description": "问题的清晰度和表述质量"
            },
            "answer_accuracy": {
                "weight": 0.30,
                "description": "答案的准确性和完整性"
            },
            "content_relevance": {
                "weight": 0.25,
                "description": "与原文内容的相关性"
            },
            "difficulty_level": {
                "weight": 0.20,
                "description": "问题难度的适宜性"
            }
        }
    
    async def assess_qa_quality(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估单个问答对的质量
        
        Args:
            qa_pair: 问答对数据
            
        Returns:
            质量评估结果
        """
        try:
            question = qa_pair.get("question", "")
            answer = qa_pair.get("answer", "")
            question_type = qa_pair.get("question_type", "")
            
            # 基础质量检查
            basic_checks = self._basic_quality_check(qa_pair)
            if not basic_checks["passed"]:
                return {
                    "overall_score": 0.0,
                    "detailed_scores": {},
                    "passed": False,
                    "issues": basic_checks["issues"],
                    "assessment_method": "basic_check"
                }
            
            # AI质量评估
            ai_assessment = await self._ai_quality_assessment(qa_pair)
            
            # 合并评估结果
            final_assessment = self._combine_assessments(basic_checks, ai_assessment)
            
            return final_assessment
            
        except Exception as e:
            logger.error(f"质量评估失败: {e}")
            return {
                "overall_score": 0.0,
                "detailed_scores": {},
                "passed": False,
                "issues": [f"评估过程出错: {str(e)}"],
                "assessment_method": "error"
            }
    
    def _basic_quality_check(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        """基础质量检查"""
        issues = []
        question = qa_pair.get("question", "")
        answer = qa_pair.get("answer", "")
        
        # 检查必要字段
        if not question or not answer:
            issues.append("缺少问题或答案")
        
        # 检查长度
        if len(question) < 10:
            issues.append("问题过短")
        if len(answer) < 20:
            issues.append("答案过短")
        if len(question) > 500:
            issues.append("问题过长")
        if len(answer) > 2000:
            issues.append("答案过长")
        
        # 检查内容质量
        if question.count('?') == 0 and question.count('？') == 0:
            issues.append("问题缺少疑问标记")
        
        # 检查重复内容
        if question.lower() in answer.lower():
            issues.append("答案中包含问题内容，可能存在重复")
        
        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "basic_score": 1.0 if len(issues) == 0 else max(0.0, 1.0 - len(issues) * 0.2)
        }
    
    async def _ai_quality_assessment(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        """使用AI进行质量评估"""
        try:
            prompt = self._build_assessment_prompt(qa_pair)
            response = await self._call_api(prompt)
            
            # 解析评估结果
            assessment_result = self._parse_assessment_response(response)
            return assessment_result
            
        except Exception as e:
            logger.error(f"AI质量评估失败: {e}")
            return {
                "ai_score": 0.5,
                "detailed_scores": {
                    "question_clarity": 0.5,
                    "answer_accuracy": 0.5,
                    "content_relevance": 0.5,
                    "difficulty_level": 0.5
                },
                "feedback": f"AI评估失败: {str(e)}"
            }
    
    def _build_assessment_prompt(self, qa_pair: Dict[str, Any]) -> str:
        """构建质量评估prompt"""
        question = qa_pair.get("question", "")
        answer = qa_pair.get("answer", "")
        question_type = qa_pair.get("question_type", "")
        
        prompt = f"""
请对以下问答对进行质量评估，从四个维度进行评分（1-5分）：

问题类型: {question_type}
问题: {question}
答案: {answer}

评估维度：
1. 问题清晰度 (question_clarity): 问题表述是否清楚、准确、易理解
2. 答案准确性 (answer_accuracy): 答案是否正确、完整、有逻辑
3. 内容相关性 (content_relevance): 问答对是否与半导体显示技术相关
4. 难度适宜性 (difficulty_level): 问题难度是否适合目标受众

请以JSON格式返回评估结果：
{{
    "question_clarity": {{
        "score": 分数(1-5),
        "reason": "评价理由"
    }},
    "answer_accuracy": {{
        "score": 分数(1-5),
        "reason": "评价理由"
    }},
    "content_relevance": {{
        "score": 分数(1-5),
        "reason": "评价理由"
    }},
    "difficulty_level": {{
        "score": 分数(1-5),
        "reason": "评价理由"
    }},
    "overall_feedback": "总体评价和建议",
    "strengths": ["优点1", "优点2"],
    "weaknesses": ["不足1", "不足2"]
}}
"""
        return prompt
    
    async def _call_api(self, prompt: str) -> str:
        """调用API进行评估"""
        try:
            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.timeout
            )
            
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一位专业的教育质量评估专家，专门评估半导体显示技术领域的问答对质量。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # 较低温度确保评估一致性
                max_tokens=1024
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            raise
    
    def _parse_assessment_response(self, response: str) -> Dict[str, Any]:
        """解析评估响应"""
        try:
            # 尝试解析JSON
            assessment_data = json.loads(response)
            
            # 提取分数
            detailed_scores = {}
            for criterion in self.quality_criteria.keys():
                if criterion in assessment_data:
                    score_data = assessment_data[criterion]
                    if isinstance(score_data, dict):
                        detailed_scores[criterion] = min(5, max(1, score_data.get("score", 3))) / 5.0
                    else:
                        detailed_scores[criterion] = min(5, max(1, float(score_data))) / 5.0
                else:
                    detailed_scores[criterion] = 0.6  # 默认分数
            
            # 计算加权平均分
            ai_score = sum(
                detailed_scores[criterion] * self.quality_criteria[criterion]["weight"]
                for criterion in self.quality_criteria.keys()
            )
            
            return {
                "ai_score": ai_score,
                "detailed_scores": detailed_scores,
                "feedback": assessment_data.get("overall_feedback", ""),
                "strengths": assessment_data.get("strengths", []),
                "weaknesses": assessment_data.get("weaknesses", [])
            }
            
        except json.JSONDecodeError:
            # 如果JSON解析失败，使用简单的文本分析
            return self._simple_text_assessment(response)
        except Exception as e:
            logger.error(f"评估响应解析失败: {e}")
            return {
                "ai_score": 0.5,
                "detailed_scores": {criterion: 0.5 for criterion in self.quality_criteria.keys()},
                "feedback": f"解析失败: {str(e)}"
            }
    
    def _simple_text_assessment(self, response: str) -> Dict[str, Any]:
        """简单的文本评估（当JSON解析失败时）"""
        # 基于关键词的简单评估
        positive_keywords = ["好", "优秀", "清晰", "准确", "完整", "相关", "适合"]
        negative_keywords = ["差", "不好", "模糊", "错误", "不完整", "无关", "太简单", "太难"]
        
        positive_count = sum(1 for keyword in positive_keywords if keyword in response)
        negative_count = sum(1 for keyword in negative_keywords if keyword in response)
        
        # 简单计算分数
        base_score = 0.6
        if positive_count > negative_count:
            score = min(1.0, base_score + (positive_count - negative_count) * 0.1)
        else:
            score = max(0.2, base_score - (negative_count - positive_count) * 0.1)
        
        return {
            "ai_score": score,
            "detailed_scores": {criterion: score for criterion in self.quality_criteria.keys()},
            "feedback": f"基于文本分析的评估结果 (正面:{positive_count}, 负面:{negative_count})"
        }
    
    def _combine_assessments(self, basic_checks: Dict[str, Any], 
                           ai_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """合并基础检查和AI评估结果"""
        basic_score = basic_checks.get("basic_score", 0.0)
        ai_score = ai_assessment.get("ai_score", 0.0)
        
        # 加权合并 (基础检查30%, AI评估70%)
        overall_score = basic_score * 0.3 + ai_score * 0.7
        
        # 如果基础检查未通过，大幅降低总分
        if not basic_checks["passed"]:
            overall_score = min(overall_score, 0.4)
        
        # 判断是否通过
        passed = overall_score >= 0.6 and basic_checks["passed"]
        
        # 合并问题和建议
        issues = basic_checks.get("issues", [])
        if ai_assessment.get("weaknesses"):
            issues.extend(ai_assessment["weaknesses"])
        
        return {
            "overall_score": round(overall_score, 3),
            "detailed_scores": ai_assessment.get("detailed_scores", {}),
            "basic_score": basic_score,
            "ai_score": ai_score,
            "passed": passed,
            "issues": issues,
            "feedback": ai_assessment.get("feedback", ""),
            "strengths": ai_assessment.get("strengths", []),
            "assessment_method": "combined"
        }
    
    def get_quality_statistics(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取质量统计信息"""
        if not qa_pairs:
            return {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}
        
        total = len(qa_pairs)
        passed = sum(1 for qa in qa_pairs if qa.get("quality_assessment", {}).get("passed", False))
        failed = total - passed
        pass_rate = passed / total if total > 0 else 0.0
        
        # 分数分布统计
        scores = [qa.get("quality_assessment", {}).get("overall_score", 0) for qa in qa_pairs]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # 按类型统计
        type_stats = {}
        for qa in qa_pairs:
            q_type = qa.get("question_type", "unknown")
            if q_type not in type_stats:
                type_stats[q_type] = {"total": 0, "passed": 0}
            type_stats[q_type]["total"] += 1
            if qa.get("quality_assessment", {}).get("passed", False):
                type_stats[q_type]["passed"] += 1
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(pass_rate, 3),
            "average_score": round(avg_score, 3),
            "score_distribution": {
                "excellent": sum(1 for s in scores if s >= 0.9),
                "good": sum(1 for s in scores if 0.7 <= s < 0.9),
                "fair": sum(1 for s in scores if 0.5 <= s < 0.7),
                "poor": sum(1 for s in scores if s < 0.5)
            },
            "by_type": type_stats
        }