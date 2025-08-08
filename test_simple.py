#!/usr/bin/env python3
"""简单测试脚本"""
import os

# 模拟问题场景
print("=== 测试场景 ===")

# 场景1：环境变量未设置时
print("\n场景1：环境变量未设置")
use_vllm_http = os.environ.get('USE_VLLM_HTTP', 'false').lower() == 'true'
print(f"USE_VLLM_HTTP环境变量: {os.environ.get('USE_VLLM_HTTP')}")
print(f"use_vllm_http值: {use_vllm_http}")

# 场景2：设置环境变量后
print("\n场景2：设置环境变量后")
os.environ['USE_VLLM_HTTP'] = 'true'
use_vllm_http = os.environ.get('USE_VLLM_HTTP', 'false').lower() == 'true'
print(f"USE_VLLM_HTTP环境变量: {os.environ.get('USE_VLLM_HTTP')}")
print(f"use_vllm_http值: {use_vllm_http}")

# 模拟LLM对象
print("\n=== 模拟LLM对象 ===")

class MockGenerator:
    def __init__(self):
        self.use_vllm_http = os.environ.get('USE_VLLM_HTTP', 'false').lower() == 'true'
        self.llm = None
        self.local_model_manager = None
        
        if self.use_vllm_http:
            self.llm = "vllm_http"
            self.local_model_manager = "MockLocalModelManager"
            print("初始化为HTTP模式")
        else:
            print("初始化为本地模式")
            
    def _generate(self, prompts):
        print(f"\n_generate调用:")
        print(f"  self.llm = {self.llm}")
        print(f"  self.local_model_manager = {self.local_model_manager}")
        print(f"  self.use_vllm_http = {self.use_vllm_http}")
        
        if self.llm == "vllm_http" and self.local_model_manager:
            print("  -> 使用HTTP模式生成")
            return ["HTTP生成结果"]
        elif self.llm is not None:
            print("  -> 使用本地模式生成")
            # 这里会调用 self.llm.generate()，如果llm是None会报错
            return ["本地生成结果"]
        else:
            print("  -> 错误：LLM未初始化")
            return ["错误：模型未初始化"]

# 测试不同场景
print("\n测试1：环境变量未设置时创建生成器")
os.environ.pop('USE_VLLM_HTTP', None)
gen1 = MockGenerator()
result1 = gen1._generate(["测试"])
print(f"结果: {result1}")

print("\n测试2：环境变量已设置时创建生成器")
os.environ['USE_VLLM_HTTP'] = 'true'
gen2 = MockGenerator()
result2 = gen2._generate(["测试"])
print(f"结果: {result2}")