#!/usr/bin/env python3
"""测试vLLM HTTP服务连接"""

import requests
import json

def test_vllm_http():
    """测试vLLM HTTP服务"""
    base_url = "http://localhost:8000/v1"
    
    # 1. 测试模型列表
    print("1. 测试获取模型列表...")
    try:
        response = requests.get(f"{base_url}/models", timeout=5)
        if response.status_code == 200:
            print(f"✅ 成功获取模型列表: {response.json()}")
        else:
            print(f"❌ 获取模型列表失败: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到vLLM服务器，请确保服务器正在运行")
        print("   请在另一个终端运行: python start_vllm_server.py")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False
    
    # 2. 测试聊天完成
    print("\n2. 测试聊天完成API...")
    try:
        data = {
            "model": "qwen-vllm",
            "messages": [
                {"role": "system", "content": "你是一个AI助手"},
                {"role": "user", "content": "你好"}
            ],
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        response = requests.post(
            f"{base_url}/chat/completions",
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 成功生成响应:")
            print(f"   内容: {result['choices'][0]['message']['content']}")
        else:
            print(f"❌ 生成失败: {response.status_code}")
            print(f"   响应: {response.text}")
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=== vLLM HTTP服务测试 ===")
    print("服务地址: http://localhost:8000/v1")
    print()
    
    success = test_vllm_http()
    
    if success:
        print("\n✅ vLLM HTTP服务正常工作！")
    else:
        print("\n❌ vLLM HTTP服务存在问题，请检查服务器状态")