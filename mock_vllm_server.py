#!/usr/bin/env python3
"""Mock vLLM HTTP server for testing"""

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockVLLMHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        if self.path == "/v1/models":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {
                "object": "list",
                "data": [{
                    "id": "qwen-vllm",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "system"
                }]
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == "/v1/completions":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            prompt = request_data.get('prompt', '')
            
            # 模拟质量评估响应
            if '评分' in prompt or '质量' in prompt:
                response_text = "【是】该文本质量良好，适合生成问答对。"
            else:
                response_text = "【否】该文本质量不佳。"
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            response = {
                "id": "cmpl-123",
                "object": "text_completion",
                "created": 1234567890,
                "model": "qwen-vllm",
                "choices": [{
                    "text": response_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            }
            self.wfile.write(json.dumps(response).encode())
            
        elif self.path == "/v1/chat/completions":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            messages = request_data.get('messages', [])
            last_message = messages[-1]['content'] if messages else ''
            
            # 模拟质量评估响应
            if '评分' in last_message or '质量' in last_message:
                response_text = "【是】该文本质量良好，适合生成问答对。"
            else:
                response_text = "【否】该文本质量不佳。"
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            response = {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "qwen-vllm",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Override to reduce noise"""
        logger.debug(f"{self.address_string()} - {format % args}")

def run_server(port=8000):
    """Run the mock server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, MockVLLMHandler)
    logger.info(f"Mock vLLM server running on port {port}")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server()