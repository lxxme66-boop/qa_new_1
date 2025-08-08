#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test script to verify vLLM HTTP functionality"""

import os
import sys
import asyncio
import json
from semiconductor_qa_generator import SemiconductorQAGenerator

async def test_text_quality_evaluation():
    """Test text quality evaluation with a sample text"""
    
    # Set environment variables for vLLM HTTP
    os.environ['USE_VLLM_HTTP'] = 'true'
    os.environ['VLLM_SERVER_URL'] = 'http://localhost:8000/v1'
    
    # Initialize generator
    print("Initializing SemiconductorQAGenerator...")
    try:
        generator = SemiconductorQAGenerator(batch_size=1, gpu_devices="0")
        print("Generator initialized successfully")
    except Exception as e:
        print(f"Failed to initialize generator: {e}")
        return
    
    # Sample text for testing
    sample_text = """
    # OLED Display Technology Research
    
    ## Abstract
    This research investigates the degradation mechanisms of blue phosphorescent OLEDs. 
    Through systematic analysis of device lifetime under various operating conditions, 
    we identified that the primary degradation pathway involves exciton-polaron annihilation 
    at the emission layer/electron transport layer interface.
    
    ## Key Findings
    1. Blue OLED lifetime improved by 45% using a novel graded junction architecture
    2. The degradation rate follows an Arrhenius relationship with activation energy of 0.85 eV
    3. Interface engineering reduced the operating voltage by 0.3V while maintaining luminance
    
    ## Technical Analysis
    The degradation mechanism can be modeled using the following equation:
    L(t) = L0 * exp(-B * t^n)
    
    Where:
    - L(t) is the luminance at time t
    - L0 is the initial luminance
    - B is the degradation constant
    - n is the acceleration factor (typically 0.5-0.7 for OLEDs)
    
    Our experiments show that n = 0.65 for the optimized device structure, 
    indicating a reduction in catastrophic failure modes.
    """
    
    # Prepare data for evaluation
    test_data = [{
        "paper_name": "test_oled_research",
        "md_content": sample_text,
        "source_info": {
            "source_file": "test.txt",
            "chunk_index": 0
        }
    }]
    
    print("\nTesting text quality evaluation...")
    print(f"Text length: {len(sample_text)} characters")
    
    try:
        # Run evaluation
        results = await generator.judge_processed_texts(test_data)
        
        print("\nEvaluation Results:")
        for result in results:
            judgment = result.get('judgment', {})
            print(f"- Suitable for QA: {judgment.get('suitable_for_qa', False)}")
            print(f"- Reason: {judgment.get('reason', 'N/A')}")
            print(f"- Score text: {judgment.get('score_text', 'N/A')[:100]}...")
            
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_text_quality_evaluation())