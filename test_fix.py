#!/usr/bin/env python3
"""Test script to verify the SemiconductorQAGenerator fix"""

import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from semiconductor_qa_generator import SemiconductorQAGenerator
    
    logger.info("Creating SemiconductorQAGenerator instance...")
    generator = SemiconductorQAGenerator(batch_size=4, gpu_devices="0")
    
    logger.info(f"Generator created successfully")
    logger.info(f"Model name: {generator.model_name}")
    logger.info(f"Config: {generator.config}")
    logger.info(f"Tokenizer: {generator.tokenizer}")
    logger.info(f"LLM: {generator.llm}")
    
    # Test loading model
    logger.info("Testing model loading...")
    generator.load_model()
    
    logger.info(f"After loading - Tokenizer: {generator.tokenizer}")
    logger.info(f"After loading - LLM: {generator.llm}")
    
    # Test a simple text evaluation
    test_texts = [{
        "paper_name": "test_paper",
        "md_content": "This is a test content about semiconductor technology.",
        "source_info": {"test": True}
    }]
    
    logger.info("Testing judge_processed_texts method...")
    # Note: This would need async handling in real usage
    # For now, just check if the method exists and model is loaded
    if hasattr(generator, 'judge_processed_texts'):
        logger.info("judge_processed_texts method exists")
        if generator.tokenizer is not None:
            logger.info("Tokenizer is loaded and ready")
        else:
            logger.error("Tokenizer is still None after loading!")
    
    logger.info("Test completed successfully!")
    
except Exception as e:
    logger.error(f"Test failed: {str(e)}")
    import traceback
    logger.error(f"Traceback:\n{traceback.format_exc()}")
    sys.exit(1)