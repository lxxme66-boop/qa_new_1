"""
示例：如何修改run_semiconductor_qa.py的1.2阶段来处理纯文本QA

这个文件展示了如何将原有的图片相关处理替换为纯文本QA生成
"""

# 在run_semiconductor_qa.py中，找到步骤1.2的位置（大约465行），替换为：

async def modified_phase_1_2(all_tasks, batch_size, config, chunks_dir):
    """
    修改后的1.2阶段：生成纯文本QA而不是提取图片信息
    """
    from run_text_qa import process_text_chunks_for_qa
    
    # 步骤1.2: 文本QA生成（替代原有的AI文本处理）
    logger.info("步骤1.2: 文本QA生成...")
    
    # 使用新的文本QA处理函数
    result_stats = await process_text_chunks_for_qa(
        text_chunks=all_tasks,
        config=config,
        output_dir=os.path.dirname(chunks_dir)
    )
    
    # 读取处理结果（兼容原有格式）
    processed_file = os.path.join(chunks_dir, "ai_processed_texts.json")
    with open(processed_file, 'r', encoding='utf-8') as f:
        processed_results = json.load(f)
    
    logger.info(f"文本QA生成完成: {result_stats['total_qa_pairs']} 个QA对")
    
    return processed_results


# 或者，如果想要更简单的集成，可以直接修改input_text_process的调用：

# 原代码（约第472行）：
# batch_results = await asyncio.gather(
#     *(input_text_process(
#         task["content"], 
#         os.path.basename(task["file_path"]),
#         chunk_index=task["chunk_index"],
#         total_chunks=len([t for t in all_tasks if t["file_path"] == task["file_path"]]),
#         prompt_index=9,  # 原来使用的是图片相关的prompt
#         config=config
#     ) for task in batch),
#     return_exceptions=True
# )

# 修改为：
batch_results = await asyncio.gather(
    *(input_text_process(
        task["content"], 
        os.path.basename(task["file_path"]),
        chunk_index=task["chunk_index"],
        total_chunks=len([t for t in all_tasks if t["file_path"] == task["file_path"]]),
        prompt_index="semiconductor_text_qa",  # 使用新的文本QA prompt
        config=config
    ) for task in batch),
    return_exceptions=True
)


# 后续的步骤也需要相应调整，例如：
# 1. 步骤1.3（质量评估）可能需要调整评估标准，因为现在评估的是QA对而不是文本质量
# 2. 第二阶段可能需要跳过或修改，因为我们已经在1.2阶段生成了QA
# 3. 数据格式可能需要调整以适应新的QA结构

# 完整的修改建议：
"""
1. 在config.json中添加一个开关：
   {
     "processing": {
       "use_text_qa_mode": true,  // 启用纯文本QA模式
       "text_qa_prompt": "semiconductor_text_qa"
     }
   }

2. 在run_semiconductor_qa.py的主函数中添加条件判断：
   if config.get('processing', {}).get('use_text_qa_mode', False):
       # 使用纯文本QA处理流程
       processed_results = await modified_phase_1_2(all_tasks, batch_size, config, chunks_dir)
   else:
       # 使用原有的图片相关处理流程
       # ... 原有代码 ...

3. 调整后续步骤以适应新的数据格式
"""