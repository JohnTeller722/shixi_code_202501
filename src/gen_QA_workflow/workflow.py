import json
import os
from datetime import datetime
from src.tools.llms import LLM
from src.tools.prompts import load_prompt
from typing import List, Tuple, Dict
import logging

def setup_logging(timestamp: str):
    """设置日志配置"""
    # 创建logs目录
    os.makedirs("logs", exist_ok=True)
    
    # 配置logging
    log_file = f"logs/workflow_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def save_intermediate_output(content: str, filename: str):
    """保存中间生成的文本"""
    os.makedirs("intermediate", exist_ok=True)
    with open(f"intermediate/{filename}", 'w', encoding='utf-8') as f:
        f.write(content)

def workflow(context: str, llm: LLM) -> List[Dict]:
    """
    Execute the full Q&A generation workflow.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(timestamp)
    
    logging.info("Starting Q&A generation workflow")
    logging.info(f"Context length: {len(context)} characters")
    
    # Load prompts
    logging.info("Loading prompts...")
    question_comprehensive_prompt = load_prompt("data/test_data_QA_split/system_question/comprehensive_1.txt")
    question_detailed_prompt = load_prompt("data/test_data_QA_split/system_question/detailed_1.txt")
    answer_prompt = load_prompt("data/test_data_QA_split/system_answer/version_1.txt")
    
    # Generate comprehensive questions
    logging.info("Generating comprehensive questions...")
    comp_response = llm.llm_chat("Qwen2.5-72B", [
        {"role": "system", "content": question_comprehensive_prompt},
        {"role": "user", "content": context}
    ])
    save_intermediate_output(comp_response, f"comprehensive_questions_{timestamp}.txt")
    comprehensive_questions = extract_question(comp_response)
    logging.info(f"Generated {len(comprehensive_questions)} comprehensive questions")
    
    # Generate detailed questions
    logging.info("Generating detailed questions...")
    detail_response = llm.llm_chat("Qwen2.5-72B", [
        {"role": "system", "content": question_detailed_prompt},
        {"role": "user", "content": context}
    ])
    save_intermediate_output(detail_response, f"detailed_questions_{timestamp}.txt")
    detailed_questions = extract_question(detail_response)
    logging.info(f"Generated {len(detailed_questions)} detailed questions")
    
    # Combine all questions
    all_questions = comprehensive_questions + detailed_questions
    logging.info(f"Total questions generated: {len(all_questions)}")
    
    # Generate answers for each question
    qa_pairs = []
    logging.info("Generating answers...")
    for i, question in enumerate(all_questions, 1):
        logging.info(f"Processing question {i}/{len(all_questions)}")
        
        # Format prompt with context and question
        user_prompt = prompt_template(context, question)
        
        # Get answer from LLM
        answer_response = llm.llm_chat("Qwen2.5-72B", [
            {"role": "system", "content": answer_prompt},
            {"role": "user", "content": user_prompt}
        ])
        save_intermediate_output(
            answer_response, 
            f"answer_q{i}_{timestamp}.txt"
        )
        
        # Extract answer and reference
        answer, reference = extract_answer(answer_response)
        
        # Add to results
        qa_pairs.append({
            "context": context,
            "question": question,
            "answer": answer,
            "reference": reference
        })
    
    logging.info("Q&A generation workflow completed successfully")
    return qa_pairs

def extract_question(text: str) -> List[str]:
    """Extract questions from LLM response"""
    questions = []
    for line in text.split("\n"):
        if "<问题>:" in line or "<问题>：" in line:
            question = line.split(":", 1)[1].strip()
            questions.append(question)
    return questions

def find_first_valid_position(text: str, markers: List[str]) -> int:
    """查找第一个在文本中出现的标记的位置"""
    positions = [text.find(marker) for marker in markers]
    valid_positions = [pos for pos in positions if pos != -1]
    return min(valid_positions) if valid_positions else -1

def extract_answer(text: str) -> Tuple[str, str]:
    """Extract answer and reference from LLM response"""
    # Define markers
    answer_markers = ["<答案>:", "<答案>："]
    reference_markers = ["<原文依据>:", "<原文依据>："]
    
    # Find positions
    answer_pos = find_first_valid_position(text, answer_markers)
    reference_pos = find_first_valid_position(text, reference_markers)
    
    if answer_pos == -1 or reference_pos == -1:
        logging.warning("Could not find answer or reference markers in response")
        return "", ""
    
    # Extract answer (add length of the actual found marker)
    marker_length = len("<答案>:") if text[answer_pos:].startswith("<答案>:") else len("<答案>：")
    answer_start = answer_pos + marker_length
    answer = text[answer_start:reference_pos].strip()
    
    # Extract reference
    marker_length = len("<原文依据>:") if text[reference_pos:].startswith("<原文依据>:") else len("<原文依据>：")
    reference_start = reference_pos + marker_length
    reference = text[reference_start:].strip()
    
    return answer, reference

def prompt_template(context: str, question: str) -> str:
    """Format prompt template with context and question"""
    return f"""
    文本: {context}
    问题: {question}
    """

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize LLM
    llm = LLM(base_url="http://127.0.0.1:1025/v1", api_key="sk-1234567890")
    
    # Example usage
    context = load_prompt("data/test_data_QA_split/user_examples/user.txt")
    qa_pairs = workflow(context, llm)
    
    # 创建output目录
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 添加json文件写入逻辑
    output_file = os.path.join(output_dir, f"qa_pairs_{timestamp}.json")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        logging.info(f"Successfully wrote QA pairs to {output_file}")
    except Exception as e:
        logging.error(f"Error writing to file: {e}")