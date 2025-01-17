import json
import os
from datetime import datetime
from src.tools.llms import LLM
from src.tools.prompts import load_prompt
from typing import List, Tuple, Dict
import logging
from src.gen_QA_workflow.components import LLMGenerator, Mediator0, Mediator1

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

def workflow(context: str, base_url: str, api_key: str) -> List[Dict]:
    """
    执行完整的QA生成工作流
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(timestamp)
    
    logging.info("Starting Q&A generation workflow")
    
    # 初始化组件
    generator = LLMGenerator(base_url=base_url, api_key=api_key)
    mediator0 = Mediator0()
    mediator1 = Mediator1()
    
    # 生成问题
    questions = []
    mediator0_outputs = mediator0.process(context)
    
    for i, output in enumerate(mediator0_outputs):
        response = generator.generate(output.messages, output.config)
        save_intermediate_output(response, f"questions_{i}_{timestamp}.txt")
        extracted_questions = extract_question(response)
        questions.extend(extracted_questions)
        logging.info(f"Generated {len(extracted_questions)} questions for format {i+1}")
    
    logging.info(f"Total generated questions: {len(questions)}")
    
    # 生成答案
    qa_pairs = []
    for question in questions:
        mediator1_output = mediator1.process(context, question)
        response = generator.generate(mediator1_output.messages, mediator1_output.config)
        
        save_intermediate_output(
            response,
            f"answer_{timestamp}.txt"
        )
        
        answer, reference = extract_answer(response)
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

def prompt_template(context: str, question: str) -> str:
    """Format prompt template with context and question"""
    return f"""
    文本: {context}
    问题: {question}
    """

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

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize LLM
    llm = LLM(base_url="http://127.0.0.1:1025/v1", api_key="sk-1234567890")
    
    # Example usage
    context = load_prompt("data/test_data_QA_split/user_examples/user.txt")
    qa_pairs = workflow(context, llm.base_url, llm.api_key)
    
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