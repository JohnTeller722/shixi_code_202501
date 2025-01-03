import json
from datetime import datetime
from tools.llms import LLM
from tools.prompts import load_prompt
from typing import List, Tuple, Dict
import logging

def workflow(context: str, llm: LLM) -> List[Dict]:
    """
    Execute the full Q&A generation workflow.
    
    Args:
        context: Input text context
        llm: LLM instance for API calls
        
    Returns:
        List of Q&A pairs with format:
        [{"question": str, "answer": str, "reference": str}, ...]
    """
    # Load prompts
    question_comprehensive_prompt = load_prompt("data/test_data_QA_split/system_question/comprehensive_1.txt")
    question_detailed_prompt = load_prompt("data/test_data_QA_split/system_question/detailed_1.txt") 
    answer_prompt = load_prompt("data/test_data_QA_split/system_answer/version_1.txt")
    
    # Generate comprehensive questions
    comp_response = llm.llm_chat("Qwen2.5-72B", [
        {"role": "system", "content": question_comprehensive_prompt},
        {"role": "user", "content": context}
    ])
    comprehensive_questions = extract_question(comp_response)
    
    # Generate detailed questions
    detail_response = llm.llm_chat("Qwen2.5-72B", [
        {"role": "system", "content": question_detailed_prompt},
        {"role": "user", "content": context}
    ])
    detailed_questions = extract_question(detail_response)
    
    # Combine all questions
    all_questions = comprehensive_questions + detailed_questions
    
    # Generate answers for each question
    qa_pairs = []
    for question in all_questions:
        # Format prompt with context and question
        user_prompt = prompt_template(context, question)
        
        # Get answer from LLM
        answer_response = llm.llm_chat("Qwen2.5-72B", [
            {"role": "system", "content": answer_prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        # Extract answer and reference
        answer, reference = extract_answer(answer_response)
        
        # Add to results
        qa_pairs.append({
            "context": context,
            "question": question,
            "answer": answer,
            "reference": reference
        })
        
    return qa_pairs

def extract_question(text: str) -> List[str]:
    """Extract questions from LLM response"""
    questions = []
    for line in text.split("\n"):
        if "<问题>:" in line:
            question = line.split(":", 1)[1].strip()
            questions.append(question)
    return questions

def extract_answer(text: str) -> Tuple[str, str]:
    """Extract answer and reference from LLM response"""
    # Extract answer part
    answer_start = text.find("<答案>:") + len("<答案>:")
    answer_end = text.find("<原文依据>:")
    answer = text[answer_start:answer_end].strip()
    
    # Extract reference part
    reference_start = text.find("<原文依据>:") + len("<原文依据>:")
    reference = text[reference_start:].strip()
    
    return answer, reference

def prompt_template(context: str, question: str) -> str:
    """Format prompt template with context and question"""
    return f"""
    文本：{context}
    问题：{question}
    """

if __name__ == "__main__":
    # Initialize LLM
    llm = LLM(base_url="http://127.0.0.1:1025/v1", api_key="sk-1234567890")
    
    # Example usage
    context = load_prompt("data/test_data_QA_split/user_examples/user.txt")
    qa_pairs = workflow(context, llm)
    
    # 添加json文件写入逻辑
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"output/qa_pairs_{timestamp}.json"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        print(f"Successfully wrote QA pairs to {output_file}")
    except Exception as e:
        print(f"Error writing to file: {e}")