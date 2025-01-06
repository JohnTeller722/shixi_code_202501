from typing import TypedDict, List, Dict, Optional
from dataclasses import dataclass
import json
from src.tools.prompts import load_prompt

class LLMConfig(TypedDict):
    """LLM配置参数"""
    temperature: float
    top_p: float
    model: str

class Message(TypedDict):
    """对话消息格式"""
    role: str
    content: str

@dataclass
class MediatorOutput:
    """Mediator输出格式"""
    messages: List[Message]
    config: LLMConfig

class LLMGenerator:
    """大模型生成器"""
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        
    def generate(self, messages: List[Message], config: LLMConfig) -> str:
        """
        调用LLM API生成内容
        """
        from src.tools.llms import LLM
        llm = LLM(base_url=self.base_url, api_key=self.api_key)
        response = llm.llm_chat(config["model"], messages)
        return response

class Mediator0:
    """构造生成问题的提示词"""
    def __init__(self):
        self.comprehensive_prompt = load_prompt("data/test_data_QA_split/system_question/comprehensive_1.txt")
        self.detailed_prompt = load_prompt("data/test_data_QA_split/system_question/detailed_1.txt")
        # 加载 prefix_suffix 配置
        with open("data/test_data_QA_split/prefix_suffix.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.prefix = data.get("prefix", [])
            self.suffix = data.get("suffix", [])
        
    def process(self, context: str) -> List[MediatorOutput]:
        """
        处理context,返回用于生成问题的配置
        每种问题类型都生成markdown和非markdown两个版本
        """
        configs = []
        prompts = [
            ("comprehensive", self.comprehensive_prompt),
            ("detailed", self.detailed_prompt)
        ]
        
        for prompt_type, prompt in prompts:
            # 基础消息配置
            base_messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": context}
            ]
            
            # 基础配置
            base_config = {
                "temperature": 0.7,
                "top_p": 0.9,
                "model": "Qwen2.5-72B"
            }
            
            # 为每个suffix生成一个配置
            for suffix in self.suffix:
                # 复制基础消息和配置
                messages = base_messages.copy()
                config = base_config.copy()
                
                # 在system prompt中添加suffix信息
                messages[0]["content"] = f"{prompt}\n请以{suffix}格式输出。"
                
                # 添加到配置列表
                configs.append(MediatorOutput(messages=messages, config=config))
        
        return configs

class Mediator1:
    """构造生成答案的提示词"""
    def __init__(self):
        self.answer_prompt_markdown = load_prompt("data/test_data_QA_split/system_answer/version_markdown_1.txt")
        self.answer_prompt_text = load_prompt("data/test_data_QA_split/system_answer/version_text_1.txt")
        
    def process(self, context: str, question: str) -> MediatorOutput:
        """
        处理context和question,返回用于生成答案的配置
        """
        # 根据问题类型选择提示词
        prompt = self.answer_prompt_markdown if "markdown" in question else self.answer_prompt_text
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"文本: {context}\n问题: {question}"}
        ]
        
        config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "model": "Qwen2.5-72B"
        }
        
        return MediatorOutput(messages=messages, config=config) 