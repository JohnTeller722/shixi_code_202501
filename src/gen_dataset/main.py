# 根据output结果生成数据集
import json
import os

def rag_dataset_gen(contents, rag_file):
    for content in contents:
        question = content["context"] + "\n请根据以上信息回答以下问题：" + content["question"]
        answer = content["answer"]
        with open(rag_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"question": question, "answer": answer}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    output_dir = "output"
    dataset_dir = "dataset"
    output_file = "qa_pairs_20250106_092634.json"
    time_stamp = "_".join(output_file.split("_")[2:4])
    rag_file = os.path.join(dataset_dir, f"rag_dataset_{time_stamp}.jsonl")
    contents = []
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    with open(os.path.join(output_dir, output_file), "r") as f:
        data = json.load(f)
        contents = data

    rag_dataset_gen(contents, rag_file)
    
        
