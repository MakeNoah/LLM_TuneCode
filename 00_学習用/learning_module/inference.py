from tqdm import tqdm
import torch
import json 
def infer_and_save(model, tokenizer, eval_dataset, output_path):
    """
    評価データセットに対して推論を実行し、結果を保存する。
    """
    results = []
    for data in tqdm(eval_dataset):
        input_text = data["input"]
        prompt = f"### 指示\n{input_text}\n### 回答\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(inputs["input_ids"], max_new_tokens=100, repetition_penalty=1.2)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({"task_id": data["task_id"], "input": input_text, "output": output_text})

    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')