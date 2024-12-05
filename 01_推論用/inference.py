import yaml
import json
import re
from pathlib import Path
from tqdm import tqdm
from unsloth import FastLanguageModel
from peft import PeftModel
import torch

class Config:
    """設定情報を読み込むクラス"""
    def __init__(self, config_file: str):
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key: str, default=None):
        """設定情報を取得する"""
        keys = key.split(".")
        value = self.config
        for k in keys:
            value = value.get(k, default)
        return value

class ModelFactory:
    """モデルのロードを担当するファクトリークラス"""
    @staticmethod
    def create_model(config: Config):
        """設定に基づいてモデルを作成する"""
        model_id = config.get("model.model_id")
        adapter_id = config.get("model.adapter_id")
        dtype = config.get("model.dtype")
        load_in_4bit = config.get("model.load_in_4bit")
        hf_token = config.get("hf_token")

        # モデルのロード
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            trust_remote_code=True,
        )

        # LoRAアダプタの統合
        model = PeftModel.from_pretrained(model, adapter_id, token=hf_token)
        return model, tokenizer

class DataHandler:
    """データの読み込みと書き込みを担当するクラス"""
    @staticmethod
    def load_data(file_path: str):
        """JSONLファイルからデータを読み込む"""
        datasets = []
        with open(file_path, "r") as f:
            item = ""
            for line in f:
                line = line.strip()
                item += line
                if item.endswith("}"):
                    datasets.append(json.loads(item))
                    item = ""
        return datasets

    @staticmethod
    def save_results(results, output_file):
        """推論結果をJSONL形式で保存する"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')

def generate_responses(config_file: str):
    """
    推論処理のエントリーポイント

    Args:
        config_file (str): 設定ファイルのパス
    """
    # 設定の読み込み
    config = Config(config_file)

    # モデルの作成
    model, tokenizer = ModelFactory.create_model(config)

    # データの読み込み
    input_file = config.get("data.input_file")
    datasets = DataHandler.load_data(input_file)

    # モデルを推論モードに変更
    FastLanguageModel.for_inference(model)

    results = []
    for dt in tqdm(datasets):
        input_text = dt["input"]
        task_id = dt["task_id"]

        # プロンプト作成
        prompt = f"""### 指示\n{input_text}\n### 回答\n"""
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

        # 推論処理
        generation_args = {
            "max_new_tokens": config.get("generation.max_new_tokens"),
            "use_cache": config.get("generation.use_cache"),
            "do_sample": config.get("generation.do_sample"),
            "repetition_penalty": config.get("generation.repetition_penalty"),
        }
        outputs = model.generate(**inputs, **generation_args)

        # 出力処理
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).split('\n### 回答')[-1]
        results.append({"task_id": task_id, "input": input_text, "output": prediction})

    # 結果を保存
    adapter_id = re.sub(".*/", "", config.get("model.adapter_id", "default"))
    output_file = config.get("data.output_file_template").format(adapter_id=adapter_id)
    DataHandler.save_results(results, output_file)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # `config.yaml` を指定して推論を実行
    CONFIG_FILE = "config.yaml"
    generate_responses(CONFIG_FILE)
