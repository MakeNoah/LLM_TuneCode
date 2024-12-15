import yaml
import json
import re
from pathlib import Path
from tqdm import tqdm
from unsloth import FastLanguageModel
from peft import PeftModel
import torch
import os

# 環境変数を設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    def quantize_model(model, bit_width=14):
        """モデルのパラメータを指定されたビット幅に量子化"""
        scale_factor = 2 ** bit_width - 1
        print(f"Quantizing model to {bit_width}-bit representation...")

        for name, param in tqdm(model.named_parameters(), desc="Quantizing parameters"):
            if param.requires_grad:  # 学習可能パラメータのみ量子化
                # スケーリングとクリッピング
                min_val, max_val = param.data.min(), param.data.max()
                param.data = torch.round((param.data - min_val) / (max_val - min_val) * scale_factor)
                # データ型を保持しつつ値をスケール
                param.data = param.data * (max_val - min_val) / scale_factor + min_val

        print(f"Model quantized to {bit_width}-bit successfully.")
        return model

    @staticmethod
    def create_model(config: Config):
        """設定に基づいてモデルを作成する"""
        model_id = config.get("model.model_id")
        adapter_id = config.get("model.adapter_id")
        hf_token = config.get("hf_token")
        bit_width = config.get("model.bit_width", 14)  # デフォルトは14-bit
        save_path = config.get("model.save_path", "./saved_model")

        # モデルのロード
        print("Loading base model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            dtype=torch.float32,  # フル精度でロード
            trust_remote_code=False,
            device_map="auto",  # 自動でデバイスを割り当て
        )
        print("Base model loaded.")

        # LoRAアダプタの統合
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, adapter_id, token=hf_token)
        print("LoRA adapter loaded.")

        # アダプタを統合して元のモデルにマージ
        print("Merging LoRA adapter into base model...")
        model.merge_and_unload()
        print("LoRA adapter merged successfully.")

        # モデルを指定されたビット幅で量子化
        model = ModelFactory.quantize_model(model, bit_width=bit_width)

        # GPUに移動（量子化後のモデルはすべてGPUに乗る）
        model = model.to("cuda")
        print("Quantized model moved to GPU.")

        # 統合されたモデルを保存
        print("Saving the unified model...")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Unified model saved to {save_path}")

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
        inputs = tokenizer([prompt], return_tensors="pt")
        
        # 入力テンソルをデバイスに転送し、型を修正
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        inputs["input_ids"] = inputs["input_ids"].long()  # input_idsを明示的にlong型に変換

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
