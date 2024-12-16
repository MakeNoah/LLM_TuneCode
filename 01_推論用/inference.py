import yaml
import json
import re
from pathlib import Path
from tqdm import tqdm
from unsloth import FastLanguageModel
from peft import PeftModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import bitsandbytes as bnb
import os
from pathlib import Path
from pprint import pprint
from duckduckgo_search import DDGS

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
        load_in_4bit = config.get("model.load_in_4bit", False)  # 量子化の設定
        hf_token = config.get("hf_token")

        print(f"Model ID: {model_id}, Adapter ID: {adapter_id}, Load in 4bit: {load_in_4bit}")

        # モデルのロード時に直接量子化を適用


        if load_in_4bit:
            # 4bit量子化用の設定をbitsandbytesで適用
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=bnb.BitsAndBytesConfig(load_in_4bit=True),
                trust_remote_code=True,
                use_auth_token=hf_token,
            )
        else:
            # 通常の8bit量子化（またはフル精度）
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                trust_remote_code=True,
                use_auth_token=hf_token,
                load_in_8bit=True,  # bitsandbytesによる8bit量子化
            )

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_auth_token=hf_token)

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
        # ファイルの親ディレクトリを取得
        parent_directory = os.path.dirname(output_file)
        
        # 親ディレクトリが存在しない場合、作成する
        if not os.path.exists(parent_directory):
            os.makedirs(parent_directory)
            print(f"親ディレクトリを作成しました: {parent_directory}")
        else:
            pass
            # print(f"親ディレクトリは既に存在しています: {parent_directory}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')



class RAG:
    """
    Retrieval-Augmented Generation (RAG) クラス
    DuckDuckGo での検索機能を簡潔に実装。
    """
    def __init__(self):
        pass

    def duckduckgo_search(self, query, num_results=2):
        """
        DuckDuckGo を使用して検索を実行するメソッド。

        Parameters:
        - query (str): 検索クエリ
        - num_results (int): 取得する最大検索結果数

        Returns:
        - list: 検索結果（タイトルと記事内容 のリスト）
        """
        query = query[:30] # 長すぎるとエラー吐くので適度にカット
        results = DDGS().text(query, max_results=num_results)  # 検索結果を取得
        
        # 結果が取得できなかった場合の処理
        if not results:
            return []
        
        # 検索結果をフォーマット
        formatted_results = [
            {"title": result["title"], "body": result["body"]} for result in results
        ]
        return formatted_results


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

    #RAGの設定
    RAG_Instance = RAG()
    for dt in tqdm(datasets):
        input_text = dt["input"]
        task_id = dt["task_id"]

        # プロンプト作成
        if use_rag:
            doc = RAG_Instance.duckduckgo_search(input_text)
            prompt = f"""以下の指示に対してステップバイステップで考え、回答を生成しましょう。### 指示\n{input_text}\n### 関連文書\n{doc}\n### 回答\n"""
        else:
            prompt = f"""以下の指示に対してステップバイステップで考え、回答を生成しましょう。### 指示\n{input_text}\n### 回答\n"""
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
        output_decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = output_decoded.split('### 回答\n')[1]
        results.append({"task_id": task_id, "input": input_text, "output": prediction})
        # debug用出力
        print("prompt\n")
        pprint(prompt)
        print("output\n")
        pprint(output_decoded)
        #pprint(prediction)

    # 結果を保存
    adapter_id = re.sub(".*/", "", config.get("model.adapter_id", "default"))
    output_file = config.get("data.output_file_template").format(adapter_id=adapter_id)
    DataHandler.save_results(results, output_file)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    use_rag = False
    # `config.yaml` を指定して推論を実行
    CONFIG_FILE = "config.yaml"
    generate_responses(CONFIG_FILE)
