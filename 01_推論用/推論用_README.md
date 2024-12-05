# 推論用コード: ELYZA-tasks-100-TV

本リポジトリは、Hugging FaceにアップロードされたqLoRAアダプタを用いて、ELYZA-tasks-100-TVの推論を行うためのコードを提供します。  
このコードはGoogle Colabでの動作を想定しており、環境構築や実行に必要な手順について記載しています。

---

## **目次**
1. [環境構築](#環境構築)
2. [設定方法](#設定方法)
3. [使用方法](#使用方法)

---

## **環境構築**

以下のコマンドを順に実行して、必要なライブラリをインストールしてください。

```
# ライブラリのインストール
pip install unsloth
pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -U torch
pip install -U peft
```

> **注意**: 
> - Colab環境を想定しています。
> - `pip install` コマンドは、適宜ローカル環境に応じて調整してください。

---

## **設定方法**

1. **設定ファイルの編集**
   - 推論に必要な情報を `config.yaml` に記載します。
   - 以下は `config.yaml` の例です。

```yaml
model:
  model_id: "llm-jp/llm-jp-3-13b"
  adapter_id: ""  # Hugging FaceにアップロードされたLoRAアダプタID
  load_in_4bit: true
  dtype: null  # 自動設定

hf_token: ""  # Hugging Face Token を指定

data:
  input_file: "./elyza-tasks-100-TV_0.jsonl"
  output_file_template: "/content/{adapter_id}_output.jsonl"

generation:
  max_new_tokens: 512
  use_cache: true
  do_sample: false
  repetition_penalty: 1.2
```

2. **Hugging Face Tokenの取得**
   - Hugging FaceのアカウントからAPIトークンを取得します。
   - [トークン取得ページ](https://huggingface.co/settings/tokens) にアクセスし、トークンをコピーして `hf_token` に設定します。

---

## **使用方法**

1. **コードの実行**
   - メインスクリプトは `inference.py` です。
   - 以下のコマンドで推論を実行します。

```
python inference.py
```

2. **結果の確認**
   - 推論結果は、`config.yaml` に記載したパスに保存されます。
   - 例: `/content/{adapter_id}_output.jsonl`

---

## **ファイル構成**

