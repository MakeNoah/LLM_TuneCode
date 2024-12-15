from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch
import bitsandbytes as bnb
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def debug_print(message, **variables):
    """デバッグ情報をコンソールに出力するユーティリティ関数"""
    print(f"[DEBUG] {message}")
    for var_name, value in variables.items():
        print(f"  - {var_name}: {value}")

import torch

def debug_and_fix_model(model):
    """
    モデル内の問題をデバッグし、データ型の修正を行う。
    """
    print("[DEBUG] Starting model debug and fix...")

    for name, module in model.named_modules():
        if hasattr(module, "forward"):
            original_forward = module.forward

            def wrapped_forward(*args, **kwargs):
                try:
                    return original_forward(*args, **kwargs)
                except RuntimeError as e:
                    if "value cannot be converted to type at::Half" in str(e):
                        print(f"[DEBUG] Detected dtype issue in module: {name}")
                        print(f"  - Error: {str(e)}")
                        # データ型をfloat32に変更
                        new_args = [arg.to(torch.float32) if isinstance(arg, torch.Tensor) and arg.dtype == torch.float16 else arg for arg in args]
                        new_kwargs = {k: (v.to(torch.float32) if isinstance(v, torch.Tensor) and v.dtype == torch.float16 else v) for k, v in kwargs.items()}
                        print(f"[DEBUG] Retrying with float32 dtype in module: {name}")
                        return original_forward(*new_args, **new_kwargs)
                    raise e

            module.forward = wrapped_forward

    print("[DEBUG] Model debug and fix complete.")
    return model

def initialize_model(config):
    """
    モデルとトークナイザーを初期化し、量子化とLoRA設定を適用する。
    必要に応じてデバッグ情報を出力し、自動修正を試みる。
    """
    # BitsAndBytesConfig設定
    debug_print("Initializing BitsAndBytesConfig...", config=config["quantization"])
    compute_dtype = torch.bfloat16 if config["quantization"]["compute_dtype"] == "bfloat16" else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=config["quantization"]["quant_type"],
        bnb_4bit_compute_dtype=compute_dtype
    )
    
    # モデルのロード
    try:
        debug_print("Loading model...", base_model_id=config["model"]["base_model_id"])
        model = AutoModelForCausalLM.from_pretrained(
            config["model"]["base_model_id"],
            quantization_config=bnb_config,
            device_map="auto",
            token=config["huggingface"]["hf_token"],
        )
        model = debug_and_fix_model(model)
        debug_print("Model loaded successfully.")
    except Exception as e:
        debug_print("Error loading model. Attempting fallback.", error=str(e))
        raise e

    # トークナイザーのロード
    try:
        debug_print("Loading tokenizer...", base_model_id=config["model"]["base_model_id"])
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["base_model_id"],
            trust_remote_code=True,
            token=config["huggingface"]["hf_token"],
            padding=True, 
            truncation=True,
        )
        debug_print("Tokenizer loaded successfully.")
    except Exception as e:
        debug_print("Error loading tokenizer. Attempting fallback.", error=str(e))
        raise e

    # LoRA設定
    debug_print("Configuring LoRA...", lora_config=config["model"]["lora"])
    lora_config = LoraConfig(
        r=config["model"]["lora"]["r"],
        lora_alpha=config["model"]["lora"]["alpha"],
        lora_dropout=config["model"]["lora"]["dropout"],
        bias=config["model"]["lora"]["bias"],
        task_type="CAUSAL_LM",
        target_modules=find_all_linear_names(model)
    )

    try:
        model = get_peft_model(model, lora_config)
        debug_print("LoRA model configured successfully.")
    except Exception as e:
        debug_print("Error configuring LoRA. Attempting fallback.", error=str(e))
        raise e

    # 最終的なモデル情報をデバッグ出力
    debug_print("Final model configuration complete.", model_structure=str(model))
    return model, tokenizer


def find_all_linear_names(model):
    """
    モデル内の4bit量子化線形層を探し、LoRA適用可能なモジュール名を取得する。
    """
    debug_print("Finding all linear module names...")
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    debug_print("Linear module names identified.", lora_module_names=lora_module_names)
    return list(lora_module_names)


# サンプル設定
config = {
    "model": {
        "base_model_id": "google/gemma-2-9b",
        "hf_token": "your_hf_token_here",
        "lora": {
            "r": 8,
            "alpha": 32,
            "dropout": 0.1,
            "bias": "none"
        }
    },
    "quantization": {
        "load_in_4bit": True,
        "quant_type": "nf4",
        "compute_dtype": "bfloat16"
    }
}

# 初期化テスト
if __name__ == "__main__":
    model, tokenizer = initialize_model(config)
