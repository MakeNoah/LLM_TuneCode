from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch
import bitsandbytes as bnb

def initialize_model(config):
    """
    モデルとトークナイザーを初期化し、量子化とLoRA設定を適用する。
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=config["quantization"]["quant_type"],
        bnb_4bit_compute_dtype=torch.bfloat16 if config["quantization"]["compute_dtype"] == "bfloat16" else torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["base_model_id"],
        quantization_config=bnb_config,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model_id"], trust_remote_code=False)

    lora_config = LoraConfig(
        r=config["model"]["lora"]["r"],
        lora_alpha=config["model"]["lora"]["alpha"],
        lora_dropout=config["model"]["lora"]["dropout"],
        bias=config["model"]["lora"]["bias"],
        task_type="CAUSAL_LM",
        target_modules=find_all_linear_names(model)
    )

    model = get_peft_model(model, lora_config)
    return model, tokenizer

def find_all_linear_names(model):
    """
    モデル内の4bit量子化線形層を探します。
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)