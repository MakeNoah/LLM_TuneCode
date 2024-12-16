import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft.peft_model import PeftModel
from peft import PeftModel, PeftConfig
import torch

def save_and_upload(model:PeftModel, tokenizer, config):
    """
    トレーニング済みモデルをHugging Face Hubにアップロードする。
    """
    RESULT_PATH =os.path.join(config["model"]["new_model_path"],config["model"]["new_model_id"])
    model.save_pretrained(RESULT_PATH)
    tokenizer.save_pretrained(RESULT_PATH)
    model.push_to_hub(config["model"]["new_model_id"], use_auth_token=config["huggingface"]["hf_token"], private=config["huggingface"]["private"])
    tokenizer.push_to_hub(config["model"]["new_model_id"], use_auth_token=config["huggingface"]["hf_token"], private=config["huggingface"]["private"])

#上でセーブしたモデルをロードして再Push
def push_from_save(config):
    """
    保存済みのアダプタモデルを再ロードしてHugging Face Hubにアップロード。
    """
    RESULT_PATH = os.path.join(config["model"]["new_model_path"], config["model"]["new_model_id"])
    
    # アダプタ設定をロード
    peft_config = PeftConfig.from_pretrained(RESULT_PATH)

    # fullでは乗らないので量子化設定を呼ぶ
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=config["quantization"]["quant_type"],
        bnb_4bit_compute_dtype=torch.bfloat16 if config["quantization"]["compute_dtype"] == "bfloat16" else torch.float16
    )
    # base_modelをロード
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model"]["base_model_id"],
        quantization_config=bnb_config,
        device_map="auto"
    )
    # アダプタモデルをベースモデルにロード
    model = PeftModel.from_pretrained(base_model, RESULT_PATH)
    
    # トークナイザのロード
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    
    # 再度Hugging Face Hubにアップロード
    model.push_to_hub(config["model"]["new_model_id"], 
                        use_auth_token=config["huggingface"]["hf_token"], 
                        private=config["huggingface"]["private"])
    tokenizer.push_to_hub(config["model"]["new_model_id"], 
                            use_auth_token=config["huggingface"]["hf_token"], 
                            private=config["huggingface"]["private"])