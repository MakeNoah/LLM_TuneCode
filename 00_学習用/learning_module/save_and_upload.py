import os

def save_and_upload(model, tokenizer, config):
    """
    トレーニング済みモデルをHugging Face Hubにアップロードする。
    """
    RESULT_PATH =os.path.join(config["model"]["new_model_path"],config["model"]["new_model_id"])
    model.save_pretrained(RESULT_PATH)
    tokenizer.save_pretrained(RESULT_PATH)
    model.push_to_hub(config["model"]["new_model_id"], use_auth_token=config["huggingface"]["hf_token"], private=config["huggingface"]["private"])
    tokenizer.push_to_hub(config["model"]["new_model_id"], use_auth_token=config["huggingface"]["hf_token"], private=config["huggingface"]["private"])
