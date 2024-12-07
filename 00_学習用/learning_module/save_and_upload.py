def save_and_upload(model, tokenizer, config):
    """
    トレーニング済みモデルをHugging Face Hubにアップロードする。
    """
    model.save_pretrained(config["model"]["new_model_id"])
    tokenizer.save_pretrained(config["model"]["new_model_id"])
    model.push_to_hub(config["model"]["new_model_id"], use_auth_token=config["huggingface"]["hf_token"], private=config["huggingface"]["private"])
    tokenizer.push_to_hub(config["model"]["new_model_id"], use_auth_token=config["huggingface"]["hf_token"], private=config["huggingface"]["private"])
