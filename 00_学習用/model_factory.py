from unsloth import FastLanguageModel

class ModelFactory:
    """
    モデルを生成するファクトリクラス。
    """

    @staticmethod
    def create_model(config):
        """
        モデルとトークナイザーを初期化する。

        Args:
            config (dict): モデル構成設定。

        Returns:
            tuple: モデルとトークナイザー。
        """
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config["id"],
            dtype=config["dtype"],
            load_in_4bit=config["load_in_4bit"],
            trust_remote_code=True,
        )
        # LoRA設定の適用
        model = FastLanguageModel.get_peft_model(
            model,
            r=config["lora"]["r"],
            target_modules=config["lora"]["target_modules"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=config["lora"]["dropout"],
            bias=config["lora"]["bias"],
        )
        return model, tokenizer
