from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

def initialize_trainer(config, model, tokenizer, dataset):
    """
    トレーナーを初期化する。

    Args:
        config (dict): トレーナー構成設定。
        model: 初期化されたモデル。
        tokenizer: モデルに対応するトークナイザー。
        dataset: トレーニングデータセット。

    Returns:
        SFTTrainer: 初期化されたトレーナー。
    """
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation"],
        learning_rate=config["training"]["learning_rate"],
        num_train_epochs=config["training"]["num_epochs"],
        warmup_steps=config["training"]["warmup_steps"],
        save_steps=config["training"]["save_steps"],
        logging_steps=config["training"]["logging_steps"],
        fp16= not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        group_by_length=config["training"]["group_by_length"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        max_seq_length=config["model"]["max_seq_length"],
        dataset_text_field="formatted_text",
        args=training_args,
    )
    return trainer