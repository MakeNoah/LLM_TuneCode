from trl import SFTTrainer
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
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_epochs"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        learning_rate=config["learning_rate"],
        fp16=config["fp16"],
        bf16=is_bfloat16_supported(),
        output_dir=config["output_dir"],
        report_to="none",
    )

    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        max_seq_length=512,
        dataset_text_field="formatted_text",
        packing=False,
        args=training_args,
    )
