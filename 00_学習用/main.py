import yaml
from dataset import prepare_dataset
from model_setup import initialize_model
from trainer import initialize_trainer
from inference import infer_and_save
from save_and_upload import save_and_upload

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()

    # モデルとデータセットの準備
    model, tokenizer = initialize_model(config)
    dataset = prepare_dataset(config)

    # トレーニング
    trainer = initialize_trainer(config, model, tokenizer, dataset)
    trainer.train()

    # 推論と保存
    infer_and_save(model, tokenizer, dataset["eval"], "evaluation_results.jsonl")

    # 保存とアップロード
    save_and_upload(model, tokenizer, config)

if __name__ == "__main__":
    main()
