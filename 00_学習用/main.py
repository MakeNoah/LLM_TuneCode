import yaml
from dataset import prepare_dataset
from model_factory import ModelFactory
from trainer import initialize_trainer

def load_config(config_path="config.yaml"):
    """YAMLファイルから設定をロードする"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    # 設定のロード
    config = load_config()
    print(config)
    # データセット準備
    dataset = prepare_dataset(config["dataset"])

    # モデルの初期化
    model, tokenizer = ModelFactory.create_model(config["model"])

    # トレーナーの初期化
    trainer = initialize_trainer(config["trainer"], model, tokenizer, dataset)

    # モデルのトレーニング
    trainer.train()
    # batchsize:2だと1分くらいかなあ…283.9948

if __name__ == "__main__":
    main()
