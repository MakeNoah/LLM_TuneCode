import yaml
from learning_module.dataset import prepare_dataset
from learning_module.model_setup import initialize_model
from learning_module.trainer import initialize_trainer
from learning_module.inference import infer_and_save
from learning_module.save_and_upload import save_and_upload,push_from_save
import os

# GPUメモリ設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def load_config(config_path="config.yaml"):
    """
    YAML設定ファイルを読み込む関数。

    Args:
        config_path (str): 設定ファイルのパス。

    Returns:
        dict: 読み込まれた設定。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    """
    メインエントリーポイント。以下を実行：
    1. モデルとデータセットの準備
    2. トレーニング
    3. 推論と結果保存
    4. モデル保存とHugging Faceへのアップロード
    """
    # 設定ファイルの読み込み
    config = load_config()

    print("Step 1: モデルとデータセットの準備...")
    model, tokenizer = initialize_model(config)
    dataset = prepare_dataset(config)

    print("Step 2: トレーニングの開始...")
    trainer = initialize_trainer(config, model, tokenizer, dataset)
    if config["use_checkpoint"]:
        trainer.train(resume_from_checkpoint=config["checkpoint"])
    else:
        trainer.train()

    print("Step 3: 推論と結果保存...")
    infer_and_save(model, tokenizer, dataset["eval"], "evaluation_results.jsonl")

    print("Step 4: モデル保存とHugging Faceへのアップロード...")
    save_and_upload(model, tokenizer, config)

    print("すべての処理が完了しました。")

# tokenミスったとき
def _savePush():
    config = load_config()
    push_from_save(config)

if __name__ == "__main__":
    main()
    #_savePush()
