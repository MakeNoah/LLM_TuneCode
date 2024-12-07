from datasets import load_dataset

def prepare_dataset(config):
    """
    データセットをロードし、トレーニングと評価用のデータセットを準備する。

    Args:
        config (dict): 設定ファイルの辞書。

    Returns:
        dict: トレーニングと評価データセットを含む辞書。
    """
    # トレーニングデータセットのロード
    dataset = load_dataset("json", data_files=config["dataset"]["train_file"])

    # プロンプトフォーマットの定義
    prompt_format = config["dataset"]["prompt_format"]
    eos_token = config.get("eos_token", "</s>")  # デフォルトでEOSトークンを設定

    def formatting_prompts_func(examples):
        """
        データセットの各サンプルをプロンプト形式に整形する。
        """
        input_text = examples["text"]
        output_text = examples["output"]
        formatted_text = prompt_format.format(input_text, output_text) + eos_token
        return {"formatted_text": formatted_text}

    # トレーニングデータセットの整形
    dataset = dataset.map(
        formatting_prompts_func,
        num_proc=4,  # 並列処理
    )

    # 評価データセットのロード
    eval_dataset = []
    eval_file = config["dataset"]["eval_file"]

    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            eval_dataset.append(eval(line.strip()))

    return {"train": dataset["train"], "eval": eval_dataset}
