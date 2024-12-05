from datasets import load_dataset

def prepare_dataset(config):
    """
    データセットをロードし、指定されたプロンプト形式で整形する。

    Args:
        config (dict): データセット関連の設定。

    Returns:
        DatasetDict: 整形されたデータセット。
    """
    print(config)
    dataset = load_dataset("json", data_files=config["path"])

    # プロンプトフォーマット関数
    def format_example(examples):
        input_text = examples["text"]
        output_text = examples["output"]
        formatted_text = config["prompt_format"].format(input_text, output_text)
        return {"formatted_text": formatted_text}

    # データセットを整形
    dataset = dataset.map(format_example, num_proc=4)
    return dataset
