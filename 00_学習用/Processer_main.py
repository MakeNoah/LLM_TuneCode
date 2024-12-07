import yaml
from preprocess_module.merge_datasets import merge_datasets
from preprocess_module.add_prefix import prepend_to_outputs
from postprocess_module.output_parser import spilt_jsonl

class DatasetProcessor:
    """
    データセットの前処理を管理するクラス。

    各ステップを柔軟に呼び出して利用可能。
    """

    def __init__(self, config_path="config.yaml"):
        """
        初期化メソッド。

        Args:
            config_path (str): 設定ファイルのパス。
        """
        self.config = self._load_config(config_path)["process_config"]

    def _load_config(self, config_path):
        """
        設定ファイルをロードする。

        Args:
            config_path (str): 設定ファイルのパス。

        Returns:
            dict: ロードした設定内容。
        """
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def merge(self):
        """
        データセットを結合する。
        """
        input_files = self.config["input_filepaths"]
        output_file = self.config["merged_filepath"]
        print(f"データセットを結合しています: {input_files} -> {output_file}")
        merge_datasets(input_files, output_file)

    def add_prefix(self):
        """
        各エントリの`output`フィールドにテキストを追加。
        """
        input_file = self.config["merged_filepath"]
        output_file = self.config["prefixed_filepath"]
        prepend_text = self.config["prepend_text"]
        print(f"出力フィールドにテキストを追加しています: {input_file} -> {output_file}")
        prepend_to_outputs(input_file, output_file, prepend_text)

    def split_output(self):
        """
        指定文字列で`output`を分割し、右側のみを残す。
        """
        input_file = self.config["prefixed_filepath"]
        output_file = self.config["split_processed_filepath"]
        split_string = self.config["split_string"]
        print(f"`output`を分割しています: {input_file} -> {output_file}")
        spilt_jsonl(input_file, output_file, split_string)


def main():
    """
    DatasetProcessorクラスを利用した前処理のエントリーポイント。
    """

processor = DatasetProcessor()
# データセットの結合
processor.merge()
# datasetにプレフィックスを追加
#processor.add_prefix()
# 出力を分割して右側のみを残す
#processor.split_output()

if __name__ == "__main__":
    main()
