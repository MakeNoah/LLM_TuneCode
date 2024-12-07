import json

def spilt_jsonl(input_file, output_file, split_string="### 回答\n"):
    """
    JSONLファイルを読み込み、指定文字列を基準に右側の文だけを採用して出力する。

    Args:
        input_file (str): 入力JSONLファイルのパス。
        output_file (str): 処理後のJSONLファイルの保存先パス。
        split_string (str): 分割基準の文字列。
    """
    processed_data = []

    # JSONLファイルを読み込み
    with open(input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            # 各行をJSONとして読み込み
            data = json.loads(line.strip())
            if "output" in data:
                # 分割して右側の文だけを採用
                split_parts = data["output"].split(split_string, 1)
                if len(split_parts) > 1:
                    data["output"] = split_parts[1].strip()
                else:
                    pass # 分割できない場合何もしない
            processed_data.append(data)

    # 処理後のデータをJSONL形式で保存
    with open(output_file, "w", encoding="utf-8") as outfile:
        for entry in processed_data:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write("\n")

if __name__ == "__main__":
    # 使用例
    spilt_jsonl("./evaluation_results.jsonl", "fixed_valuation_results.jsonl.jsonl")
