import json

# CoTやstep-by-step思考を強制しようと思ったが単純に前につけるだけだと精度が下がってしまった…
def prepend_to_outputs(input_file, output_file, prepend_text="う〜んなるほど…よく考えさせてください。\n"):
    """
    JSON形式のデータセットを処理し、指定テキストを出力フィールドに追加。
    不正なデータがある場合はデバッグ情報を表示。

    Args:
        input_file (str): 入力JSONファイルのパス。
        output_file (str): 処理後のJSONファイルの保存先パス。
        prepend_text (str): 出力フィールドに追加するテキスト。
    """
    try:
        # JSONファイルを読み込み
        with open(input_file, "r", encoding="utf-8") as infile:
            data = json.load(infile)
    except json.JSONDecodeError as e:
        print(f"Error loading JSON file: {e}")
        return

    processed_data = []
    for idx, entry in enumerate(data):
        try:
            # outputフィールドの先頭にテキストを追加
            if "output" in entry:
                entry["output"] = prepend_text + entry["output"]
            processed_data.append(entry)
        except Exception as e:
            # デバッグ情報を表示
            print(f"Error on entry {idx}: {e}")
            print(f"Content: {entry}")

    # 処理後のデータをJSON形式で保存
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(processed_data, outfile, ensure_ascii=False, indent=2)

if __name__ == "__main__":

    # 使用例
    prepend_to_outputs("./dataset/ichikara-instruction-003-001-1.json", "./dataset/addThink_ichikara-instruction-003-001-1.json")
