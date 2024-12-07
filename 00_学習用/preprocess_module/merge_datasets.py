import json
import re
import shutil

def merge_datasets(input_files, output_file):
    """
    複数のJSONデータセットを結合し、エラー箇所を自動修正しながら1つのJSONファイルに保存する。

    Args:
        input_files (list): 結合するJSONファイルのパスリスト。
        output_file (str): 保存するJSONファイルのパス。
    """
    all_entries = []

    for json_file in input_files:
        with open(json_file, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content  # オリジナルの内容を保持

        try:
            data = json.loads(content)
            all_entries.extend(data)
            continue  # エラーがない場合は次のファイルへ
        except json.JSONDecodeError:
            print(f"\n{json_file} でエラーが発生しました。エントリごとに処理を試みます。")
            # ファイル全体のパースに失敗した場合、エントリごとに処理を行う
            data = []
            # JSON配列として括弧で囲まれていると仮定して、先頭と末尾の括弧を除去
            entries_str = content.strip()
            if entries_str.startswith('[') and entries_str.endswith(']'):
                entries_str = entries_str[1:-1]
            else:
                print(f"{json_file} はJSON配列形式ではありません。スキップします。")
                continue
            # エントリを分割
            entries = re.split(r'(?<=\}),\s*(?=\{)', entries_str)
            entries_fixed = []  # 修正後のエントリ文字列を保持
            file_modified = False  # ファイルが修正されたかどうか

            for idx, entry_str in enumerate(entries):
                entry_str = entry_str.strip()
                if not entry_str.startswith('{'):
                    entry_str = '{' + entry_str
                if not entry_str.endswith('}'):
                    entry_str = entry_str + '}'
                try:
                    entry = json.loads(entry_str)
                    data.append(entry)
                    entries_fixed.append(entry_str)
                except json.JSONDecodeError as e:
                    print(f"\nエラーが発生したエントリ（インデックス {idx}）を修正します。")
                    print("修正前のエントリ:")
                    print(entry_str)
                    # エントリを修正
                    fixed_entry_str = fix_invalid_escape_sequences(entry_str)
                    try:
                        entry = json.loads(fixed_entry_str)
                        data.append(entry)
                        entries_fixed.append(fixed_entry_str)
                        print("修正後のエントリ:")
                        print(fixed_entry_str)
                        print(f"エントリ（インデックス {idx}）の修正に成功しました。")
                        file_modified = True  # 修正が行われた
                    except json.JSONDecodeError as e2:
                        print(f"エントリ（インデックス {idx}）の修正に失敗しました。エラー: {e2}")
                        # このエントリをスキップ
                        continue

            all_entries.extend(data)

    # 結合データをJSON形式で保存
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(all_entries, outfile, ensure_ascii=False, indent=2)

    print(f"Successfully merged {len(input_files)} files into {output_file}")


def fix_invalid_escape_sequences(content):
    """
    不正なエスケープシーケンスのみを修正します。

    Parameters:
    - content (str): JSONコンテンツの文字列

    Returns:
    - content (str): 修正後のJSONコンテンツの文字列
    """
    def replace_invalid_escapes(match):
        s = match.group(0)
        # 正しいエスケープシーケンスをプレースホルダーに置換
        valid_escapes = {
            '\\"': 'PLACEHOLDER_QUOTE',
            '\\\\': 'PLACEHOLDER_BACKSLASH',
            '\\/': 'PLACEHOLDER_SLASH',
            '\\b': 'PLACEHOLDER_BACKSPACE',
            '\\f': 'PLACEHOLDER_FORMFEED',
            '\\n': 'PLACEHOLDER_NEWLINE',
            '\\r': 'PLACEHOLDER_CARRIAGE_RETURN',
            '\\t': 'PLACEHOLDER_TAB',
            # Unicodeエスケープシーケンス
            '\\u': 'PLACEHOLDER_UNICODE'
        }
        for esc_seq, placeholder in valid_escapes.items():
            s = s.replace(esc_seq, placeholder)
        # 不正なエスケープシーケンスを検出してエスケープ
        s = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)
        # プレースホルダーを元に戻す
        for esc_seq, placeholder in valid_escapes.items():
            s = s.replace(placeholder, esc_seq)
        return s

    # JSON文字列内の文字列リテラルを検出
    content = re.sub(r'("(?:[^"\\]|\\.)*")', replace_invalid_escapes, content)
    return content
