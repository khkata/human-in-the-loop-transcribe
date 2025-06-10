import os
import argparse

from src.config import *

from openai import OpenAI
client = OpenAI(api_key=api_key)

def load_transcription(path: str) -> str:
    """テキストファイルを読み込んで返す"""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def call_gpt4(text: str) -> str:
    """
    入力テキストに対して
      ・表記ゆれ・細かい誤字を修正した完全文
      ・Markdown形式の要約
    を一度に返すプロンプトを送信する
    """
    system_prompt = (
        "あなたは日本語の文章校正と要約の専門家です。"
        "ユーザーから与えられたテキストを以下の形式で出力してください。\n\n"
        "【出力形式】\n"
        "-----\n"
        "## 修正済みテキスト\n"
        "<ここに表記ゆれや誤字を直し、流れを自然に整えた完全な本文を記載>\n\n"
        "## 要約（Markdown）\n"
        "- 箇条書き1\n"
        "- 箇条書き2\n"
        "…\n"
        "-----\n\n"
        "テキスト:\n"
    )

    full_prompt = system_prompt + text

    response = client.responses.create(
        model="gpt-4.1",
        input=full_prompt
    )

    return response.output_text.strip()

def main(args):
    in_path = f"{args.out_dir}/merged_transcriptions.txt"
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"{in_path} が見つかりません")

    raw = load_transcription(in_path)
    result = call_gpt4(raw)
    print(result)

    out_path = f"{args.out_dir}/merged_transcription_with_LLM.txt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(result)
    print(f"GPT-4.1 の出力をファイルに保存しました: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM-Review-Summary')
    parser.add_argument('--out_dir', default=OUTPUT_DIR)
    args = parser.parse_args()
    main(args)