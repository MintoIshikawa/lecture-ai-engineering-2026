"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
--temperature で出力の多様さを調整できる（省略時はモデルの既定値）。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, temperature: float | None = None) -> str:
    client = genai.Client(api_key=get_api_key())
    config = types.GenerateContentConfig(temperature=temperature) if temperature is not None else None
    response = client.models.generate_content(model=MODEL, contents=question, config=config)

    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage else None
    output_tokens = usage.candidates_token_count if usage else None
    print(f"[tokens] 入力: {input_tokens if input_tokens is not None else '不明'} / 出力: {output_tokens if output_tokens is not None else '不明'}")

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--temperature",
        type=float,
        metavar="VALUE",
        help="生成設定の temperature（省略時はモデルの既定値）",
    )
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    if args.temperature is not None:
        print(f"[temperature] {args.temperature}")
    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question, temperature=args.temperature))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
