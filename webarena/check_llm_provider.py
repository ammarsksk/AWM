"""Check the configured OpenAI-compatible LLM provider.

Examples:
  python3 webarena/check_llm_provider.py --model google/gemini-2.5-pro
  python3 webarena/check_llm_provider.py --model gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import json

from openai import OpenAI

from provider_config import get_openai_compatible_kwargs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    kwargs = get_openai_compatible_kwargs()
    safe_kwargs = {
        "api_key_present": bool(kwargs.get("api_key")),
        "base_url": kwargs.get("base_url"),
    }
    print("Provider config:")
    print(json.dumps(safe_kwargs, indent=2))

    client = OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "Return exactly: OK"}],
        temperature=0,
        max_tokens=16,
    )
    print("Model response:")
    print(response.choices[0].message.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
