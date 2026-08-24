"""OpenAI-compatible structured-output classifier (LLM fallback).

Transport only. The prompt, the result type, and the parser live in llm.py.
"""

import json
from collections.abc import Mapping, Sequence

from openai import AsyncOpenAI

from financial_dashboard.services.categorization.fewshot import FewShotExample
from financial_dashboard.services.categorization.llm import (
    LLM_TIMEOUT_MS,
    LlmResult,
    build_prompt,
    parse_result,
)


async def classify(
    *,
    fields: Mapping[str, str | None],
    examples: Sequence[FewShotExample],
    active_slugs: list[str],
    api_key: str,
    model: str,
    base_url: str,
    name_tokens: Sequence[str] = (),
) -> LlmResult:
    prompt = build_prompt(
        fields=fields,
        examples=examples,
        active_slugs=active_slugs,
        name_tokens=name_tokens,
    )
    client = AsyncOpenAI(
        api_key=api_key, base_url=base_url or None, timeout=LLM_TIMEOUT_MS / 1000
    )
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    content = response.choices[0].message.content
    data = json.loads(content or "{}")
    return parse_result(data, active_slugs)
