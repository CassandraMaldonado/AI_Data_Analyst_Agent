from __future__ import annotations

import json
import logging
import os
from typing import TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

T = TypeVar("T", bound=BaseModel)


def ask_llm(system_prompt: str, user_prompt: str, model: str = "gpt-4.1-mini") -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content
        return (content or "").strip()
    except Exception:
        logger.exception("ask_llm failed")
        raise


def ask_llm_structured(
    model_cls: type[T],
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4.1-mini",
) -> T:
    schema = model_cls.model_json_schema()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": model_cls.__name__,
                    "schema": schema,
                    "strict": True,
                },
            },
        )
        raw = response.choices[0].message.content or "{}"
        return model_cls.model_validate_json(raw)
    except Exception as e:
        logger.warning(
            "Structured parse failed (%s); retrying with json_object fallback", e
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        return model_cls.model_validate(data)
