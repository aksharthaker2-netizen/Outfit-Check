"""
llm_phraser.py — rephrases rule-based Recommendation objects into more
natural language, using Groq's free-tier API (openai/gpt-oss-120b).

Design constraint (critical): the LLM is given ONLY the structured facts
already computed by rule_engine.py — never the raw image, never free rein.
The system prompt explicitly forbids introducing any claim not present in
the input. This keeps generation grounded in verified pipeline output,
rather than letting a language model free-associate over fashion advice
it has no actual basis for.
"""

import os
from dotenv import load_dotenv
from groq import Groq

from ml.src.recommendations.rule_engine import Recommendation

load_dotenv()

MODEL_NAME = "openai/gpt-oss-120b"

SYSTEM_PROMPT = (
    "You rephrase structured outfit-styling recommendations into natural, "
    "friendly language for an app. Rules:\n"
    "1. Only rephrase the exact facts given — never add advice, claims, or "
    "reasoning not present in the input.\n"
    "2. Do not mention garment fit, cut, fabric, or brand — no information "
    "about those exists in the input.\n"
    "3. Keep it to 1-2 short sentences per recommendation.\n"
    "4. Be encouraging, not critical — this is styling guidance, not a "
    "judgment of the person."
    "5. Do not add emphasis, certainty, or magnitude words (e.g. 'big difference', "
    "'definitely', 'dramatically') beyond what the input states."
)


def phrase_recommendations(recommendations: list[Recommendation]) -> str:
    """
    Takes the rule engine's structured Recommendation list and returns a
    single natural-language paragraph. Falls back to the raw rule-based
    messages, concatenated, if the API call fails.
    """
    if not recommendations:
        return "This outfit works well as-is."

    structured_input = "\n".join(
        f"- [{r.category.value}] {r.message}" for r in recommendations
    )

    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=300,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Rephrase these outfit recommendations naturally:\n\n"
                        f"{structured_input}"
                    ),
                },
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM phrasing failed ({e}), falling back to rule-based text.")
        return " ".join(r.message for r in recommendations)