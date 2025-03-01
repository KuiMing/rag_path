"""
Extract commentaries from the given url.
"""

import pandas as pd
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from markdownify import markdownify as md
import httpx


class Commentary(BaseModel):
    time: str
    text: str


class Match(BaseModel):
    commentaries: list[Commentary]


def main():
    """
    Extract commentaries from the given passage.
    """

    url = "https://espn.com/soccer/commentary/_/gameId/704489"
    resp = httpx.get(url, follow_redirects=True)
    resp.raise_for_status()
    content = md(resp.text, strip=["a", "img"])

    load_dotenv(find_dotenv())
    client = OpenAI()
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": "Extract commentaries from the given passage. Do not fabricate any information.",
            },
            {"role": "user", "content": f"Passage:{content}"},
        ],
        response_format=Match,
    )

    event = completion.choices[0].message.parsed
    commentaries = pd.DataFrame(event.dict()["commentaries"])
    print(commentaries)


if __name__ == "__main__":
    main()
