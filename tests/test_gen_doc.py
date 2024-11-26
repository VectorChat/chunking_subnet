import asyncio
import random

from openai import AsyncOpenAI

from chunking.utils.synthetic import generate_doc_with_llm
from tests.utils.articles import get_articles

async def main():

    articles = get_articles() 

    k = 3

    random_articles = random.sample(articles, 3)

    document, article_names = await generate_doc_with_llm(
        validator=None,
        k=k,
        pageids=random_articles,
        override_client=AsyncOpenAI(),
    )

    print(f"Got document with {len(document)} characters")
    print(article_names)

    assert document is not None
    assert len(document) > 10000
    assert len(article_names) == k


def test_gen_doc():
    asyncio.run(main())
