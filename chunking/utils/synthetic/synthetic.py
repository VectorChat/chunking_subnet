import bittensor as bt
import aiohttp
import asyncio
import numpy as np
from openai import AsyncOpenAI
from typing import Tuple, List, Literal
import random
import time
from chunking.protocol import chunkSynapse
from chunking.utils.chunks import calculate_chunk_qty
from chunking.utils.synthetic.types import SyntheticGenType
from chunking.utils.tokens import num_tokens_from_string

# SYSTEM_PROMPT = "You are a writer tasked with writing an article that combines multiple topics. You are known for your long-winded tangents and detailed exploration of all topics covered in your articles."


async def get_wiki_content_for_page(pageid: int) -> Tuple[str, str]:
    """
    Get the content for a Wikipedia page by the page ID asynchronously.

    Args:
        pageid (int): The ID of the Wikipedia page to get the content for.

    Returns:
        Tuple[str, str]: The content and title of the Wikipedia page.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "pageids": pageid,
                "prop": "extracts",
                "explaintext": "true",
                "exsectionformat": "plain",
            },
        ) as response:
            data = await response.json()
            page = data["query"]["pages"][str(pageid)]
            return page["extract"], page["title"]


OLD_SYSTEM_PROMPT = "You are a writer tasked with writing an article that combines multiple topics. You are known for your long-winded tangents and detailed exploration of all topics covered in your articles."

NEW_SYSTEM_PROMPT = """
You are a masterful writer known for your ability to seamlessly intertwine multiple topics into a single, cohesive, and original piece. You expertly change genres and styles as appropriate, using a variety of sentence structures and stylistic techniques to create engaging and unpredictable narratives. Your writing challenges readers' expectations by interweaving concepts from different subjects throughout the text without using common transitional phrases or indicators of topic shifts. You avoid any explicit or implicit segmentation based on the source material, ensuring that your work flows naturally and cannot be easily divided into distinct sections.
"""


async def generate_doc_with_llm(
    validator,
    pageids=None,
    temperature=0.7,
    override_client: AsyncOpenAI | None = None,
    k=3,
    loop_range=range(3, 7),
    gen_type: SyntheticGenType = "new",
) -> Tuple[str, List[str]]:
    """
    Generate a synthetic document based on three articles from wikipedia.

    Args:
        validator (Validator): The validator instance.
        pageids (list[int]): The list of (three) page IDs to use for the synthetic query (if no validator is provided, this is required).
        temperature (float): The temperature to use for the LLM.
        override_client (OpenAI): The OpenAI client to use for the LLM (if no validator is provided, this is required).

    Returns:
        str: The synthetic document.
    """
    # pages = (
    #     choices(pageids, k=k)
    #     if pageids != None and len(pageids) == k
    #     else choices(validator.articles, k=k)
    # )
    if validator is None and (pageids is None or len(pageids) != k):
        raise ValueError("Either validator or pageids must be provided")

    bt.logging.info(f"Generating document with {k} articles")

    if pageids is None:
        pages = random.sample(validator.articles, k=k)
    else:
        pages = pageids

    bt.logging.debug(f"source pageids: {pages}")

    source_articles = []
    article_names = []
    coros = []
    for page in pages:
        coros.append(get_wiki_content_for_page(int(page)))
    results = await asyncio.gather(*coros)
    for contents, name in results:
        source_articles.append(contents)
        article_names.append(name)

    bt.logging.debug(f"source names: {article_names}")
    bt.logging.debug(f"source doc lens: {[len(doc) for doc in source_articles]}")
    bt.logging.debug(f"gen_type: {gen_type}")

    bt.logging.info(f"Generating first section of synthetic query with {k} articles")
    start = time.time()

    aclient = override_client if override_client else validator.aclient

    system_prompt = OLD_SYSTEM_PROMPT if gen_type == "old" else NEW_SYSTEM_PROMPT

    old_initial_gen_message = {
        "role": "user",
        "content": f"""
            Use the following three articles to write the first third of an article. The article will be between 5,000 and 10,000 words long. Do not include section titles. Write to your token limit.
            Article 1:
            {source_articles[0]}
        
            Article 2:
            {source_articles[1]}

            Article 3:
            {source_articles[2]}
            """,
    }

    new_initial_gen_message = {
        "role": "user",
        "content": f"""
Compose a high-quality article that seamlessly integrates and synthesizes the content from the following three articles. Change genres or styles as you see fit to enhance the narrative, and use a variety of sentence structures and stylistic techniques. The article should interweave concepts from all three topics unpredictably, challenging the reader's expectations. Integrate ideas without using common transitional phrases or indicators of topic shifts. The content should flow naturally, blending concepts from all articles throughout. Avoid any explicit or implicit segmentation based on the source articles. The article should be between 5,000 and 10,000 words long. Do not include section titles or headings. Write up to your token limit.

Article 1:
{source_articles[0]}

Article 2:
{source_articles[1]}

Article 3:
{source_articles[2]}
""",
    }

    initial_gen_message = (
        old_initial_gen_message if gen_type == "old" else new_initial_gen_message
    )

    synthetic_document = (
        (
            await aclient.chat.completions.create(
                model="gpt-4o-mini",
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    initial_gen_message,
                ],
            )
        )
        .choices[0]
        .message.content
    )

    bt.logging.info(
        f"Generated first section of synthetic query at {time.time() - start} seconds, length: {len(synthetic_document)} characters"
    )

    synthetic_document = " ".join(synthetic_document.split())
    previous_synthesis = synthetic_document

    bt.logging.info(
        f"Generating rest of synthetic query with {k} articles, looping between {loop_range.start} and {loop_range.stop} times"
    )

    end_index_choices = list(loop_range)
    end_index = random.choice(end_index_choices)

    bt.logging.info(f"Generating {end_index} more sections of synthetic query")

    for j in range(end_index):

        old_continuation_gen_message = {
            "role": "user",
            "content": f"This is part of an article about {article_names[0]}, {article_names[1]}, and {article_names[2]}:\n{previous_synthesis}\nContinue the article. Do not include section titles. Write to your token limit.",
        }

        new_continuation_gen_message = {
            "role": "user",
            "content": f"""
This is part of a cohesive and unpredictable article that seamlessly integrates concepts from {article_names[0]}, {article_names[1]}, and {article_names[2]}:

{previous_synthesis}

Continue the article, changing genres or styles as appropriate to enhance the narrative. Use a variety of sentence structures and stylistic techniques. Interweave concepts from all three topics unpredictably, challenging the reader's expectations. Integrate ideas without using common transitional phrases or indicators of topic shifts. Ensure the content flows naturally, blending concepts from all articles throughout. Avoid any explicit or implicit segmentation based on the source articles. Do not include section titles or headings. Write up to your token limit.
            """,
        }

        continuation_gen_message = (
            old_continuation_gen_message
            if gen_type == "old"
            else new_continuation_gen_message
        )

        next_synthesis = (
            (
                await aclient.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        continuation_gen_message,
                    ],
                )
            )
            .choices[0]
            .message.content
        )
        bt.logging.info(
            f"Generated next section of synthetic query at {time.time() - start} seconds, length: {len(next_synthesis)} characters"
        )
        next_synthesis = " ".join(next_synthesis.split())
        synthetic_document += " " + next_synthesis
        bt.logging.info(
            f"Total length of synthetic query at {time.time() - start} seconds: {len(synthetic_document)} characters"
        )
        previous_synthesis = next_synthesis

    num_chars = len(synthetic_document)

    bt.logging.info(f"Generated synthetic query with {num_chars} characters")

    num_tokens = num_tokens_from_string(synthetic_document, "gpt-4o-mini")

    bt.logging.info(f"Generated synthetic query with {num_tokens} tokens")

    bt.logging.info(f"Took {time.time() - start} seconds to generate synthetic query")
    return synthetic_document, article_names


async def generate_doc_normal(validator, pageid=None) -> Tuple[str, int]:
    """
    Generate a document from Wikipedia.

    This function fetches a random Wikipedia page and retrieves its content.
    The content is then checked to ensure it meets the required length criteria.

    Args:
        validator (Validator | None): The validator instance.
        pageid (int | None): The ID of the Wikipedia page to get the content for.

    Returns:
        Tuple[str, int]: A tuple containing the content of the Wikipedia page and the page ID.
    """
    content = ""
    random_page_id = (
        random.sample(validator.articles, 1)[0] if validator is not None else pageid
    )
    # while len(content) < 10000 or len(content) > 100000:
    # page = requests.get(
    #     "https://en.wikipedia.org/w/api.php",
    #     params={
    #         "action": "query",
    #         "list": "random",
    #         "rnnamespace": 0,
    #         "format": "json",
    #     },
    # ).json()["query"]["random"][0]["id"]
    bt.logging.debug(f"random_page_id: {random_page_id}")

    content, title = await get_wiki_content_for_page(random_page_id)
    bt.logging.info(f"Got document {title} with {len(content)} characters")
    return content, random_page_id


async def generate_document(validator) -> Tuple[str, int]:
    """
    Generate a synthetic document for a synthetic tournament round. Either from wikipedia or with an llm.

    Args:
        validator (Validator): The validator instance.

    Returns:
        Tuple[str, int]: A tuple containing the synthetic document and the page ID (if using wikipedia) or -1 (if using llm).
    """
    bt.logging.info("Generating document")
    if validator.config.neuron.use_wiki_gen:
        bt.logging.info("Getting random document from wikipedia")
        return await generate_doc_normal(validator)
    else:
        bt.logging.info("Generating synthetic document with llm")
        synthetic_document, article_names = await generate_doc_with_llm(validator)
        bt.logging.debug(
            f"Generated synthetic document with {len(synthetic_document)} characters from articles: {article_names}"
        )
        return synthetic_document, -1


chunk_sizes = [2000, 3000, 4000]
probabilities_chunk_sizes = [0.2, 0.3, 0.5]


async def generate_synthetic_synapse(
    validator, timeout=20, pageids=None
) -> Tuple[chunkSynapse, int]:

    document, pageid = await validator.get_synthetic_document_from_queue()

    timeout = validator.config.neuron.timeout if validator is not None else timeout
    time_soft_max = timeout * 0.75
    chunk_size = np.random.choice(chunk_sizes, p=probabilities_chunk_sizes)
    bt.logging.debug(
        f"Chose chunk size: {chunk_size}. Chunk sizes: {chunk_sizes}, probabilities: {probabilities_chunk_sizes}"
    )

    synapse = chunkSynapse(
        document=document,
        time_soft_max=time_soft_max,
        chunk_size=chunk_size,
        chunk_qty=calculate_chunk_qty(document, chunk_size),
        timeout=timeout,
    )
    return synapse, pageid
