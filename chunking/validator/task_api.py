import random
import time
from typing import Literal, Optional, List, Tuple
from venv import logger
import aiohttp
import bittensor as bt
from openai import AsyncOpenAI, OpenAI
import tiktoken
from chunking.protocol import chunkSynapse, chunkSynapseType
import requests
import numpy as np
from sr25519 import sign
import json
import os
from random import choice, choices
from math import ceil

from chunking.utils.chunks import calculate_chunk_qty
from chunking.utils.tokens import num_tokens_from_string
from chunking.validator.types import TaskType
from chunking.utils.relay.relay import make_relay_payload
from neurons.validator import Validator


class Task:
    """
    A class to represent a task that the validator can assign to a miner.
    """

    def __init__(
        self,
        synapse: chunkSynapse,
        task_type: TaskType,
        task_id: int,
        page_id: int = -1,
        miner_uids: Optional[List[int]] = None,
    ):
        self.synapse = synapse
        self.task_type = task_type
        self.task_id = task_id
        self.miner_uids = miner_uids
        self.page_id = page_id

    @classmethod
    def get_organic_task(cls, validator: Validator) -> Optional["Task"]:
        """
        Get an organic task from the API host.

        Args:
            validator (Validator): The validator instance requesting the task.

        Returns:
            Task: The organic task
        """
        hotkey = validator.wallet.get_hotkey()
        nonce = time.time_ns()
        data = {"hotkey_address": hotkey.ss58_address, "nonce": nonce}

        bt.logging.debug(
            f"Requesting task from API host: '{os.environ['CHUNKING_API_HOST']}'"
        )
        bt.logging.debug(f"Request body: {data}")

        # sign request with validator hotkey
        request_signature = sign(
            (hotkey.public_key, hotkey.private_key), str.encode(json.dumps(data))
        ).hex()

        API_host = os.environ["CHUNKING_API_HOST"]
        task_url = f"{API_host}/task_api/get_new_task/"
        headers = {"Content-Type": "application/json"}
        request_data = {"data": data, "signature": request_signature}
        bt.logging.debug(f"Request data: {request_data}")

        try:
            response = requests.post(url=task_url, headers=headers, json=request_data)
            if response.status_code == 502:
                raise Exception(f"API Host: '{API_host}' is down")
            elif response.status_code == 403:
                raise Exception(response.text())
            elif response.status_code != 200:
                raise Exception(
                    f"Post to API failed with status code: {response.status_code}"
                )
            else:
                task = response.json()
                if task["task_id"] != -1:
                    task_id = task["task_id"]
                    miner_uids = task.get("miner_uids")
                    bt.logging.debug(f"Received organic query with task id: {task_id}")
                    if task.get("time_soft_max") == None:
                        task["time_soft_max"] = 5.0
                    if task.get("chunk_size") == None:
                        task["chunk_size"] = 4096
                    if task.get("chunk_qty") == None:
                        task["chunk_qty"] = ceil(
                            ceil(len(task["document"]) / task["chunk_size"]) * 1.5
                        )
                    bt.logging.debug(f"task: {task}")

                    synapse = chunkSynapse(
                        document=task["document"],
                        time_soft_max=float(task["time_soft_max"]),
                        chunk_size=int(task["chunk_size"]),
                        chunk_qty=int(task["chunk_qty"]),
                    )
                else:
                    bt.logging.info(
                        f"No organic task available. Generating synthetic query"
                    )
                    raise Exception("No organic task available")
                return (
                    Task(
                        synapse=synapse,
                        task_type="organic",
                        task_id=task_id,
                        miner_uids=miner_uids,
                    ),
                )

        except Exception as e:
            bt.logging.error(
                f"Failed to get task from API host: '{API_host}'. Exited with exception\n{e}"
            )
            return None

    @classmethod
    async def get_synthetic_task(cls, validator: Validator) -> "Task":
        """
        Get a synthetic task from the API host.

        Args:
            validator (Validator): The validator instance requesting the task.

        Returns:
            Task: The synthetic task
        """
        bt.logging.debug("Generating synthetic query")
        synapse, page = await generate_synthetic_synapse(validator)
        return Task(synapse=synapse, task_type="synthetic", task_id=-1, page_id=page)

    @classmethod
    async def get_new_task(self, validator: Validator) -> "Task":
        """
        Get a new task based on the validator's config.

        This function returns either an organic or synthetic task.

        If the validator is configured to allow organic tasks, it will request an organic task from the API host.
        If no organic task is available, it will generate a synthetic task.

        If the validator is not configured to allow organic tasks, it will generate a synthetic task regardless.

        Args:
            validator (Validator): The validator instance requesting the task.

        Returns:
            Tuple[Task, int]: A tuple containing the task and the page ID.
        """
        if os.environ.get("ALLOW_ORGANIC_CHUNKING_QUERIES") == "True":
            task = self.get_organic_task(validator)
        else:
            task = None

        if task is None:
            task = await self.get_synthetic_task(validator)

        bt.logging.debug(f"Created task: {task}")

        try:
            # Make relay payload and pin to IPFS for both organic and synthetic queries
            CID = await make_relay_payload(
                task.synapse.document, validator.aclient, validator.wallet
            )
        except Exception as e:
            bt.logging.error(f"Failed to pin document to IPFS: {e}")
            CID = None

        task.synapse.CID = CID

        bt.logging.debug(f"Added CID: {CID} to task")

        return task

    @classmethod
    def return_response(cls, validator, response_data):
        """
        Return a miner response with chunks to the API host.

        Args:
            validator (Validator): The validator instance returning the response.
            response_data (dict): The data to be returned to the API host.
        """
        validator_hotkey = validator.wallet.get_hotkey()
        validator_sig = sign(
            (validator_hotkey.public_key, validator_hotkey.private_key),
            str.encode(json.dumps(response_data)),
        ).hex()
        API_host = os.environ["CHUNKING_API_HOST"]
        task_url = f"{API_host}/task_api/organic_response/"
        headers = {"Content-Type": "application/json"}
        data = {
            "response_data": response_data,
            "validator_signature": validator_sig,
        }
        try:
            response = requests.post(task_url, headers=headers, json=data)
        except Exception as e:
            bt.logging.error(
                f"Failed to return response to API host: '{API_host}'. Exited with exception\n{e}"
            )

    @classmethod
    def upload_logs(cls, validator, log_data):
        """
        Upload logs to the API host.

        Args:
            validator (Validator): The validator instance uploading the logs.
            log_data (dict): The data to be uploaded to the API host.
        """
        hotkey = validator.wallet.get_hotkey()
        signature = sign(
            (hotkey.public_key, hotkey.private_key), str.encode(json.dumps(log_data))
        ).hex()

        API_host = os.environ["CHUNKING_API_HOST"]
        task_url = f"{API_host}/task_api/log/"
        headers = {"Content-Type": "application/json"}
        data = {
            "log_data": log_data,
            "signature": signature,
        }
        try:
            response = requests.post(task_url, headers=headers, json=data)
            bt.logging.debug(f"upload_logs: response: {response.status_code}")

            if response.status_code == 502:
                raise Exception(f"API Host: '{API_host}' is down")
            elif response.status_code == 403:
                raise Exception(response.text())
            elif response.status_code != 200:
                raise Exception(
                    f"Post to API failed with status code: {response.status_code}"
                )
            else:
                bt.logging.debug(
                    f"Successfully uploaded logs to API host: '{API_host}'"
                )

        except Exception as e:
            bt.logging.error(
                f"Failed to upload logs to API host: '{API_host}'. Exited with exception\n{e}"
            )


SYSTEM_PROMPT = "You are a writer tasked with writing an article that combines multiple topics. You are known for your long-winded tangents and detailed exploration of all topics covered in your articles."


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


async def generate_doc_with_llm(
    validator: Validator,
    pageids=None,
    temperature=0.7,
    override_client: AsyncOpenAI | None = None,
) -> Tuple[str, int]:
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
    pages = (
        choices(pageids, k=3)
        if pageids != None and len(pageids) == 3
        else choices(validator.articles, k=3)
    )
    source_articles = []
    article_names = []
    for page in pages:
        contents, name = await get_wiki_content_for_page(page)
        source_articles.append(contents)
        article_names.append(name)

    bt.logging.debug(f"source pageids: {pages}")

    bt.logging.info("Generating first section of synthetic query")
    start = time.time()

    aclient = override_client if override_client else validator.aclient

    synthetic_document = (
        (await aclient.chat.completions.create(
            model="gpt-4o-mini",
            temperature=temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
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
                },
            ],
        ))
        .choices[0]
        .message.content
    )

    bt.logging.info(
        f"Generated first section of synthetic query at {time.time() - start} seconds, length: {len(synthetic_document)} characters"
    )

    synthetic_document = " ".join(synthetic_document.split())
    previous_synthesis = synthetic_document

    bt.logging.info("Generating rest of synthetic query")

    end_index_choices = list(range(3, 7))

    end_index = choice(end_index_choices)

    bt.logging.info(f"Generating {end_index} more sections of synthetic query")

    for j in range(end_index):
        next_synthesis = (
            (await aclient.chat.completions.create(
                model="gpt-4o-mini",
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"This is part of an article about {article_names[0]}, {article_names[1]}, and {article_names[2]}:\n{previous_synthesis}\nContinue the article. Do not include section titles. Write to your token limit.",
                    },
                ],
            ))
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
    return synthetic_document, -1


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
    random_page_id = random.sample(validator.articles, 1)[0]
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


async def generate_synthetic_synapse(
    validator, timeout=20, pageids=None
) -> Tuple[chunkSynapse, int]:

    bt.logging.info("Generating synthetic query with llm")
    if validator.config.neuron.use_wiki_gen:
        document, pageid = await generate_doc_normal(validator)
    else:
        document, pageid = await generate_doc_with_llm(validator)
    timeout = validator.config.neuron.timeout if validator is not None else timeout
    time_soft_max = timeout * 0.75
    chunk_size = 4096
    synapse = chunkSynapse(
        document=document,
        time_soft_max=time_soft_max,
        chunk_size=chunk_size,
        chunk_qty=calculate_chunk_qty(document, chunk_size),
        timeout=timeout,
    )
    return synapse, pageid
