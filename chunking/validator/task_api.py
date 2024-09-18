import time
from typing import Optional, List, Tuple
from venv import logger
import bittensor as bt
import tiktoken
from chunking.protocol import chunkSynapse
import requests
import numpy as np
from sr25519 import sign
import json
import os
from random import choice, choices
from math import ceil

from neurons.validator import Validator


class Task:
    """
    A class to represent a task that the validator can assign to a miner.
    """

    def __init__(
        self,
        synapse: chunkSynapse,
        task_type: str,
        task_id: int,
        miner_uids: Optional[List[int]] = None,
    ):
        self.synapse = synapse
        self.task_type = task_type
        self.task_id = task_id
        self.miner_uids = miner_uids

    @classmethod
    def get_new_task(self, validator: Validator):
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
                response = requests.post(
                    url=task_url, headers=headers, json=request_data
                )
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
                        bt.logging.debug(
                            f"Received organic query with task id: {task_id}"
                        )
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
                    -1,
                )
            except Exception as e:
                bt.logging.error(
                    f"Failed to get task from API host: '{API_host}'. Exited with exception\n{e}"
                )
        bt.logging.debug("Generating synthetic query")
        synapse, page = generate_synthetic_synapse(validator)
        return Task(synapse=synapse, task_type="synthetic", task_id=-1), page

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


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """
    Helper function to calculate the number of tokens in a string.

    Args:
        string (str): The string to calculate the number of tokens for.
        encoding_name (str): The name of the encoding to use.

    Returns:
        int: The number of tokens in the string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


SYSTEM_PROMPT = "You are a writer tasked with writing an article that combines multiple topics. You are known for your long-winded tangents and detailed exploration of all topics covered in your articles."


def get_wiki_content_for_page(pageid: int) -> str:
    """
    Get the content for a Wikipedia page by the page ID.

    Args:
        pageid (int): The ID of the Wikipedia page to get the content for.

    Returns:
        str: The content of the Wikipedia page.
    """
    return requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "pageids": pageid,
            "prop": "extracts",
            "explaintext": True,
            "exsectionformat": "plain",
        },
    ).json()["query"]["pages"][str(pageid)]["extract"]


def generate_doc_with_llm(validator, pageids=None, timeout=20) -> str:
    pages = (
        choices(validator.articles, k=3)
        if pageids == None or len(pageids) < 3
        else pageids
    )
    source_articles = []
    article_names = []
    for page in pages:
        contents, name = get_wiki_content_for_page(page)
        source_articles.append(contents)
        article_names.append(name)
    
    bt.logging.debug(f"source pageids: {pages}")
    
    bt.logging.info("Generating first section of synthetic query")
    start = time.time()
    
    synthetic_document = validator.client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,    
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
        ).choices[0].message.content

    synthetic_document = ' '.join(synthetic_document.split())
    previous_synthesis = synthetic_document

    bt.logging.info("Generating rest of synthetic query")

    for j in range(5):
        next_synthesis = validator.client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"This is part of an article about {article_names[0]}, {article_names[1]}, and {article_names[2]}:\n{previous_synthesis}\nContinue the article. Do not include section titles. Write to your token limit.",}
            ]).choices[0].message.content
        next_synthesis = ' '.join(next_synthesis.split())
        synthetic_document += ' ' + next_synthesis
        previous_synthesis = next_synthesis

    num_chars = len(synthetic_document)
    
    bt.logging.info(f"Generated synthetic query with {num_chars} characters")  
    
    num_tokens = num_tokens_from_string(synthetic_document, "o200k_base")
    
    bt.logging.info(f"Generated synthetic query with {num_tokens} tokens")  
       
    bt.logging.info(f"Took {time.time() - start} seconds to generate synthetic query")
    return synthetic_document


def generate_doc_normal(validator: Validator | None, pageid=None) -> Tuple[str, int]:
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
    while len(content) < 10000 or len(content) > 100000:
        page = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "random",
                "rnnamespace": 0,
                "format": "json",
            },
        ).json()["query"]["random"][0]["id"]

        content = get_wiki_content_for_page(page)
    return content, page


def generate_synthetic_synapse(validator, timeout=20) -> Tuple[chunkSynapse, int]:

    document = generate_doc_with_llm(validator)
    timeout = validator.config.neuron.timeout if validator is not None else timeout
    time_soft_max = timeout * 0.75
    chunk_size = 4096
    chunk_qty = ceil(ceil(len(document) / chunk_size) * 1.5)
    synapse = chunkSynapse(
        document=document,
        time_soft_max=time_soft_max,
        chunk_size=chunk_size,
        chunk_qty=chunk_qty,
        timeout=timeout,
    )
    return synapse, -1
