from re import T
import time
from typing import Optional, List, Tuple
import bittensor as bt
from chunking.protocol import chunkSynapse
import requests
import numpy as np
from sr25519 import sign
import json
import os
from random import choice
from math import ceil

from neurons.validator import Validator

class Task():
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

        if os.environ.get('ALLOW_ORGANIC_CHUNKING_QUERIES') == 'True':
            hotkey = validator.wallet.get_hotkey()
            nonce = time.time_ns()
            data = {
                'hotkey_address': hotkey.ss58_address,
                'nonce': nonce
            }
            
            bt.logging.debug(f"Requesting task from API host: \'{os.environ['CHUNKING_API_HOST']}\'")
            bt.logging.debug(f"Request body: {data}")

            # sign request with validator hotkey
            request_signature = sign(
                (hotkey.public_key, hotkey.private_key),
                str.encode(json.dumps(data))
                ).hex()

            API_host = os.environ['CHUNKING_API_HOST']
            task_url = f"{API_host}/task_api/get_new_task/"
            headers = {"Content-Type": "application/json"}
            request_data = {
                'data': data, 
                'signature': request_signature
                }
            bt.logging.debug(f"Request data: {request_data}")
            
            try:
                response = requests.post(url=task_url, headers=headers, json=request_data)
                if response.status_code == 502:
                    raise Exception(f"API Host: \'{API_host}\' is down")
                elif response.status_code == 403:
                    raise Exception(response.text())
                elif response.status_code != 200:
                    raise Exception(f"Post to API failed with status code: {response.status_code}")
                else:
                    task = response.json()
                    if task["task_id"] != -1:
                        task_id = task["task_id"]
                        miner_uids = task.get('miner_uids')
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
                        bt.logging.info(f"No organic task available. Generating synthetic query")
                        raise Exception("No organic task available")                
                return Task(synapse=synapse, task_type="organic", task_id=task_id, miner_uids=miner_uids), -1
            except Exception as e:
                bt.logging.error(f"Failed to get task from API host: \'{API_host}\'. Exited with exception\n{e}")
        bt.logging.debug("Generating synthetic query")
        synapse, page = generate_synthetic_synapse(validator)
        return Task(synapse=synapse, task_type="synthetic", task_id=-1), page

    @classmethod
    def return_response(cls, validator, response_data):
        validator_hotkey = validator.wallet.get_hotkey()
        validator_sig = sign(
            (validator_hotkey.public_key, validator_hotkey.private_key),
            str.encode(json.dumps(response_data))
            ).hex()
        API_host = os.environ['CHUNKING_API_HOST']
        task_url = f"{API_host}/task_api/organic_response/"
        headers = {"Content-Type": "application/json"}
        data = {
            'response_data': response_data,
            'validator_signature': validator_sig,
        }
        try:
            response = requests.post(task_url, headers=headers, json=data)
        except Exception as e:
            bt.logging.error(f"Failed to return response to API host: \'{API_host}\'. Exited with exception\n{e}")


    @classmethod
    def upload_logs(cls, validator, log_data):
        hotkey = validator.wallet.get_hotkey()
        signature = sign(
            (hotkey.public_key, hotkey.private_key),
            str.encode(json.dumps(log_data))
            ).hex()
            
        API_host = os.environ['CHUNKING_API_HOST']
        task_url = f"{API_host}/task_api/log/" 
        headers = {"Content-Type": "application/json"}
        data = {
            'log_data': log_data,
            'signature': signature,
        }
        try:
            response = requests.post(task_url, headers=headers, json=data)            
            bt.logging.debug(f"upload_logs: response: {response.status_code}")
            
            if response.status_code == 502:
                raise Exception(f"API Host: \'{API_host}\' is down")
            elif response.status_code == 403:
                raise Exception(response.text())
            elif response.status_code != 200:
                raise Exception(f"Post to API failed with status code: {response.status_code}")
            else:
                bt.logging.debug(f"Successfully uploaded logs to API host: \'{API_host}\'")            
                            
        except Exception as e:
            bt.logging.error(f"Failed to upload logs to API host: \'{API_host}\'. Exited with exception\n{e}")


def generate_synthetic_synapse(validator, pageid = None, timeout = 20) -> Tuple[chunkSynapse, int]:
    page = choice(validator.articles) if pageid == None else pageid
    document = requests.get('https://en.wikipedia.org/w/api.php', params={
        'action': 'query',
        'format': 'json',
        'pageids': page,
        'prop': 'extracts',
        'explaintext': True,
        'exsectionformat': 'plain',
        }).json()['query']['pages'][str(page)]['extract']
    document = document.replace("\n", " ").replace("\t", " ")
    document = ' '.join(document.split())
    timeout = validator.config.neuron.timeout if validator is not None else timeout
    time_soft_max = timeout * 0.75
    chunk_size = 4096
    chunk_qty = ceil(
        ceil(len(document) / chunk_size) * 1.5
    )
    synapse = chunkSynapse(
        document=document,
        time_soft_max=time_soft_max,
        chunk_size=chunk_size,
        chunk_qty=chunk_qty,
        timeout=timeout
    )
    return synapse, page
