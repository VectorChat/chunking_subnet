import logging
from math import ceil
from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import time
import json
import numpy as np
from scalecodec.utils.ss58 import ss58_decode
import argparse
import asyncio
from typing import List, Optional

from api.helper import get_helper, init_helper

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = FastAPI()

router = APIRouter(prefix="/task_api")

# Input and Output models (unchanged)
class TaskRequest(BaseModel):
    signature: str
    data: dict = Field(..., example={
        "hotkey_address": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        "nonce": 1630000000000000000
    })

class TaskResponse(BaseModel):
    task_id: int
    chunk_size: int
    chunk_qty: int
    document: str

class OrganicResponse(BaseModel):
    validator_signature: str
    response_data: dict = Field(..., example={
        "nonce": 1630000000000000000,
        "validator_hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        "miner_signature": "0x...",
        "miner_hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        "task_id": 1,
        "task_data": {
            "document": "...",
            "chunk_size": 4096,
            "chunk_qty": 10,
            "chunks": ["..."]
        }
    })

class OrganicResponseResult(BaseModel):
    status: str

class ChunkLanguageRequest(BaseModel):
    document: str
    time_soft_max: float = 15.0
    chunk_size: int = 4096
    chunk_qty: Optional[int] = None

class ChunkLanguageResponse(BaseModel):
    chunks: List[str]

@router.post("/get_new_task", response_model=TaskResponse)
async def send_task(request: TaskRequest):
    logger.info(f"Received new task request: {request}")
    signature = request.signature
    request_data = request.data
    hotkey = request_data["hotkey_address"]
    public_key = ss58_decode(hotkey)
    nonce = request_data["nonce"]

    helper = get_helper()
    logger.debug(f"Helper instance retrieved: {helper}")

    if hotkey not in helper.whitelist:
        logger.warning(f"Hotkey {hotkey} not in whitelist")
        raise HTTPException(status_code=403, detail=f"Hotkey {hotkey} not whitelisted")    

    logger.info(f"Verifying signature for hotkey: {hotkey}")
    try:
        helper.verify_signature(signature, json.dumps(request_data), public_key)
        logger.info(f"Signature verified for hotkey: {hotkey}")
    except Exception as e:
        logger.error(f"Signature verification failed for hotkey {hotkey}: {str(e)}")
        raise HTTPException(status_code=403, detail="Signature verification failed")
    
    logger.info(f"Verifying nonce for hotkey: {hotkey}")
    try:
        helper.verify_nonce(hotkey, nonce)
        logger.info(f"Nonce verified for hotkey: {hotkey}")
    except Exception as e:
        logger.error(f"Nonce verification failed for hotkey {hotkey}: {str(e)}")
        raise HTTPException(status_code=403, detail="Nonce verification failed")

    if len(helper.task_queue) == 0:
        logger.info("No tasks available in the queue")
        return TaskResponse(task_id=-1, chunk_size=0, chunk_qty=0, document="")

    task_id = helper.task_queue.pop(0)
    helper.task_status[task_id] += 1
    task = helper.tasks[task_id]
    task['validator_hotkey'] = hotkey

    logger.info(f"Assigned task {task_id} to hotkey {hotkey}")
    logger.debug(f"Task details: {task}")

    return TaskResponse(
        task_id=task_id,
        chunk_size=task['chunk_size'],
        chunk_qty=task['chunk_qty'],
        document=task['document'],
    )

@router.post("/organic_response", response_model=OrganicResponseResult)
async def receive_response(response: OrganicResponse):
    logger.info("Received organic response")
    logger.debug(f"Response data: {response}")

    validator_signature = response.validator_signature
    response_data = response.response_data
    nonce = response_data['nonce']
    validator_hotkey = response_data['validator_hotkey']
    validator_key = ss58_decode(validator_hotkey)
    miner_signature = response_data['miner_signature']
    miner_hotkey = response_data['miner_hotkey']
    miner_key = ss58_decode(miner_hotkey)
    task_id = response_data["task_id"]
    
    helper = get_helper()
    logger.debug(f"Helper instance retrieved: {helper}")
    
    try:
        task = helper.tasks[task_id]
        logger.info(f"Retrieved task {task_id}")
        logger.debug(f"Task details: {task}")
    except KeyError:
        logger.error(f"Task {task_id} not found")
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    logger.info(f"Verifying task assignment for task {task_id} and validator {validator_hotkey}")
    try:
        helper.verify_task_assignment(task_id, validator_hotkey)
        logger.info("Task assignment verified")
    except Exception as e:
        logger.error(f"Task assignment verification failed: {str(e)}")
        raise HTTPException(status_code=403, detail="Task assignment verification failed")

    logger.info(f"Verifying validator signature for hotkey {validator_hotkey}")
    try:
        helper.verify_signature(validator_signature, json.dumps(response_data), validator_key)
        logger.info("Validator signature verified")
    except Exception as e:
        logger.error(f"Validator signature verification failed: {str(e)}")
        raise HTTPException(status_code=403, detail="Validator signature verification failed")

    logger.info(f"Verifying nonce for validator {validator_hotkey}")
    try:
        helper.verify_nonce(validator_hotkey, nonce)
        logger.info("Nonce verified")
    except Exception as e:
        logger.error(f"Nonce verification failed: {str(e)}")
        raise HTTPException(status_code=403, detail="Nonce verification failed")

    task_data = response_data['task_data']
    logger.info(f"Verifying miner signature for hotkey {miner_hotkey}")
    try:
        helper.verify_signature(miner_signature, json.dumps(task_data), miner_key)
        logger.info("Miner signature verified")
    except Exception as e:
        logger.error(f"Miner signature verification failed: {str(e)}")
        raise HTTPException(status_code=403, detail="Miner signature verification failed")

    logger.info("Verifying task data")
    try:
        helper.verify_task_data(task, task_data)
        logger.info("Task data verified")
    except Exception as e:
        logger.error(f"Task data verification failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Task data verification failed")

    helper.tasks[task_id]['chunks'] = task_data['chunks']
    helper.task_status[task_id] *= -1
    logger.info(f"Updated task {task_id} status and chunks")    

    return OrganicResponseResult(status="success")

@router.post("/add_task", response_model=ChunkLanguageResponse)
async def receive_task(request: ChunkLanguageRequest):
    logger.info("Received chunk language request")
    logger.debug(f"Request data: {request}")

    data = request.dict()
    if data['chunk_qty'] is None:
        data['chunk_qty'] = ceil(ceil(len(data["document"]) / data["chunk_size"]) * 1.5)
        logger.info(f"Calculated chunk_qty: {data['chunk_qty']}")

    helper = get_helper()
    logger.debug(f"Helper instance retrieved: {helper}")

    if len(helper.task_status) != 0 and helper.task_status.min() == -1:
        task_id = helper.task_status.argmin()
        helper.task_status[task_id] += 1
        helper.tasks[task_id] = data
        helper.task_queue.append(int(task_id))
        logger.info(f"Updated existing task {task_id}")
    else:
        task_id = len(helper.task_status)
        helper.task_status = np.append(helper.task_status, int(0))
        helper.tasks.append(data)
        helper.task_queue.append(task_id)
        logger.info(f"Created new task {task_id}")

    logger.debug(f"Task queue: {helper.task_queue}")
    logger.debug(f"Task status: {helper.task_status}")

    start_time = time.time()
    logger.info(f"Waiting for task {task_id} to complete")
    while (time.time() - start_time < data['time_soft_max'] * 4/3 and helper.tasks[task_id].get('chunks') is None):
        await asyncio.sleep(0.25)

    if helper.tasks[task_id].get('chunks') is None:
        logger.warning(f"Task {task_id} timed out")
        raise HTTPException(status_code=403, detail="Request timed out")

    logger.info(f"Task {task_id} completed successfully")
    return ChunkLanguageResponse(chunks=helper.tasks[task_id]['chunks'])

app.include_router(router)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, help="Host for HTTP server", default="0.0.0.0")
    parser.add_argument("--port", type=int, help="Port for HTTP server", default=8080)
    parser.add_argument("--whitelist_hotkeys", type=str, help="Validator hotkey addresses to send queries to. Separated by commas. All registered validators are whitelisted by default.", default=None)
    parser.add_argument("--netuid", type=int, help="Netuid for the metagraph", default=40)
    parser.add_argument("--network", type=str, help="Bittensor network to use ('finney' or 'test' or any websocket endpoint)", default="finney")
    parser.add_argument("--log_level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help="Set the logging level")

    args = parser.parse_args()
    
    # Set log level based on argument
    logging.getLogger().setLevel(args.log_level)
    
    logger.info(f"Using args: {args}")
    
    whitelisted_hotkeys = args.whitelist_hotkeys.split(',') if args.whitelist_hotkeys is not None else None                
    
    init_helper(args.netuid, args.network, whitelisted_hotkeys)
    
    import uvicorn
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)