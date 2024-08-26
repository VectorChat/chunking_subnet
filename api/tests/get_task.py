import argparse
import json
import os
import time
from sr25519 import sign
import requests
import bittensor as bt

if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument(
        "--hotkey",
        type=str,
        help="The hotkey of the validator.",
    )           
    
    nonce = time.time_ns()
    
    args = argparser.parse_args()
    
    
    
    data = {
        "hotkey_address": args.hotkey,
        "nonce": nonce
    }
    
    owner = bt.wallet("owner")
    
    hotkey = bt.wallet.get_hotkey(owner)
    
    hotkey = args.hotkey
    
    # sign request with validator hotkey
    request_signature = sign(
        (hotkey.public_key, hotkey.private_key),
        str.encode(json.dumps(data))
        ).hex()
    
    request_data = {
        "data": data, 
        "signature": request_signature
    }

    host = "127.0.0.1"
    port = 8080
    
    print(f"Request body: {data}") 
    
    response = requests.post(url=f"http://{host}:{port}/task_api/get_new_task/", json=data)
    
    print(response.json())
