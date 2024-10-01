import asyncio
from datetime import timedelta, datetime
from typing import Dict
import httpx
from pydantic import BaseModel, ValidationError
import requests
import json
import bittensor as bt

from chunking.validator.relay import RelayPayload


def add_to_ipfs_cluster(file_path: str, api_url="http://localhost:9094"):
    """
    Add content to an IPFS cluster via the localhost IPFS Cluster API.

    Args:
        file_path (str): Path to the file to be added
        api_url (str): URL of the IPFS Cluster API (default: http://localhost:9094)

    Returns:
        CID of the added content
    """
    endpoint = f"{api_url}/add"

    print(f"Adding {file_path} to IPFS Cluster at {api_url}")

    try:
        with open(file_path, "rb") as file:
            files = {"file": file}
            print(f"Files: {files}")
            response = requests.post(endpoint, files=files)
            print(f"Response: {response}")

        if response.status_code == 200:
            result = json.loads(response.text)
            print(f"Result: {result}")
            return result["cid"]
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


async def get_pinned_cids(
    cluster_api_url="http://localhost:9094",
    ipfs_api_url="http://localhost:5001",
    delta=timedelta(minutes=20),
    batch_size=50,
    verbose=False,
):
    """
    Get CIDs that are pinned and have been pinned within the last delta.
    It parses the content of the CIDs as `RelayPayload` and returns a map of CIDs to `RelayPayload`.

    Args:
        cluster_api_url (str): URL of the IPFS Cluster API (default: http://localhost:9094)
        ipfs_api_url (str): URL of the IPFS API (default: http://localhost:5001)
        delta (timedelta): Time delta to filter pins by (default: 20 minutes)

    Returns:
        dict: A dictionary with CIDs as keys and `RelayPayload` as values
    """
    endpoint = f"{cluster_api_url}/pins"

    def _verbose(msg):
        if verbose:
            bt.logging.debug(msg)

    _verbose(f"Getting all CIDs from IPFS Cluster at {cluster_api_url}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint)
            response.raise_for_status()

        pins = response.json()

        cur_datetime = datetime.now()

        def prefilter_pin(pin):
            if pin.get("status") != "pinned":
                return False
            if pin.get("cid") is None:
                return False
            if datetime.fromisoformat(pin.get("created_at")) < cur_datetime - delta:
                return False

            return True

        # filter out pins that are not pinned and are older than delta
        prefiltered_pins = filter(prefilter_pin, pins)

        pin_map: Dict[str, RelayPayload] = {}

        async def handle_pin(pin):
            try:
                _verbose(f"Getting payload for {pin.get('cid')}")
                payload = await get_from_ipfs(pin.get("cid"), ipfs_api_url)
                # pin_map[pin.get("cid")] = payload
                return (pin.get("cid"), payload)
            except ValidationError as e:
                bt.logging.error(
                    f"Error parsing RelayPayload for {pin.get('cid')}: {e}"
                )
                return (pin.get("cid"), None)

        for i in range(0, len(prefiltered_pins), batch_size):
            batch = prefiltered_pins[i : i + batch_size]
            _verbose(
                f"Processing batch {i} of {len(prefiltered_pins)}, {len(batch)} pins, from index {i} to {i+len(batch)}"
            )

            tasks = [handle_pin(pin) for pin in batch]
            results = await asyncio.gather(*tasks)

            for cid, payload in results:
                if payload is None:
                    _verbose(f"Payload for {cid} is None")
                    continue
                pin_map[cid] = payload

        return pin_map
    except requests.exceptions.RequestException as e:
        bt.logging.error(f"An error occurred while fetching CIDs: {str(e)}")
        return None


async def get_from_ipfs(cid: str, api_url="http://localhost:5001", verbose=False):
    """
    Get content from IPFS via the IPFS HTTP API.

    Args:
        cid (str): CID of the content to be retrieved
        api_url (str): URL of the IPFS API (default: http://localhost:5001)

    Returns:
        str: Content of the retrieved CID

    Raises:
        Exception: If the content is not found, an error occurs, or the content is not a valid RelayPayload
    """
    endpoint = f"{api_url}/api/v0/cat"
    params = {"arg": cid}

    if verbose:
        bt.logging.info(f"Getting {cid} from IPFS at {api_url}")

    async with httpx.AsyncClient() as client:
        response = await client.post(endpoint, params=params)
        response.raise_for_status()

    if verbose:
        bt.logging.debug(f"Response: {response}")
        bt.logging.debug(f"Response status: {response.status_code}")
        bt.logging.debug(f"Response content: {response.content}")
    raw_content = response.content.decode("utf-8")

    obj = json.loads(raw_content)

    return RelayPayload(**obj)


# Example usage
if __name__ == "__main__":
    test_str = "hello worlddddd"
    test_file = "test.txt"
    with open(test_file, "w") as f:
        f.write(test_str)

    cid = add_to_ipfs_cluster(test_file)
    if cid:
        print(f"File added successfully. CID: {cid}")

        content = get_from_ipfs(cid)
        if content:
            print(f"Content: {content}")
        else:
            print("Failed to get content from IPFS.")
    else:
        print("Failed to add file to IPFS cluster.")

    all_cids = get_all_cids_and_content()
    if all_cids:
        print("All CIDs and their content:")
        for cid, content in all_cids.items():
            print(f"CID: {cid}")
            print(f"Content: {content}")
            print()
