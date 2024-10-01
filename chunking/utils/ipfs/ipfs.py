import asyncio
from datetime import timedelta, datetime
from typing import Dict, List
import httpx
from pydantic import ValidationError
import pytz
import requests
import json
import bittensor as bt
from chunking.utils.ipfs.types import IPFSPin
from chunking.utils.relay.types import RelayPayload


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
    Get CIDs that are pinned and have been pinned within the last delta. Order by most recent first.

    Args:
        cluster_api_url (str): URL of the IPFS Cluster API (default: http://localhost:9094)
        ipfs_api_url (str): URL of the IPFS API (default: http://localhost:5001)
        delta (timedelta): Time delta to filter pins by (default: 20 minutes)

    Returns:
        dict: A dictionary with CIDs as keys and the raw content as values
    """
    endpoint = f"{cluster_api_url}/pins"

    def _verbose(msg):
        if verbose:
            bt.logging.debug(msg)

    _verbose(f"Getting all CIDs from IPFS Cluster at {cluster_api_url}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint)
            print(f"Response: {response}, {response.text}")
            _verbose(f"Response: {response}")
            response.raise_for_status()

        res_text_as_json_array = f"[{",".join(response.text.splitlines())}]"
        pins = json.loads(res_text_as_json_array)

        _verbose(f"{endpoint} response:\n\n{response}\n\n")

        def parse_pin(pin):
            try:
                return IPFSPin(
                    cid=pin.get("cid"),
                    created_at=datetime.fromisoformat(pin.get("created")),
                    status=pin.get("status"),
                )
            except ValidationError as e:
                bt.logging.error(f"Error parsing pin: {e}")
                return None

        pins = list(map(parse_pin, pins))
        pins = list(filter(lambda x: x is not None, pins))

        cur_datetime = datetime.now()

        cur_datetime_utc = cur_datetime.astimezone(pytz.utc)

        def prefilter_pin(pin):
            if pin.cid is None:
                return False
            if pin.created_at < cur_datetime_utc - delta:
                return False

            return True

        # filter out pins that are not pinned and are older than delta
        prefiltered_pins = filter(prefilter_pin, pins)

        # sort by created_at descending
        prefiltered_pins = sorted(
            prefiltered_pins, key=lambda x: x.created_at, reverse=True
        )

        async def handle_pin(pin: IPFSPin):
            try:
                _verbose(f"Getting payload for {pin.cid}")
                raw_content = await get_from_ipfs(pin.cid, ipfs_api_url)
                pin.raw_content = raw_content
                return pin
            except Exception as e:
                bt.logging.error(f"Error getting content for {pin.cid}: {e}")
                return None

        final_pins: List[IPFSPin] = []

        for i in range(0, len(prefiltered_pins), batch_size):
            batch = prefiltered_pins[i : i + batch_size]
            _verbose(
                f"Processing batch {i} of {len(prefiltered_pins)}, {len(batch)} pins, from index {i} to {i+len(batch)}"
            )

            tasks = [handle_pin(pin) for pin in batch]
            results = await asyncio.gather(*tasks)

            for pin in results:
                if pin is None:
                    _verbose(f"Content for {pin.cid} is None")
                    continue
                final_pins.append(pin)

        return final_pins
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

    return raw_content


async def main():
    test_str = "hello worlddddd"
    test_file = "test.txt"
    with open(test_file, "w") as f:
        f.write(test_str)

    bt.logging.set_debug()

    cid = add_to_ipfs_cluster(test_file)
    if cid:
        bt.logging.debug(f"File added successfully. CID: {cid}")

        content = await get_from_ipfs(cid)
        if content:
            bt.logging.debug(f"Content: {content}")
        else:
            bt.logging.error("Failed to get content from IPFS.")
    else:
        bt.logging.error("Failed to add file to IPFS cluster.")

    pins = await get_pinned_cids()
    if pins:
        bt.logging.debug("Pinned CIDs and their content:")
        for pin in pins:
            bt.logging.debug(f"CID: {pin.cid}")
            bt.logging.debug(f"Content: {pin.raw_content}")
            bt.logging.debug("")
    else:
        bt.logging.error("Failed to get pinned CIDs.")


# Example usage
if __name__ == "__main__":
    asyncio.run(main())
