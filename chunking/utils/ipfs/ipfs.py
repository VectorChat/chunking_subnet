import asyncio
from datetime import timedelta, datetime
import os
from typing import Dict, List
import httpx
from pydantic import ValidationError
import pytz
import requests
import json
import bittensor as bt
from chunking.utils.ipfs.types import IPFSPin

import httpx
import json
from datetime import datetime, timedelta


async def add_to_ipfs_and_pin_to_cluster(
    file_path: str,
    expiry_delta: timedelta = None,
    ipfs_api_url: str = "http://localhost:5001",
    cluster_api_url: str = "http://localhost:9094",
    verbose: bool = False,
    **pin_options,
):
    """
    Add content to IPFS and then pin it to an IPFS cluster with optional expiry time and other pin options.

    Args:
        file_path (str): Path to the file to be added
        expiry_delta (timedelta): Time until the pin should expire (optional)
        ipfs_api_url (str): URL of the IPFS API (default: http://localhost:5001)
        cluster_api_url (str): URL of the IPFS Cluster API (default: http://localhost:9094)
        **pin_options: Additional pin options (in snake_case)

    Returns:
        dict: Response from the cluster pin operation
    """

    def _verbose(msg):
        if verbose:
            bt.logging.debug(msg)

    _verbose(f"Adding {file_path} to IPFS at {ipfs_api_url}")

    try:
        async with httpx.AsyncClient() as client:
            if not os.path.exists(file_path):
                raise Exception(f"File {file_path} does not exist")

            # Add file to IPFS
            with open(file_path, "rb") as file:
                files = {"file": file}
                ipfs_response = await client.post(
                    f"{ipfs_api_url}/api/v0/add", files=files
                )
                ipfs_response.raise_for_status()
                ipfs_result = ipfs_response.json()
                _verbose(f"IPFS result: {ipfs_result}")
                cid = ipfs_result["Hash"]
                _verbose(f"File added to IPFS with CID: {cid}")

            # Prepare pin options
            options = {
                "name": pin_options.get("name"),
                "mode": pin_options.get("mode"),
                "replication-min": pin_options.get("replication_factor_min"),
                "replication-max": pin_options.get("replication_factor_max"),
                "shard-size": pin_options.get("shard_size"),
                "user-allocations": (
                    ",".join(pin_options.get("user_allocations", []))
                    if pin_options.get("user_allocations")
                    else None
                ),
                "pin-update": pin_options.get("pin_update"),
                "origins": (
                    ",".join(pin_options.get("origins", []))
                    if pin_options.get("origins")
                    else None
                ),
            }

            # Add expiry time if provided
            if expiry_delta:
                expiry_time = datetime.utcnow() + expiry_delta
                options["expire-at"] = expiry_time.isoformat() + "Z"

            # Add metadata if provided
            if "metadata" in pin_options:
                options.update(
                    {f"meta-{k}": v for k, v in pin_options["metadata"].items()}
                )

            # Remove None values
            options = {k: v for k, v in options.items() if v is not None}

            # Pin to IPFS Cluster
            pin_endpoint = f"{cluster_api_url}/pins/{cid}"
            _verbose(f"Pinning CID {cid} to IPFS Cluster at {cluster_api_url}")
            _verbose(f"Pin options: {options}")

            cluster_response = await client.post(pin_endpoint, params=options)
            cluster_response.raise_for_status()
            result = json.loads(cluster_response.text)
            _verbose(f"Pin result: {result}")
            cid = result.get("cid")
            return cid

    except httpx.HTTPStatusError as e:
        _verbose(f"HTTP error occurred: {e}")
        return None
    except Exception as e:
        _verbose(f"An error occurred: {str(e)}")
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
            _verbose(f"Response: {response}")
            response.raise_for_status()

        res_text_as_json_array = f"[{','.join(response.text.splitlines())}]"
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

    def _verbose(msg):
        if verbose:
            bt.logging.debug(msg)

    _verbose(f"Getting {cid} from IPFS at {api_url}")

    async with httpx.AsyncClient() as client:
        _verbose(f"Sending request to {endpoint} with params {params}")
        response = await client.post(endpoint, params=params)
        _verbose(f"Response: {response}")
        response.raise_for_status()

    _verbose(f"Response: {response}")
    _verbose(f"Response status: {response.status_code}")
    _verbose(f"Response content: {response.content}")
    raw_content = response.content.decode("utf-8")

    return raw_content


async def main():
    test_str = "test test test test"
    test_file = "test.txt"
    with open(test_file, "w") as f:
        f.write(test_str)

    bt.logging.set_debug()

    cid = await add_to_ipfs_and_pin_to_cluster(
        test_file, expiry_delta=timedelta(minutes=1), verbose=True
    )

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
