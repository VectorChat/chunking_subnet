import requests
import json


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


def get_from_ipfs(cid: str, api_url="http://localhost:5001"):
    """
    Get content from IPFS via the IPFS HTTP API.

    Args:
        cid (str): CID of the content to be retrieved
        api_url (str): URL of the IPFS API (default: http://localhost:5001)

    Returns:
        bytes: Content of the retrieved CID

    """
    endpoint = f"{api_url}/api/v0/cat"
    params = {"arg": cid}

    print(f"Getting {cid} from IPFS at {api_url}")

    try:
        response = requests.post(endpoint, params=params)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        print(f"Response: {response}")
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.content}")
        return response.content.decode("utf-8")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {str(e)}")
        return None


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
