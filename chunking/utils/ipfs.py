import requests
import json


def add_to_ipfs_cluster(file_path: str, api_url="http://localhost:9094"):
    """
    Add content to an IPFS cluster via the localhost IPFS Cluster API.

    :param file_path: Path to the file to be added
    :param api_url: URL of the IPFS Cluster API (default: http://localhost:9094)
    :return: CID of the added content
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


# Example usage
if __name__ == "__main__":
    test_str = "hello world"
    test_file = "test.txt"
    with open(test_file, "w") as f:
        f.write(test_str)

    cid = add_to_ipfs_cluster(test_file)
    if cid:
        print(f"File added successfully. CID: {cid}")
    else:
        print("Failed to add file to IPFS cluster.")
