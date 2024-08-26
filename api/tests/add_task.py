import requests
from chunking.validator.task_api import generate_synthetic_synapse

page = 33653136 

# Generate the 'synthetic' query: a featured article from wikipedia.
synapse, pageid = generate_synthetic_synapse(None, pageid=page, timeout=20)

print(f"Created synapse with document: {synapse.document[:100]} ...")

request_body = {
    "document": synapse.document,
    "time_soft_max": synapse.time_soft_max,
    "chunk_size": synapse.chunk_size,
}

request_endpoint = "http://localhost:8080/task_api/add_task/"

print(f"Sending request to {request_endpoint}")

response = requests.post(url=request_endpoint, json=request_body)

print(f"Received response with status code {response.status_code}")

if response.status_code != 200:
    print(f"Response: {response.text}")
    exit()

chunks = response.json()["chunks"]

print(f"Received {len(chunks)} chunks")
print(f"Chunk sizes: {[len(chunk) for chunk in chunks]}")