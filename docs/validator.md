# Validator

The default validator is responsible for evaluating the responses of miners and setting weights accordingly. These mechanisms are defined in detail throughout the [subnet architecture](../README.md#architecture).

## Computational Requirements

The computing power needed to successfully validate on this subnet is relatively low and should work on most configurations with a stable internet connection.

The minimum and suggested specs are outlined in the [min_compute.yml](../min_compute.yml):

```yml
validator:
  cpu:
    min_cores: 2 # Minimum number of CPU cores
    min_speed: 2.5 # Minimum speed per core (GHz)
    recommended_cores: 4 # Recommended number of CPU cores
    recommended_speed: 3.5 # Recommended speed per core (GHz)
    architecture: "x86_64" # Architecture type (e.g., x86_64, arm64)

  gpu:
    required: False # Does the application require a GPU?

  memory:
    min_ram: 4 # Minimum RAM (GB)
    min_swap: 2 # Minimum swap space (GB)
    recommended_swap: 4 # Recommended swap space (GB)
    ram_type: "DDR4" # RAM type (e.g., DDR4, DDR3, etc.)

  storage:
    min_space: 10 # Minimum free storage space (GB)
    recommended_space: 32 # Recommended free storage space (GB)
    type: "SSD" # Preferred storage type (e.g., SSD, HDD)
    min_iops: 1000 # Minimum I/O operations per second (if applicable)
    recommended_iops: 5000 # Recommended I/O operations per second

  os:
    name: "Ubuntu" # Name of the preferred operating system(s)
    version: 20.04 # Version of the preferred operating system(s)

network_spec:
  bandwidth:
    download: 100 # Minimum download bandwidth (Mbps)
    upload: 100 # Minimum upload bandwidth (Mbps)
```

## Prerequisites

- Review the minimum computational requirements for the desired role

- Ensure that you have gone through the [checklist for validating and mining](https://docs.bittensor.com/subnets/checklist-for-validating-mining)

- Ensure that you have registered a hotkey for our subnet

- Running a validator requires an OpenAI API key and a W&B API key

## Installation/Setup

This repository requires python 3.8 or higher. The following command will install the necessary dependencies and clone the repository:

```bash
curl -sSL https://raw.githubusercontent.com/VectorChat/chunking_subnet/main/setup.sh | bash
```

To run the validator issue the following command after setting your OpenAI API key in the environment variable (brackets indicate optional arguments):

```bash
bash run-validator.sh [...]
```

## Flags

When validating, something to consider is number of embeddings you’re willing to generate per miner evaluation.

When evaluating a miner, a random sample of 3-sentence segments are taken from the response and embedded. The dot product of every possible pair of these embeddings is then compared and added to the final score if the embeddings originated from the same chunk or are subtracted from the final score if they originated from different chunks. A greater sample size will likely result in more accurate evaluation and higher dividends, but comes at the cost of increased API calls to generate the embeddings and more time and resources to then compare them against each other.

You can set this value with the following argument when running your validator:

```bash
--num_embeddings <VALUE>
```

To earn additional revenue, you can opt into receiving organic queries from users wanting to use the subnet. To do this, set the environment variable: ACCEPT_ORGANIC_CHUNKING_QUERIES to 'True' or use the following argument when running your validator:

```bash
--accept_organic_queries
```

Organic queries come with a list of miners to query. If these miners do not respond to your query, the code will send back a response from a different miner. Responses from miners not in specified list will result in lower pay. Payment is sent to your validator's coldkey once your response is verified.
