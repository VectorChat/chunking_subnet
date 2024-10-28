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

This repository requires python 3.10 or higher. The following command will install the necessary dependencies and clone the repository:

```bash
curl -sSL https://raw.githubusercontent.com/VectorChat/chunking_subnet/main/setup.sh | bash
```

Then, setup _Lucid_, our relay mining prevention mechanism, by following the instructions [here](./lucid/setup.md). _Lucid_ is required for running a validator and runs
as a separate group of services.

To run the validator issue the following command after setting your OpenAI API key in the environment variable (brackets indicate optional arguments):

```bash
bash run-validator.sh [...]
```

## Configuration

| Argument                                    | Description                                                                                               |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `--neuron.timeout`                          | Timeout for calling miners in synthetic tournament rounds                                                 |
| `--neuron.num_concurrent_forwards`          | Number of concurrent forwards/synthetic tournament rounds at a time                                       |
| `--wandb.project_name`                      | Name of the wandb project to log to (only needs to be changed if on testnet or logging to custom project) |
| `--neuron.disable_set_weights`              | Disable the set weights mechanism                                                                         |
| `--neuron.axon_off`                         | Set this flag to not attempt to serve an Axon                                                             |
| `--num_embeddings`                          | Number of embeddings to generate and compare when rewarding miners in tournament rounds                   |
| `--neuron.skip_set_weights_extrinsic`       | Skip the set_weights extrinsic call (only logs to W&B)                                                    |
| `--wandb.wandb_off`                         | Turn off wandb logging                                                                                    |
| `--enable_task_api`                         | If set, runs the integrated API for that can be queried by external clients                               |
| `--task_api.host`                           | The host for the task API                                                                                 |
| `--task_api.port`                           | The port for the task API                                                                                 |
| `--neuron.synthetic_query_interval_seconds` | The interval to sleep between synthetic queries in seconds.                                               |
| `--debug.on`                                | Turn on debug logging.                                                                                    |
| `--debug.all_log_handlers`                  | If in debug mode, turns on all third party log handlers                                                   |
