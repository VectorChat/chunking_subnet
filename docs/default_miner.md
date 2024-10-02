# Default Miner

The default miner simply splits the incoming document into individual sentences and then forms each chunk by concatenating adjacent sentences until the token limit specified by the validator is reached. This is not an optimal strategy and will likely result in very low yields or deregistration.

For tips on how build a better miner, [view our guide](./miner_guide.md).

## Prerequisites

- Review the minimum computational requirements for the desired role

- Ensure you have gone through the [checklist for validating and mining](https://docs.bittensor.com/subnets/checklist-for-validating-mining)

- Ensure that you have registered a hotkey for our subnet

- This repository requires python3.8 or higher

## Installation/Setup

This repository requires python 3.8 or higher. The following command will install the necessary dependencies and clone the repository:

```bash
curl -sSL https://raw.githubusercontent.com/VectorChat/chunking_subnet/main/setup.sh | bash
```

It is highly recomended that you write your own logic for neurons/miner.py:forward to optimize chunking and maximize rewards. For guidance creating a custom miner, check out [the miner guide](./miner_guide.md).

To run the miner, issue the following command:

```bash
bash run-miner.sh
```

Make sure to have your environment variables properly set in your `.env` file.
