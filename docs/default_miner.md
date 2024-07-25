# Default Miner

The default miner simply splits the incoming document into individual sentences and then forms each chunk by concatenating adjacent sentences until the token limit specified by the validator is reached. This is not an optimal strategy and will likely result in very low yields or deregistration.

For tips on how build a better miner, [view our guide](./miner_guide.md).

## Prerequisites
- Review the minimum computational requirements for the desired role

- Ensure you have gone through the [checklist for validating and mining](https://docs.bittensor.com/subnets/checklist-for-validating-mining)

- Ensure that you have registered a hotkey for our subnet

- This repository requires python3.8 or higher

## Installation

To install, simply clone this repository and install the requirements.
```bash
git clone https://github.com/VectorChat/chunking_subnet
cd chunking_subnet
pip3 install -e .
```

Install `punkt` tokenizer via the python REPL
```bash
python3
>>> import nltk
>>> nltk.download('punkt')
... 
>>> quit()
```

To run the miner, issue the following command:
```bash
python3 neurons/miner.py --netuid $uid --wallet.name <COLDKEY> --wallet.hotkey <HOTKEY> --log_level debug
```