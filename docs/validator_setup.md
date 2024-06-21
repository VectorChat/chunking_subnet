# Prerequisites

- Review the minimum computational requirements for the desired role

- Ensure that you have gone through the [checklist for validating and mining](https://docs.bittensor.com/subnets/checklist-for-validating-mining)

- Ensure that you have registered a hotkey for our subnet and stake > X TAO

# Installation

This repository requires python3.8 or higher. To install, simply clone this repository and install the requirements.
```bash
git clone https://github.com/VectorChat/chunking_subnet
cd chunking_subnet
python -m pip install -r requirements.txt
python -m pip install -e .
```


Running a validator requires an OpenAI API key. To run the validator issue the following command:
```bash
python3 neurons/validator.py --netuid $uid  --wallet.name <COLDKEY> --wallet.hotkey <HOTKEY> --log_level debug --openaikey <OPENAIKEY>
```

# Considerations

Something to consider when running a validator is the number of embeddings you’re going to generate per miner evaluation. When evaluating a miner, a random sample of 3-sentence segments are taken from the response and embedded. The dot product of every possible pair of these embeddings is then compared and added to the final score if the embeddings originated from the same chunk or are subtracted from the final score if they originated from different chunks. A greater sample size will likely result in more accurate evaluation and higher dividends. This comes at the cost of more API calls to generate the embeddings and more time and resources to compare them against each other.
