# Prerequisites

- Review the minimum computational requirements for the desired role

- Ensure that you have gone through the [checklist for validating and mining](https://docs.bittensor.com/subnets/checklist-for-validating-mining)

- Ensure that you have registered a hotkey for our subnet and stake > X TAO

# Installation

This repository requires python3.8 or higher. To install, simply clone this repository and install the requirements.
```bash
git clone https://github.com/VectorChat/chunking_subnet
cd chunking_subnet
pip3 install -e .
```

Install `punkt` model via the python repl
```bash
python3
>>> import nltk
>>> nltk.download('punkt')
...
>>> quit()
```

Running a validator requires an OpenAI API key. To run the validator issue the following command:
```bash
python3 neurons/validator.py --netuid $uid  --wallet.name <COLDKEY> --wallet.hotkey <HOTKEY> --log_level debug --openaikey <OPENAIKEY>
```

# Considerations

When validating, something to consider is number of embeddings you’re willing to generate per miner evaluation.

When evaluating a miner, a random sample of 3-sentence segments are taken from the response and embedded. The dot product of every possible pair of these embeddings is then compared and added to the final score if the embeddings originated from the same chunk or are subtracted from the final score if they originated from different chunks. A greater sample size will likely result in more accurate evaluation and higher dividends, but comes at the cost of increased API calls to generate the embeddings and more time and resources to then compare them against each other. 

You can set this value with the following argument when running your validator:
```bash
--numEmbeddings <VALUE>
```

To earn additional revenue, you can opt into receiving organic queries from users wanting to use the subnet. To do this, set the environment variable: ACCEPT_ORGANIC_CHUNKING_QUERIES to 'True' or use the following argument when running your validator:
```bash
--accept_organic_queries
```

Organic queries come with a list of miners to query. If these miners do not respond to your query, the code will send back a response from a different miner. Responses from miners not in specified list will result in lower pay. Payment is sent to your validator's coldkey once your response is verified.