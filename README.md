<div align="center">

# **Chunking Subnet on Bittensor** <!-- omit in toc -->

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

## Leading the way for better NLP <!-- omit in toc -->

[Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)
</div>

---
- [Introduction](#introduction)
- [Get Started](#get-started)
  - [Chunking](#chunking)
  - [Validator](#validator)
    - [Hardware Requirements](#hardware-requirements)
  - [Miner](#miner)
    - [Default Miner](#default-miner)
    - [Custom Miner](#custom-miner)
        - [Miner Considerations](#miner-considerations)
        - [Chunking Strategies](#chunking-strategies)

---
## Introduction

**Key Objective** - provide optimal chunking for use with large language models (LLMs)

Chunking is the splitting of large texts into smaller pieces with related contents. This could include things like splitting an article into sections, a book into chapters, or a screeplay into scenes.

**Example Use Cases**
- Retrieval-Augmented Generation (RAG) - RAG utilizes a database of relevant documents to give LLMs the proper context to answer a parse a particular query. Better chunking results in more relevant and specific texts being included in the LLMs context window, resulting in better responses.
- Classification - Chunking can be used to seperate texts into similar sections which can then be classified and assigned labels.
- Semantic Search - Better chunking can enhance the accuracy and reliability of semantic searching algorithms that return results based off of similarity in semantic meaning instead of keyword matching.
---

## Get Started

- Review min compute requirements for desired role
- Read through [Bittensor documentation](https://docs.bittensor.com/)
- Ensure you've gone through the [appropriate checklist](https://docs.bittensor.com/subnets/checklist-for-validating-mining)

### Chunking
This repository requires python3.8 or higher. To install, simply clone this repository and install the requirements.
```bash
git clone https://github.com/VectorChat/chunking_subnet
cd chunking_subnet
python -m pip install -r requirements.txt
python -m pip install -e 
```
Register your wallet to testnet or localnet
Testnet:
```bash
btcli subnet register --wallet.name $coldkey --wallet.hotkey $hotkey --subtensor.network test --netuid $uid
```
Localnet:
```bash
btcli subnet register --wallet.name <COLDKEY> --wallet.hotkey <HOTKEY> --subtensor.chain_endpoint ws://127.0.0.1:9946 --netuid 1
```

### Validator
Running a validator requires an OpenAI API key.
To run the validator:
```bash
# on localnet
python3 neurons/validator.py --netuid 1 --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name <COLDKEY> --wallet.hotkey <HOTKEY> --log_level debug --openaikey <OPENAIKEY>

# on testnet
python3 neurons/validator.py --netuid $uid --subtensor.network test --wallet.name <COLDKEY> --wallet.hotkey <HOTKEY> --log_level debug --openaikey <OPENAIKEY>
```
Something to consider when running a validator is the number of embeddings you're going to generate per miner evaluation. 
#### Hardware Reuquirements


### Miner
It is highly recomended that you write your own logic for neurons.miner.forward in order to achieve better chunking and recieve better rewards, for help on doing this, see [Custom Miner](#custom-miner).

```bash
# on localnet
python3 neurons/miner.py --netuid 1 --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name <COLDKEY> --wallet.hotkey <HOTKEY> --log_level debug

# on testnet
python3 neurons/miner.py --netuid $uid --subtensor.network test --wallet.name <COLDKEY> --wallet.hotkey <HOTKEY> --log_level debug
```
#### Default Miner
The default miner simply splits the incoming document into individual sentences and then forms each chunk by concatenating adjacent sentences until the token limit is reached. This is not an optimal strategy and will likely see you deregistered. For help on writing a custom miner, see [Custom Miner](#custom-miner).


#### Custom Miner
To change the behavior of the default miner, edit the logic in neurons.miner.forward to your liking.

##### Miner Considerations
Validators check each chunk that is sent to them against the source document that they sent out. To ensure that your chunks match the source document, it is highly encouraged that you use NLTKTextSplitter to split the document by sentences before combining them into chunks.

Each incoming query contains a variable called maxTokensPerChunk. Exceeding this number of tokens in any of your chunks will result in severe reductions in your score, so ensure that your logic does not produce chunks that exceed that number of tokens.

Chunk quality is calculated based on the semantic consistency within a given chunk and its semantic similarity to other chunks. In order to produce the best chunks, ensure that all text in a chunk is related and that text from different chunks are not.

##### Chunking Strategies
There are various approaches to chunking that can produce high-quality chunks. We recommend that you use recursive or semantic chunking. To learn more about chunking, we recommend you read through [this blogpost](https://www.pinecone.io/learn/chunking-strategies/)

###### Recursive Chunking
Recursive chunking first splits the document into a small number of chunks. It then preforms checks if the chunks fit the desired criteria (size, semantic self-similarity, etc.) If the chunks do not fit those criteria, the recursive chunking algorithm is called on the individual chunks to split them further.

###### Semantic Chunking
Semantic chunking first splits the document into individual sentences. It then creates a sentence group for each sentence consisting of the 'anchor' sentence and some number of surrounding sentences. These sentence groups are then compared to each other sequentially. Wherever there is a high semantic difference between adjacent sentence groups, one chunk is deliniated from the next.

