# Roadmap
The following phases are not necessarily sequential and may occur concurrently.

## Setup
The initial phase of the subnet, containing the basic functionality of the subnet, verifying the incentive mechanism works as intended, and creating resources for miners, validators, and potential consumers to monitor the subnet.

###  Objectives:
- [x] The first contest: the chunking of unstructured text
- [x] Autoupdate for Miners and Validators
- [x] Miner blacklist by Stake and Address
- [x] Open-source Task API Framework to query the subnet
- [ ] Subnet Dashboard for Miners and Validators
  - [ ] Display Subnet KPIs
  - [ ] Periodic global benchmark
  - [ ] Display subnet performance over time
  - [ ] Chunking visualizer of all synthetic chunks produced
  - [x] Display all Miner and Validator data
  - [x] Show all tournament rounds: global, by validator, and by miner
- [ ] Unique synthetic queries via Wikipedia + LLM, mitigating potential for lookup attacks and providing miners a way to detect and prevent relay mining
- [ ] Release extensive guides on how to build excellent chunkers
- [ ] Begin hosting multiple winner-take-all contests, for different resultant types of smart chunking
  - [ ] Unstructured images
    - [ ] image -> image
    - [ ] image -> text
  - [ ] Unstructured audio
    - [ ] audio -> audio
    - [ ] audio -> text
  - [ ] Unstructured video
    - [ ] video -> video
    - [ ] video -> text
  - [ ] Special file types (PDFs, CSV, JSON -> text)
  - [ ] Omnimodal (text, image, audio, video -> text, image, audio, video)


##  Production
The next phase of the subnet aims to make the intelligence incentivized by this subnet viable for commercial, enterprise, and personal use. Ensuring end-user data privacy becomes paramount. 

The incentive mechanism must change such that Validators and Miners never have access to the documents sent to be chunked, and such that no other parties ever gain the models created by Miners. 

*But how might that work?* [See our (very tentative) approach.](https://docs.google.com/document/d/1tmk9LuvWmKozC7DBvON4o9Dywe5D3S78TgLeuSah1MI/edit?usp=sharing)

###  Objectives:
- [ ] Private Organic Query System
    - [ ] Finalize, collaborate, and share with other subnets in Bittensor
    - [ ] Launch on testnet
    - [ ] Release on mainnet
- [ ] Launch of Chunking.com Task API, helping deliver organic demand
    - [ ] Dashboard for Validators to monitor compensation and bandwidth use
- [ ] Achieve significant real world demand flowing into the subnet

## ETL for RAG
The third phase has the subnet expand to subsume the full Retrieval-Augmented Generation (RAG) pipeline, forming a complete connection between Bittensor and viable, real world demand.

*As all components of the subnet are meant to be used in RAG, the Evaluation will change to a standardized RAG benchmark, where the independent variable for any given contest is the subject of that very contest.*

- [ ] Preprocessing Contests
    - [ ] Text and Structure Extraction
- [ ] Embedding Contests
    - [ ] Text
    - [ ] Audio
    - [ ] Image
    - [ ] Video
- [ ] Vector DB Contests
    - [ ] Vector Search