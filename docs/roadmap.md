# Roadmap
The following phases are not necessarily sequential and may occur concurrently.

# Setup
The initial phase of the subnet, containing the basic functionality of the subnet, verifying the incentive mechanism works as intended, and creating resources for miners, validators, and potential consumers to monitor the subnet.

- [x] The first contest: the chunking of unstructured text
- [x] Autoupdate for Miners and Validators
- [x] Miner blacklist by Stake and Address
- [ ] Open-source Task API Framework to query the subnet
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
  - [ ] Unstructured audio (audio -> audio & audio -> text)
    - [ ] audio -> audio
    - [ ] audio -> text
  - [ ] Unstructured video (video -> video & video -> text)
    - [ ] video -> video
    - [ ] video -> text
  - [ ] Special file types (PDFs, CSV, JSON -> text)
  - [ ] Omnimodal (text, image, audio, video -> text, image, audio, video)


# Production
The subnet, having proven its ability to incentivize the best models, now shifts to enabling real, enterprise use. 

Protecting end-user data privacy becomes paramount; Validators and Miners cannot see the actual requested document, yet Miners must retain full ownership over their models as this is an inference based subnet.

- [ ] Introduce Facilitator System
    - [ ] Launch on testnet
    - [ ] Release 
    - [ ]
- [ ] Launch of Chunking.com Task API, helping deliver organic demand
    - [ ] Compensates Validators for Bandwidth

# Expansion
Subnet expands to embody the full RAG pipeline, forming a complete connection between Bittensor and consistent, real world demand.

As all components of the subnet are meant to be used in RAG, the Evaluation will change to a standardized RAG benchmark, where the independent variable is the subject of an individual contest.

- [ ] Begin Embedding Contests
    - [ ]
    - [ ]
- [ ] Begin Vector Database Contests
    - [ ] Vector Search Contest