# Synthetic Queries

In the [default validator](./validator.md), synthetic queries are generated only in the absence of organic queries. Initially, requests will be primarily synthetic until the [Chunking.com Task API](./organic.md) is up and running. After that, the subnet will gradually shift to primarily organic queries.

Note that the load while the subnet is delivering primarily synthetic requests is likely far lower than the load when the subnet is primarily handling organic queries. Miners must be able to meet this demand, as non-answers are treated as zeros in the [Evaluation](./evaluation.md). 

Miners may have to deprioritize or ignore requests from lower-stake validators. See [Prioritzation & Blacklist](./miner_guide.md/#prioritzation--blacklist) in our [Guide to Mining](./miner_guide.md) to learn more.

## Generation

Each default validator, every time step (default: 60 seconds), and for each concurrent thread (default: 1), calls ```get_new_task()``` in [forward.py](../chunking/validator/forward.py).

If ```--accept_organic_queries``` is set to its default value of ```false```, or if the validator has not received an organic query from its Task API, it will generate a synthetic request by randoming selecting an article from a large repository of data.

The current iteration uses [Wikipedia’s Featured Articles](https://en.wikipedia.org/wiki/Wikipedia:Featured_articles) as the source data, consisting of 6,544 articles. If necessary, the source will expand to include the entire English Wikipedia (6,857,032 articles) or larger datasets such as [The Pile](https://pile.eleuther.ai/).

The ```generate_synthetic_synapse()``` function in [forward.py](../chunking/validator/forward.py):

```python
def generate_synthetic_synapse(validator) -> chunkSynapse:
    page = choice(validator.articles)
    document = requests.get('https://en.wikipedia.org/w/api.php', params={
        'action': 'query',
        'format': 'json',
        'pageids': page,
        'prop': 'extracts',
        'explaintext': True,
        'exsectionformat': 'plain',
        }).json()['query']['pages'][str(page)]['extract']
    document = document.replace("\n", " ").replace("\t", " ")
    document = ' '.join(document.split())
    synapse = chunkSynapse(document=document, time_soft_max=5.0, chunk_size=4096)
    return synapse
```

By default, synthetic requests have a ```chunk_size``` of 4096 characters and a ```time_soft_max``` of 5 seconds. Exceeding either the size or time restrictions results in exponential penalties to your score. See [Evaluation](./evaluation.md) to learn more.