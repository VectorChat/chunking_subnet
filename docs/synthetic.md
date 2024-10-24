# Synthetic Queries

Synthetic queries are generated continuously via an LLM. Currently, the LLM generates an article based off of a few Wikipedia pages.

The synthetic synapse generation function can be found here:

https://github.com/VectorChat/chunking_subnet/blob/cd54f5ebc082613bcfbb8326b3f06800a272a6bf/chunking/validator/task_api.py#L435-L455

The logic for generating the synthetic document can be found here:

https://github.com/VectorChat/chunking_subnet/blob/cd54f5ebc082613bcfbb8326b3f06800a272a6bf/chunking/validator/task_api.py#L287-L400
