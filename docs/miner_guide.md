# Tips For How To Improve Your Miner

## Evaluation

First and foremost, we recommend you gain a strong understand of the Incentive Mechanism used by Validators, as that is what you are optimizing for.

Validators check each chunk that is sent to them against the source document that they sent out. To ensure that your chunks match the source document, it is highly encouraged that you use NLTK's sentence_tokenizer to split the document by sentences before combining them into chunks.

Each incoming query contains a variable called tokensPerChunk. Exceeding this number of tokens in any of your chunks results in exponentially severe reductions to your score, so ensure that your logic does not produce chunks that exceed that number of tokens.

Chunk quality is calculated based on the semantic similarity within a given chunk and its dissimilarity to other chunks. In order to produce the best chunks, ensure that all text in a chunk is related and that text from different chunks are not.

Finally, note that there is a soft-time limit, currently set to X seconds. Validators exponentially penalize responses for each second they are late.
```python
reward *= (2/3) ** over_time
```

## Chunking Strategies
There are various approaches to chunking that can produce high-quality chunks. We recommend that you start by exploring recursive and semantic chunking. To learn more about chunking, we recommend you read this [Pinecone article](https://www.pinecone.io/learn/chunking-strategies/).

### Recursive Chunking

Recursive chunking begins by splitting the data set into a small number of chunks. It then checks whether each chunk meets the desired criteria (such as size or semantic self-similarity.) If a chunk does not meet these criteria, the algorithm recursively splits that chunk into smaller chunks. This process continues until all chunks satisfy the specified criteria. 

Here is a diagram of this process:

![recursive_chunking](../assets/recursive_chunking.png)

### Semantic Chunking

Semantic chunking starts by splitting the entire data set into individual sentences. Each sentence, called an anchor sentence, is then grouped with a number of surrounding sentences to form a sentence group. These sentence groups are compared sequentially. A chunk boundary is established wherever the semantic difference between adjacent sentence groups crosses some threshold.

Here is an example with a threshold of 1:

![semantic_chunking](../assets/semantic_chunking.png)

### Prebuilt Solutions

There exist many freely available chunking utilities that can help you get a head start on your chunking algorithm, see the following links:
[Pinecone's Respository](https://github.com/pinecone-io/examples/tree/master/learn/generation/better-rag)
[LangChain Text Splitting Documentation](https://js.langchain.com/v0.1/docs/modules/data_connection/document_transformers/)