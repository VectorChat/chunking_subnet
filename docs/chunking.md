# What is Chunking?

Chunking is the process of breaking down a set of data into smaller, more manageable "chunks" of data. This technique is essential in natural language processing (NLP) and particularly useful when working with large language models (LLMs). Chunking can involve various methods of segmentation, such as splitting an article into sections, a screenplay into scenes, or a recording of a concerto into movements.


## Why Chunk? 

For LLMs to provide accurate information, it must have access to that information. Thus, when LLMs need to access extensive knowledge beyond its training data, that information needs to be part of the request to prevent hallucinations. Due to the high cost of inference, including the entire corpus of data in every request is impractical.

We address this issue with chunking. By breaking down data into smaller chunks and transforming these chunks into vectors with embedded meanings, we store them in a vector database. When a user sends a query, we embed it as a vector and identify the vectors in the database that are most related to the query. For example, instead of loading an entire book into the model’s context, we retrieve only the relevant chunks of text, significantly reducing the total number of tokens processed per query.

Chunking, therefore, enables efficient and cost-effective querying by focusing on relevant portions of text, maintaining the balance between comprehensive knowledge and resource management.

Chunking is a crucial preliminary step for many machine learning (ML) tasks that use large amounts of data, such as:

- **Retrieval-Augmented Generation (RAG):** RAG utilizes a database of relevant documents to give LLMs the proper context to parse a particular query. Effective chunking results in more relevant and specific texts being included in the LLM’s context window, leading to better responses.

- **Classification:** Chunking can separate texts into similar sections, which can then be classified and assigned labels. This enhances the accuracy and efficiency of classification tasks.

- **Semantic Search:** Improved chunking can enhance the accuracy and reliability of semantic searching algorithms, which return results based on semantic meaning rather than simple keyword matching.

## Why chunk intelligently?

See [The Case for Intelligent Chunking](https://medium.com/@vectorchat/the-case-for-intelligent-chunking-3f903aa3a72c).