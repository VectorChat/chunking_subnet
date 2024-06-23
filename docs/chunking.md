**What is Chunking?** 

Chunking is the process of breaking down a set of data into smaller, more manageable chunks. This technique is essential in natural language processing (NLP) and particularly useful when working with large language models (LLMs). Chunking can involve various methods of segmentation, such as splitting an article into sections, a screenplay into scenes, or a recording of a concerto into movements.


**Why Chunk?** 

For LLMs to 'know' things, that information has to be included in their training data or in the request. When working with LLMs that need to access a vast base of knowledge, this knowledge base needs to be included in the request to prevent hallucinations. Due to the high cost of inference, it is impractical to include the entire corpus of data in every request.

To address this issue, we utilize chunking. By breaking down a set of data into smaller, manageable chunks and transforming these chunks into vectors with embedded meanings, we store them in a vector database. When a user sends a query, we embed the query as a vector and identify the vectors in the database with meanings most related to the query. Instead of loading an entire book into the model’s context, we only retrieve the relevant chunks of text, significantly reducing the total number of tokens processed per query.

Chunking, therefore, enables efficient and cost-effective querying by focusing on relevant portions of text, maintaining the balance between comprehensive knowledge and resource management.

Chunking is an important preliminary step for many machine learning (ML) tasks that use large amounts of data, such as:
- Retrieval-Augmented Generation (RAG): Rag utilizes a database of relevant documents to give LLMs the proper context to parse a particular query. Effective chunking results in more relevant and specific texts being included in the LLM’s context window, leading to better responses

- Classification: Chunking can be used to separate texts into similar sections, which can be then classified and assigned labels. This enhances the accuracy and efficiency of classification tasks

- Semantic Search: Improved chunking can enhance the accuracy and reliability of semantic searching algorithms, which return results based on the similarity in semantic meaning rather than simple keyword matching

**Why chunk intelligently?**

Unfortunately, current chunking solutions are inefficient and ineffective. When chunking many forms of data, it is necessary to retain parts of its structure in order for the resulting chunks to remain useful. For example, a csv file must be chunked such that all chunks contain the header and no rows are split between chunks. Strategies like fixed-size chunking fail to consider this and often result in useless chunks. Additionally, worse chunking algorithms result in pieces of information being spread out across multiple chunks, increasing the cost of inference. If a scene from a screenplay is split between two chunks, queries relating to things that happened in that scene will have to include both chunks.