from chunking.validator.reward import check_chunk_words_in_document
from chunking.validator.task_api import (
    generate_doc_normal,
    generate_synthetic_synapse,
    get_wiki_content_for_page,
)
from nltk.tokenize import sent_tokenize
import random

from tests.utils import get_articles, base_chunker


def create_bad_chunk(chunk: str):
    # remove a random word from the chunk
    words = chunk.split()
    if len(words) == 1:
        return chunk
    index = random.randint(0, len(words) - 1)
    words.pop(index)
    return " ".join(words)


def test_chunk_words():
    # courteousy of tvxq19910509
    test_document = (
        "Mammoths – Giants of the Ice Age (3 ed.). With some extra words here."
    )
    test_chunk = "Giants of the Ice Age (3 ed.)."

    assert (
        check_chunk_words_in_document(test_chunk, test_document, verbose=True) == True
    )

    test_doc, pageid = generate_doc_normal(None)

    print(f"created test doc of length {len(test_doc)}, pageid = {pageid}")

    test_chunks = base_chunker(test_doc, 4096)

    print(f"created {len(test_chunks)} chunks")

    checks = [
        check_chunk_words_in_document(chunk, test_doc, verbose=True)
        for chunk in test_chunks
    ]

    assert all(checks)

    # create bad chunks by removing word from random chunk
    bad_chunks = [create_bad_chunk(chunk) for chunk in test_chunks[:10]]

    assert check_chunk_words_in_document(bad_chunks[0], test_doc, verbose=True) == False

    # test random articles

    articles = get_articles()

    sample_size = 200

    random_articles = random.sample(articles, sample_size)

    additional_pageids = [26623321]

    articles_to_test = additional_pageids + random_articles

    for article in articles_to_test:
        print(f"testing article {article}")
        document = get_wiki_content_for_page(article)

        chunks = base_chunker(document, 4096)

        checks = [
            check_chunk_words_in_document(chunk, document, verbose=True)
            for chunk in chunks
        ]

        assert all(checks)
