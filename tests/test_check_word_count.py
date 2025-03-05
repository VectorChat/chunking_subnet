from chunking.validator.reward import (
    check_chunk_words_in_document,
    check_chunks_end_on_sentence_boundaries,
    check_document_words_in_chunks,
    check_word_count,
)
from tests.utils.test_cases import read_word_count_test_cases


def test_check_word_count():

    document = "This is a test document. It is a test document. These sentences do not mean anything."
    ok_chunks = [
        "This is a test document.",
        "It is a test document.",
        "These sentences do not mean anything.",
    ]
    bad_chunks = [
        "This is a test document. It is a test document. These sentences do not",
        "These sentences do not mean anything.",
    ]

    # will still pass other checks

    assert check_chunks_end_on_sentence_boundaries(ok_chunks, document, True) == True
    assert check_chunks_end_on_sentence_boundaries(bad_chunks, document, True) == True

    assert check_document_words_in_chunks(document, ok_chunks, 3000) == True
    assert check_document_words_in_chunks(document, bad_chunks, 3000) == True

    for chunk in ok_chunks:
        assert check_chunk_words_in_document(chunk, document, True) == True
    for chunk in bad_chunks:
        assert check_chunk_words_in_document(chunk, document, True) == True

    # should fail word count check

    assert check_word_count(document, ok_chunks, True) == True
    assert check_word_count(document, bad_chunks, True) == False

    for doc, ok_chunks, bad_chunks in read_word_count_test_cases():
        print(f"Checking doc {doc[:100]}...")
        assert check_word_count(doc, ok_chunks, True) == True
        assert check_word_count(doc, bad_chunks, True) == False
