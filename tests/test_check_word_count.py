from chunking.validator.reward import check_word_count
from tests.utils.test_cases import read_word_count_test_cases


def test_check_word_count():

    for doc, ok_chunks, bad_chunks in read_word_count_test_cases():
        print(f"Checking doc {doc[:100]}...")
        assert check_word_count(doc, ok_chunks, True) == True
        assert check_word_count(doc, bad_chunks, True) == False
