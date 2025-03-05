import os
import pandas as pd


def read_test_cases():
    directory = "tests/test_cases"

    for file in os.listdir(directory):
        with open(os.path.join(directory, file), "r") as f:
            yield f.read()


def read_chunks_from_round_csv(file_path: str):
    df = pd.read_csv(file_path)

    chunks = []

    for index, row in df.iterrows():
        yield row["Chunk Content"]


def read_word_count_test_cases():
    directory = "tests/word_count_cases"

    i = 1
    while True:
        doc_path = os.path.join(directory, f"{i}.txt")
        if not os.path.exists(doc_path):
            break

        with open(doc_path, "r") as f:
            doc = f.read()

        ok_chunks = read_chunks_from_round_csv(os.path.join(directory, f"{i}_ok.csv"))
        bad_chunks = read_chunks_from_round_csv(os.path.join(directory, f"{i}_bad.csv"))

        yield doc, ok_chunks, bad_chunks

        i += 1
