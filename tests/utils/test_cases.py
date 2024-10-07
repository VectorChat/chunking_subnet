import os


def read_test_cases():
    directory = "tests/test_cases"

    for file in os.listdir(directory):
        with open(os.path.join(directory, file), "r") as f:
            yield f.read()
