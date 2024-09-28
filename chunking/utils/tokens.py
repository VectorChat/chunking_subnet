import tiktoken


def get_tokens_from_string(string: str, model: str) -> int:
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(string)
    return tokens


def get_string_from_tokens(tokens: list[int], model: str) -> str:
    encoding = tiktoken.encoding_for_model(model)
    return encoding.decode(tokens)


def num_tokens_from_string(string: str, model: str) -> int:
    """
    Helper function to calculate the number of tokens in a string.

    Args:
        string (str): The string to calculate the number of tokens for.
        encoding_name (str): The name of the encoding to use.

    Returns:
        int: The number of tokens in the string.
    """
    return len(get_tokens_from_string(string, model))
