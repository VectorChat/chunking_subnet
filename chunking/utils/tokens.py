import tiktoken


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """
    Helper function to calculate the number of tokens in a string.

    Args:
        string (str): The string to calculate the number of tokens for.
        encoding_name (str): The name of the encoding to use.

    Returns:
        int: The number of tokens in the string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
