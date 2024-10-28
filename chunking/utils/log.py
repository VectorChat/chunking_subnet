class PrefixStream:
    def __init__(self, stream, prefix):
        self.stream = stream
        self.prefix = prefix

    def write(self, message):
        # Handle newlines appropriately
        if message != '\n':
            self.stream.write(f"{self.prefix}{message}")
        else:
            self.stream.write(message)

    def flush(self):
        self.stream.flush()

    def isatty(self):
        return self.stream.isatty()

        
def debug_log_dict(data, truncate=100, indent=0):
    """
    Logs the contents of a dictionary, truncating values that exceed a certain length.

    Args:
        data: The dictionary to log
        truncate: The maximum number of characters for each value
        indent: The current indentation level (used for recursive formatting)
    """
    for key, value in data.items():
        prefix = ' ' * indent
        if isinstance(value, dict):
            print(f"{prefix}{key}: {{")
            debug_log_dict(value, truncate, indent + 2)
            print(f"{prefix}}}")
        elif isinstance(value, list):
            print(f"{prefix}{key}: [")
            for item in value:
                if isinstance(item, dict):
                    debug_log_dict(item, truncate, indent + 2)
                else:
                    truncated_item = (str(item)[:truncate] + '...') if len(str(item)) > truncate else item
                    print(f"{' ' * (indent + 2)}{truncated_item}")
            print(f"{prefix}]")
        else:
            truncated_value = (str(value)[:truncate] + '...') if len(str(value)) > truncate else value
            print(f"{prefix}{key}: {truncated_value}")
