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