import re

TEST_INPUT = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
"""
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    vocab: dict[int, bytes] = {}
    freq: dict[tuple[bytes], int] = {}

    def _initialize_vocabs(self):
        for i in range(256):
            self.vocab[i] = bytes([i])
        self.vocab[256] = b"<|endoftext|>"

    def __init__(self):
        self._initialize_vocabs()

    def pre_tokenization(self):
        list = TEST_INPUT.split()
        for word in list:
            word_bytes_tuple = tuple([bytes([byte]) for byte in word.encode()])
            self.freq[word_bytes_tuple] = self.freq.get(word_bytes_tuple, 0) + 1
        print(self.freq)

    def merge(iteration=1):
        pass


if __name__ == "__main__":
    tokenizer = Tokenizer()
    print(tokenizer.vocab)
    tokenizer.pre_tokenization()
