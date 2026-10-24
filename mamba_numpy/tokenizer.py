"""A fitted Unicode character tokenizer; index zero is an unknown token."""

import numpy as np


class CharTokenizer:
    def __init__(self, characters):
        chars = list(characters)
        if any(not isinstance(c, str) or len(c) != 1 for c in chars) or len(set(chars)) != len(chars):
            raise ValueError("characters must be unique single-character strings")
        self.characters = chars
        self.stoi = {c: i + 1 for i, c in enumerate(chars)}

    @classmethod
    def fit(cls, text):
        if not text:
            raise ValueError("cannot fit tokenizer on empty text")
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.characters) + 1

    def encode(self, text):
        return np.array([self.stoi.get(c, 0) for c in text], dtype=np.int64)

    def decode(self, tokens):
        result = []
        for token in tokens:
            if not isinstance(token, (int, np.integer)) or not 0 <= token < self.vocab_size:
                raise ValueError("token ID outside tokenizer vocabulary")
            result.append(self.characters[token - 1] if token else "<unk>")
        return "".join(result)

    def to_dict(self):
        return {"type": "character", "characters": self.characters}

    @classmethod
    def from_dict(cls, data):
        if data.get("type") != "character":
            raise ValueError("unsupported tokenizer type")
        return cls(data["characters"])
