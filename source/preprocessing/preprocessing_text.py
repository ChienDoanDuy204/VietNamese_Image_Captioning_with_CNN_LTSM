class Tokenizer:
    def __init__(self):
        pass
    def __call__(self, documentation):
        if documentation is None:
            raise ValueError(f"Tokenizer requirement a parameter documnetation !")
        return documentation.split(' ')

class BuiltVocabFromIterator:
    def __init__(self, itertor, vocab_size = 2000, special_tokens: list = None):
        self.vocab = {}
        for token in itertor:
            self.vocab[token] = self.vocab.get(token, 0) +1
    def show_vocab(self):
        print(self.vocab)