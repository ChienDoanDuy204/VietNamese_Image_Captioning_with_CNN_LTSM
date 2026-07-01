import json
class Tokenizer:
    def __init__(self):
        pass
    def __call__(self, documentation):
        if documentation is None:
            raise ValueError(f"Tokenizer requirement a parameter documnetation !")
        return documentation.split(' ')

class BuiltVocabFromIterator:
    def __init__(self, iterator = None, vocab_size = 2000, special_tokens: list = None):
        self.vocab_freq = {}
        self.vocab = None
        self.special_tokens = special_tokens
        self.vocab_size = vocab_size

        if iterator is not None:
            # variable to check type data of special_tokens is list
            check_type = False
            if special_tokens is not None:
                if isinstance(special_tokens, list) and len(special_tokens):
                    check_type = True
                else:
                    raise ValueError(f"special_tokens must be list contain tokens")
            else:
                raise ValueError("BuiltVocabFromIterator forget a parameter special_tokens")
            # build vocab frequency of each word
            sentence = next(iterator, 0) # trả về 0 thay vì lỗi nếu out index
            while sentence:
                for token in sentence:
                    self.vocab_freq[token] = self.vocab_freq.get(token, 0) +1
                    sentence = next(iterator,0)

            # buil vocab index presentation top vocab_size word have most frequency is sorted follow the values of vocab_freq
            TopKwordMostFreq = sorted(self.vocab_freq, key= lambda x:self.vocab_freq[x], reverse=True)[:vocab_size]
            if special_tokens is None:
                self.vocab = {word : index for index, word in enumerate(TopKwordMostFreq)}
            if special_tokens is not None and check_type:
                total_word = special_tokens.copy()
                total_word.extend(TopKwordMostFreq[:(len(TopKwordMostFreq)-len(self.special_tokens))])
                self.vocab = {word : indx for indx, word in enumerate(total_word)}
        self.idx2str = {idx: word for word, idx in self.vocab.items()}
    
    def set_default(self,key):
        self.default_key = key

    # string to index
    def stoi(self):
        return self.vocab
    # index to string
    def itos(self):
        return self.idx2str

    
    # save information of dict
    def save(self, path:str):
        data = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "vocab": self.vocab,
            "idx2str": self.idx2str
        }
        with open(file= path, mode ='w', encoding='utf-8') as f:
            json.dump(data,f, ensure_ascii=False, indent=2)
        print(f"Save success vocab at {path}")
    
    # load vocab to use
    def load(self, path:str):
        with open(file = path, mode ='r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.vocab = data["vocab"]
        self.special_tokens = data["special_tokens"]
        self.idx2str = data["idx2str"]
    
    
    # convert text2index - presentation indexing
    def __call__(self, list_token):
        present_idx = [self.vocab.get(token, self.vocab[self.default_key]) for token in list_token]
        return present_idx
