import string
class Preprocessing:
    def __init__(self) -> None:
        pass
    def remove_punctuation_digit(self, setence):
        char_remove = string.punctuation + string.digits
        for char in char_remove:
            sentence = setence.replace(char,' ')
        return sentence
    def tranfer2Lower(self, sentence):
        sentence = sentence.lower()
        return sentence
