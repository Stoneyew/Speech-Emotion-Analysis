# synthesizer/vocab.py
import string

class Vocabulary:
    def __init__(self):
        # Use printable characters and add special tokens for padding and unknown characters
        self.characters = list(string.printable) + ["<pad>", "<unk>"]
        self.char_to_index = {char: idx for idx, char in enumerate(self.characters)}
        self.index_to_char = {idx: char for idx, char in enumerate(self.characters)}
        self.pad_index = self.char_to_index["<pad>"]
        self.unk_index = self.char_to_index["<unk>"]

    def text_to_sequence(self, text):
        return [self.char_to_index.get(char, self.unk_index) for char in text]

    def sequence_to_text(self, sequence):
        return "".join([self.index_to_char[idx] for idx in sequence])

vocab = Vocabulary()
