from .tokenizer import Tokenizer

import torch


class CharacterTokenizer(Tokenizer):
    def __init__(self, verbose: bool = False):
        """
        Initializes the CharacterTokenizer class for French to English translation.
        If verbose is True, prints out the vocabulary.

        We ignore capitalization.

        Implement the remaining parts of __init__ by building the vocab.
        Implement the two functions you defined in Tokenizer here. Once you are
        done, you should pass all the tests in test_character_tokenizer.py.
        """
        super().__init__()

        self.vocab = {}

        # Normally, we iterate through the dataset and find all unique characters. To simplify things,
        # we will use a fixed set of characters that we know will be present in the dataset.
        self.characters = """aàâæbcçdeéèêëfghiîïjklmnoôœpqrstuùûüvwxyÿz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}’•–í€óá«»… º◦©ö°äµ—ø­·òãñ―½¼γ®⇒²▪−√¥£¤ß´úª¾є™，ﬁõ  �►□′″¨³‑¯≈ˆ§‰●ﬂ⇑➘①②„≤±†✜✔➪✖◗¢ไทยếệεληνικαåşıруский 한국어汉语ž¹¿šćþ‚‛─÷〈¸⎯×←→∑δ■ʹ‐≥τ;∆℡ƒð¬¡¦βϕ▼⁄ρσ⋅≡∂≠π⎛⎜⎞ω∗"""

        self.vocab = {c : i for i,c in enumerate(list(self.characters))}
        # tokens maps index -> character for decoding
        self.tokens = {i: c for c, i in self.vocab.items()}
        if verbose:
            print("Vocabulary:", self.vocab)
        

    def encode(self, text: str) -> torch.Tensor:
        text = text.lower()
        result = []
        for s in text:
            # if character not in vocab, you can choose to skip or raise; we'll skip
            if s in self.vocab:
                result.append(self.vocab[s])
        if len(result) == 0:
            return torch.tensor([], dtype=torch.long)
        return torch.tensor(result, dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        # tokens can be a torch.Tensor or iterable of ints
        result = []
        if isinstance(tokens, torch.Tensor):
            iterable = tokens.tolist()
        else:
            iterable = list(tokens)

        for s in iterable:
            if s in self.tokens:
                result.append(self.tokens[s])
        return ''.join(result)

a = CharacterTokenizer()
print(a.vocab)

        
