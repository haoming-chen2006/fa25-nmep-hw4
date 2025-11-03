from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from seq2seq.tokenizer.bpe_tokenizer import BPETokenizer


tokenizer = BPETokenizer()


class FrEnDataset(Dataset):
    def __init__(self, fr_en_path: Path):
        with open(fr_en_path / "europarl-v7.fr-en.fr", "r") as f:
            self.fr_lines = [line.rstrip() for line in f]

        with open(fr_en_path / "europarl-v7.fr-en.en", "r") as f:
            self.en_lines = [line.rstrip() for line in f]

    def __len__(self):
        return len(self.fr_lines)

    def __getitem__(self, idx: int):
        fr = self.fr_lines[idx]
        en = self.en_lines[idx]

        fr_tok = tokenizer.encode(fr)
        en_tok = tokenizer.encode(en)

        # Ensure all tensors are long type and truncate to reasonable length
        return torch.cat(
            [
                torch.tensor([tokenizer.bos_token_id], dtype=torch.long),
                fr_tok.long(),
                torch.tensor([tokenizer.eos_token_id], dtype=torch.long),
            ]
        )[:200], torch.cat(  # Truncate to max 200 tokens
            [
                torch.tensor([tokenizer.bos_token_id], dtype=torch.long),
                en_tok.long(),
                torch.tensor([tokenizer.eos_token_id], dtype=torch.long),
            ]
        )[:200]  # Truncate to max 200 tokens


def collate_fn(batch):
    in_seq = [item[0] for item in batch]
    target_seq = [item[1] for item in batch]

    pad_in = pad_sequence(
        in_seq, batch_first=True, padding_value=tokenizer.pad_token_id
    ).long()  # Ensure long type
    pad_target = pad_sequence(
        target_seq, batch_first=True, padding_value=tokenizer.pad_token_id
    ).long()  # Ensure long type

    return pad_in, pad_target
