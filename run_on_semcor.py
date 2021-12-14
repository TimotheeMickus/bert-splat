import collections
import random
import re


from nltk.corpus import semcor
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
import tqdm

from linear_structure import model, read_factors_last_layer

def build_refs():
    refs = collections.defaultdict(set)
    for sentence in semcor.tagged_sents(tag='sem'):
        for word in sentence:
            try:
                key = word.label().name()
                value = word.label().key()
                refs[key].add(value)
            except AttributeError:
                if type(word) not in (list, str):
                    assert type(word.label()) == str
                    key = word.label().split('.', 1)[0]
                    value = word.label()
                    refs[key].add(value)

                # continue
    return {
        k: list(v)
        for k, v in refs.items()
        if len(v) > 1
    }

refs = build_refs()
refs_stoi = sorted(refs.keys())

IPT_SIZE = 768
INNER_SIZE = 256

wsd_model = nn.ModuleList([
    nn.Sequential(
            nn.Linear(IPT_SIZE, INNER_SIZE),
            nn.ReLU()
    ),
    nn.ModuleList([
        nn.Linear(INNER_SIZE, len(refs[k]))
        for k in refs_stoi
    ])
])

def overlap(a_start, a_end, b_start, b_end):
    return (a_start <= b_start and a_end > b_start) \
        or (a_start < b_end and a_end >= b_end) \
        or (a_start <= b_start and a_end >= b_end) \
        or (a_start >= b_start and a_end <= b_start)

def parse_semcor():
    all_items = []
    was_grad_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    for sentence in tqdm.tqdm(semcor.tagged_sents(tag='sem')):
        sentence_tokens = [
            word
            for subtree in sentence
            for word in (
                subtree if type(subtree) is list else subtree.flatten()
            )
        ]
        sentence_str = " ".join(sentence_tokens).replace('``', '"').replace("''", '"')
        tagged = []
        offsets = []
        prev_idx = 0
        for tok in sentence:
            if type(tok) == list:
                prev_idx += 1 + len(" ".join(tok).replace('``', '"').replace("''", '"'))
            else:
                segment_len = len(" ".join(tok.flatten()).replace('``', '"').replace("''", '"'))
                offsets.append([prev_idx, prev_idx + segment_len])
                tagged.append(tok)
                prev_idx += 1 + segment_len

        if not len(tagged) > 0:
            import pdb; pdb.set_trace()
        embedded_toks = read_factors_last_layer(sentence_str)
        for tagged_item, offset in zip(tagged, offsets):
            relevant_embs = [
                emb
                for emb in embedded_toks
                if overlap(
                    emb['start_idx'],
                    emb['end_idx'],
                    offset[0],
                    offset[1],
                )
            ]
            assert len(relevant_embs) > 0
            dict_item = {
                'ipt': sum(r['ipt'] for r in relevant_embs),
                'norm': sum(r['norm'] for r in relevant_embs),
                'ff': sum(r['ff'] for r in relevant_embs),
                'mha': sum(r['mha'] for r in relevant_embs),
                'tag': tagged_item
            }
            all_items.append(dict_item)
    torch.set_grad_enabled(was_grad_enabled)
    return all_items


class BertFactorSemcorDataset(Dataset):
    def __init__(self, items=None):
        if items is not None:
            self.items = items
        else:
            self.items = parse_semcor()

    def save(self, file):
        torch.save(self, file)

    @staticmethod
    def load(file):
        return torch.load(file)

    @classmethod
    def splits(cls):
        items = parse_semcor()
        random.shuffle(items)
        train_ = (len(items) * 8) // 10
        dev_ = (len(items) * 9) // 10
        train = items[:train_]
        dev = items[train_:dev_]
        test = items[dev_:]
        return cls(items=train), cls(items=dev), cls(items=test)

train_dataset, dev_dataset, test_dataset = BertFactorSemcorDataset.splits()
