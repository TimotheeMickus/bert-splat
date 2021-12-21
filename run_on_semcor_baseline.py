import collections
import itertools
import functools
import json
import math
import pathlib
import pprint
import random
import re
import skopt

import more_itertools
from nltk.corpus import semcor
from nltk.corpus import wordnet as wn
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
import tqdm

import linear_structure

DROP_MONOSEMOUS = True

def build_refs():
    """build references for easier computations: dict {lemma -> [tags]}"""
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
    if DROP_MONOSEMOUS:
        return {
            k: sorted(list(v))
            for k, v in refs.items()
            if len(v) > 1
        }
    return {
        k: sorted(list(v))
        for k, v in refs.items()
    }

refs = build_refs()
refs_itos = sorted(refs.keys())
refs_stoi = {
    lemma: idx
    for idx, lemma in enumerate(refs_itos)
}
tags_stoi = {
    tag: torch.tensor([idx], dtype=torch.long)
    for idx, tag in enumerate(
        item
        for lemma in refs_stoi
        for item in refs[lemma]
    )
}

def build_lemma_mask(lemma):
    mask = torch.tensor([True] * len(tags_stoi))
    for valid_tag in refs[lemma]:
        mask[tags_stoi[valid_tag].item()] = False
    return mask

lemma_masks = {
    lemma: build_lemma_mask(lemma)
    for lemma in refs_stoi
}

def overlap(a_start, a_end, b_start, b_end):
    """determine whether a and b overlap"""
    return (a_start <= b_start and a_end > b_start) \
        or (a_start < b_end and a_end >= b_end) \
        or (a_start <= b_start and a_end >= b_end) \
        or (a_start >= b_start and a_end <= b_start)

def toks_to_str(tokens):
    """return a clean str represnting the original tokens"""
    return " ".join(tokens).replace('``', '"').replace("''", '"')

# TODO: you really need to speed this up. Some possibilities:
# - batch,
# - use distilbert rather than bert,
# - move to gpu
# - downsample semcor
# This takes roughly 5h+ on CPU with BERT.
# NB: Quite possibly it would be better without my hack-in to extract factors.
def parse_semcor():
    """convert semcor into usable items"""
    all_items = []
    was_grad_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    for sentence in tqdm.tqdm(semcor.tagged_sents(tag='sem'), desc='Parsing SemCor', disable=None):
        sentence_tokens = [
            word
            for subtree in sentence
            for word in (
                subtree if type(subtree) is list else subtree.flatten()
            )
        ]
        sentence_str = toks_to_str(sentence_tokens)
        tagged = []
        offsets = []
        prev_idx = 0
        for tok in sentence:
            if type(tok) == list:
                prev_idx += 1 + len(toks_to_str(tok))
            else:
                segment_len = len(toks_to_str(tok.flatten()))
                offsets.append([prev_idx, prev_idx + segment_len])
                tagged.append(tok)
                prev_idx += 1 + segment_len
        lemmas_tagged = [
            (
                tagged_item.label().split('.', 1)[0]
                if type(tagged_item.label()) is str else
                tagged_item.label().name()
            )
            for tagged_item in tagged
        ]
        true_tags = [
            (
                tagged_item.label()
                if type(tagged_item.label()) is str else
                tagged_item.label().key()
            )
            for tagged_item in tagged
        ]
        if DROP_MONOSEMOUS:
            zipped_tagged_items = list(zip(lemmas_tagged, tagged, true_tags))
            try:
                lemmas_tagged, tagged, true_tags = zip(*filter(lambda t: t[0] in refs, zipped_tagged_items))
            except ValueError as ve:
                assert ve.args[0] == 'not enough values to unpack (expected 3, got 0)'
                lemmas_tagged, tagged, true_tags = [], [], []
        if len(tagged) == 0:
            continue
        embedded_toks = linear_structure.read_factors_last_layer(sentence_str)
        for tagged_item, offset, lemma, true_tag in zip(tagged, offsets, lemmas_tagged, true_tags):
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
                'ipt': sum(r['ipt'] for r in relevant_embs).cpu().detach(),
                'norm': sum(r['norm'] for r in relevant_embs).cpu().detach(),
                'ff': sum(r['ff'] for r in relevant_embs).cpu().detach(),
                'mha': sum(r['mha'] for r in relevant_embs).cpu().detach(),
                'lemma': lemma,
                'tag': true_tag
            }
            all_items.append(dict_item)
    torch.set_grad_enabled(was_grad_enabled)
    return all_items


class BertFactorSemcorDataset(Dataset):
    """torch object for convenience"""
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
        """read semcor and produce subsplits"""
        items = parse_semcor()
        random.shuffle(items)
        train_ = (len(items) * 8) // 10
        dev_ = (len(items) * 9) // 10
        train = items[:train_]
        dev = items[train_:dev_]
        test = items[dev_:]
        return cls(items=train), cls(items=dev), cls(items=test)

    def __getitem__(self, idx):
        base_dict = self.items[idx]
        if not 'tgt_idx' in base_dict:
            base_dict['tgt_idx'] = tags_stoi[base_dict['tag']]
            base_dict['tgt_idx'].requires_grad = False
            base_dict['lemma_mask'] = lemma_masks[base_dict['lemma']]
            base_dict['lemma_mask'].requires_grad = False
        return base_dict

    def __len__(self):
        return len(self.items)

CACHE_DIR = pathlib.Path('data')
if DROP_MONOSEMOUS:
    CACHE_DIR = CACHE_DIR / 'drop'
else:
    CACHE_DIR = CACHE_DIR / 'no-drop'


IPT_SIZE = 768

BATCH_SIZE = 2048
EPOCHS = 20
KEYS_TO_SUM = 'ipt', 'norm', 'mha', 'ff'
DEVICE = 'cpu'

def get_dataloaders(cache_dir=CACHE_DIR, batch_size=BATCH_SIZE):
    """get (or build) Datasets, then convert them to DataLoader"""
    try:
        train_dataset = BertFactorSemcorDataset.load(cache_dir / 'train.pt')
        dev_dataset = BertFactorSemcorDataset.load(cache_dir / 'dev.pt')
        test_dataset = BertFactorSemcorDataset.load(cache_dir / 'test.pt')
        tqdm.tqdm.write(f'Using cached dataset files at {cache_dir}.')
    except FileNotFoundError:
        tqdm.tqdm.write('Dataset files not cached, splitting from scratch.')
        cache_dir.mkdir(exist_ok=True, parents=True)
        train_dataset, dev_dataset, test_dataset = BertFactorSemcorDataset.splits()
        train_dataset.save(cache_dir / 'train.pt')
        dev_dataset.save(cache_dir / 'dev.pt')
        test_dataset.save(cache_dir / 'test.pt')
    return (
        DataLoader(train_dataset, shuffle=True, batch_size=batch_size),
        DataLoader(dev_dataset, shuffle=False, batch_size=batch_size),
        DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    )



def powerset():
    combinations = (
        itertools.combinations(KEYS_TO_SUM, r)
        for r in range(1, len(KEYS_TO_SUM) + 1)
    )
    yield from itertools.chain.from_iterable(combinations)

train, dev, test = get_dataloaders(batch_size=1)
neighborhood_by_lemma = collections.defaultdict(list)
for item in train.dataset:
    neighborhood_by_lemma[item['lemma']].append(item['tag'])
    # rep = sum(item[key] for key in keys_to_sum)
pbar = tqdm.tqdm(dev.dataset, desc="Val.", leave=False, disable=None)
all_preds = []
running_mfs, running_rand  = 0, 0
total_items = 0
for item in pbar:
    neighborhood = neighborhood_by_lemma[item['lemma']]
    try:
        pred_mfs = collections.Counter(neighborhood).most_common(1)[0]
        pred_rand = 1 / len(set(neighborhood))
    except IndexError:
        # not in train set
        pred_mfs = None, None
        pred_rand = 0
    total_items += 1
    running_mfs += pred_mfs[0] == item['tag']
    running_rand += pred_rand
    all_preds.append((pred_mfs[0], item['tag']))
    pbar.set_description(f"Valid (A={running_mfs/total_items:.4f})")
unk_removed = [p[0] == p[1] for p in all_preds if p[0] is not None]
print(f'MFS Accuracy (dev):', running_mfs/total_items, 'unk removed:', sum(unk_removed)/len(unk_removed))
print(f'Rand Accuracy (dev):', running_rand/total_items, 'unk removed:', running_rand/len(unk_removed))

pbar = tqdm.tqdm(test.dataset, desc="Val.", leave=False, disable=None)
all_preds = []
running_mfs, running_rand  = 0, 0
total_items = 0
for item in pbar:
    neighborhood = neighborhood_by_lemma[item['lemma']]
    try:
        pred_mfs = collections.Counter(neighborhood).most_common(1)[0]
        pred_rand = 1 / len(set(neighborhood))
    except IndexError:
        # not in train set
        pred_mfs = None, None
        pred_rand = 0
    total_items += 1
    running_mfs += pred_mfs[0] == item['tag']
    running_rand += pred_rand
    all_preds.append((pred_mfs[0], item['tag']))
    pbar.set_description(f"Valid (A={running_mfs/total_items:.4f})")
unk_removed = [p[0] == p[1] for p in all_preds if p[0] is not None]
print(f'MFS Accuracy (test):', running_mfs/total_items, 'unk removed:', sum(unk_removed)/len(unk_removed))
print(f'Rand Accuracy (test):', running_rand/total_items, 'unk removed:', running_rand/len(unk_removed))
