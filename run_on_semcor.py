import collections
import itertools
import pathlib
import random
import re


from nltk.corpus import semcor
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
import tqdm

from linear_structure import model, read_factors_last_layer

DROP_MONOSEMOUS = False

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

IPT_SIZE = 768
INNER_SIZE = 256


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
    for sentence in tqdm.tqdm(semcor.tagged_sents(tag='sem')):
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
            zipped_tagged_items = zip(lemmas_tagged, tagged, true_tags)
            lemmas_tagged, tagged, true_tags = zip(*filter(lambda t: t[0] in refs, zipped_tagged_items))
        if len(tagged) == 0:
            continue
        embedded_toks = read_factors_last_layer(sentence_str)
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

def get_dataloaders(cache_dir=pathlib.Path('data'), batch_size=256):
    """get (or build) Datasets, then convert them to DataLoader"""
    try:
        train_dataset = BertFactorSemcorDataset.load(cache_dir / 'train.pt')
        dev_dataset = BertFactorSemcorDataset.load(cache_dir / 'dev.pt')
        test_dataset = BertFactorSemcorDataset.load(cache_dir / 'test.pt')
    except FileNotFoundError:
        cache_dir.mkdir(exists_ok=True, parents=True)
        train_dataset, dev_dataset, test_dataset = BertFactorSemcorDataset.splits()
        train_dataset.save(cache_dir / 'train.pt')
        dev_dataset.save(cache_dir / 'dev.pt')
        test_dataset.save(cache_dir / 'test.pt')
    return (
        DataLoader(train_dataset, shuffle=True, batch_size=batch_size),
        DataLoader(dev_dataset, shuffle=False, batch_size=batch_size),
        DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    )

BATCH_SIZE=100
EPOCHS=10
KEYS_TO_SUM = 'ipt', 'norm', 'mha', 'ff'
DEVICE = 'cpu'

def powerset():
    combinations = (
        itertools.combinations(KEYS_TO_SUM, r)
        for r in range(len(KEYS_TO_SUM), 0, -1)
    )
    yield from itertools.chain.from_iterable(combinations)

torch.set_grad_enabled(True)

for keys_to_sum in powerset():

    print("Using as input:", " + ".join(keys_to_sum))
    train, dev, test = get_dataloaders(batch_size=BATCH_SIZE)

    print("data:",
        f"\ttrain: {len(train.dataset)}",
        f"\tdev: {len(dev.dataset)}",
        f"\ttest: {len(test.dataset)}",
        sep='\n')
    # this model is cheap enough to run on CPU in a decent amount of time
    wsd_model = nn.Sequential(
        nn.Linear(IPT_SIZE, INNER_SIZE),
        nn.ReLU(),
        nn.Linear(INNER_SIZE, len(tags_stoi))
    ).to(DEVICE)

    print("model:", wsd_model, sep='\n')

    optimizer = optim.Adam(wsd_model.parameters())
    criterion = nn.CrossEntropyLoss()
    for epoch in tqdm.trange(EPOCHS, desc="Epochs"):
        pbar = tqdm.tqdm(train, desc="Train", leave=False)
        wsd_model.train()
        losses = collections.deque(maxlen=100)
        for batch in pbar:
            optimizer.zero_grad()
            with torch.no_grad():
                all_ipts = sum(batch[key].to(DEVICE) for key in keys_to_sum)
            mlp_output = wsd_model(all_ipts)
            # using a mask means we can use the same matrix for all preds, and zero out irrelevant items
            lemma_specific_output = mlp_output.masked_fill(batch['lemma_mask'].to(DEVICE), -float('inf'))
            loss = criterion(lemma_specific_output, batch['tgt_idx'].view(-1).to(DEVICE))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_description(f"Train (L={sum(losses)/len(losses):.4f})")
        pbar.close()
        pbar = tqdm.tqdm(dev, desc="Val.", leave=False)
        wsd_model.eval()
        running_loss, total_items = 0, 0
        total_acc = 0
        with torch.no_grad():
            for batch in pbar:
                all_ipts = sum(batch[key].to(DEVICE) for key in keys_to_sum)
                mlp_output = wsd_model(all_ipts)
                lemma_specific_output = mlp_output.masked_fill(batch['lemma_mask'].to(DEVICE), -float('inf'))
                loss = F.cross_entropy(lemma_specific_output, batch['tgt_idx'].view(-1).to(DEVICE), reduction='sum')
                acc = (F.softmax(lemma_specific_output, dim=-1).argmax(dim=-1) == batch['tgt_idx'].view(-1)).sum()
                running_loss += loss.item()
                total_acc += acc.item()
                total_items += batch['tgt_idx'].numel()
                pbar.set_description(f"Valid (L={running_loss/total_items:.4f}, A={total_acc/total_items:.4f})")
        tqdm.tqdm.write(f"Epoch {epoch}, loss: {running_loss/total_items}, acc.: {total_acc/total_items}")
        pbar.close()
