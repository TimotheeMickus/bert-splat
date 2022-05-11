import collections
import itertools
import json
import functools
import math
import pathlib
import pprint
import random
import re

import more_itertools
from nltk.corpus import semcor
from nltk.corpus import wordnet as wn
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
import tqdm
import sklearn.metrics
import skopt

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

def build_or_get_refs(REFS_FILENAME="refs-semcor.json"):
    if pathlib.Path(REFS_FILENAME).is_file():
        with open(REFS_FILENAME, 'r') as istr:
            return json.load(istr)
    else:
        refs = build_refs()
        with open(REFS_FILENAME, 'w') as ostr:
            json.dump(refs, ostr)
        return refs

refs = build_or_get_refs()
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
    import linear_structure
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
DEVICE = 'cuda'

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

@torch.no_grad()
def get_or_compute_bert_wn_embs(cache_dir=CACHE_DIR):
    try:
        cached_bert_embs = torch.load(CACHE_DIR / 'bert_wn_defs.pt')
        tqdm.tqdm.write('using cached BERT WN embs')
        return cached_bert_embs
    except FileNotFoundError:
        tqdm.tqdm.write('computing BERT WN embs, this might take a while...')
        import linear_structure
        wn_key_to_bert_emb = {}
        definitions = []
        for wn_key in tqdm.tqdm(tags_stoi.keys(), leave=False, desc="Init", disable=None):
            try:
                wn_def = wn.lemma_from_key(wn_key).synset().definition()
                definitions.append([wn_key, wn_def])
            except ValueError as ve:
                assert ve.args[0] == 'not enough values to unpack (expected 2, got 1)'
                # ok, whatever

        tqdm.tqdm.write(f"found {len(definitions)} / {len(tags_stoi)} synsets ({len(definitions)/len(tags_stoi)*100:.2f}%)")
        bert_batched = more_itertools.chunked(definitions, BATCH_SIZE)
        for batch in tqdm.tqdm(bert_batched, total=math.ceil(len(definitions) / BATCH_SIZE), leave=False, desc="Init", disable=None):
            batch_keys, batch_defs = zip(*batch)
            inputs = linear_structure.tokenizer(list(batch_defs), return_tensors='pt', truncation=True, padding=True)
            embs = linear_structure.model(**inputs).last_hidden_state.sum(1)
            for wn_key, bert_emb in zip(batch_keys, embs):
                wn_key_to_bert_emb[wn_key] = bert_emb
        torch.save(wn_key_to_bert_emb, CACHE_DIR / 'bert_wn_defs.pt')
        return wn_key_to_bert_emb

torch.set_grad_enabled(True)

class BestTracker():
    def __init__(self, model_name):
        self.best = -float('inf')
        self.model_name = model_name

    def try_dump(self, model, acc):
        if acc > self.best:
            tqdm.tqdm.write('dumping best model')
            torch.save(model, self.model_name)
            self.best = acc


MODELS_DIR_BASE = pathlib.Path('models-wsd')
MODELS_DIR_BASE.mkdir(exist_ok=True, parents=True)

train, dev, test = get_dataloaders(batch_size=BATCH_SIZE)


for RUN in tqdm.trange(1,6, desc="runs", position=0):
    MODELS_DIR = MODELS_DIR_BASE / str(RUN)
    for keys_to_sum in tqdm.tqdm(powerset(), total=15, desc="terms", position=1):
        tqdm.tqdm.write("Using as input: " + " + ".join(keys_to_sum))
        MODEL_NAME = MODELS_DIR / ("_".join(keys_to_sum) + f'.{RUN}.pt')
        best_tracker = BestTracker(MODEL_NAME)

        tqdm.tqdm.write("data:\n"+
            f"\ttrain: {len(train.dataset)}\n"+
            f"\tdev: {len(dev.dataset)}\n"+
            f"\ttest: {len(test.dataset)}")
        # this model is cheap enough to run on CPU in a decent amount of time
        search_space = [
            skopt.space.Real(0., .5, "uniform", name="dropout_p"),
            skopt.space.Real(1.e-5, 1.0, "log-uniform", name="lr"),
            skopt.space.Real(.9, 1. - 1.e-3, "log-uniform", name="beta_a"),
            skopt.space.Real(.9, 1. - 1.e-3, "log-uniform", name="beta_b"),
            skopt.space.Real(0., 1., "uniform", name="weight_decay"),
            # skopt.space.Categorical([True, False], name="init_wn"),
            skopt.space.Categorical([True, False], name="use_scheduler"),
        ]

        @skopt.utils.use_named_args(search_space)
        def fit(**hparams):
            tqdm.tqdm.write("current fit:\n" + pprint.pformat(hparams))
            wsd_model = nn.Sequential(
                nn.Linear(IPT_SIZE, IPT_SIZE),
                nn.ReLU(),
                nn.Dropout(hparams['dropout_p']),
                nn.Linear(IPT_SIZE, len(tags_stoi), bias=False)
            ).to(DEVICE)

            @torch.no_grad()
            def init_output_embs_from_wn_():
                wn_views = get_or_compute_bert_wn_embs()
                pbar = tqdm.tqdm(wn_views.items(), total=len(wn_views), leave=False, desc="Init", disable=None)
                for wn_key, bert_emb in pbar:
                    idx = tags_stoi[wn_key]
                    wsd_model[-1].weight[idx,:] = bert_emb.to(DEVICE)
                    # wsd_model[-1].bias[:] = 0
                torch.nn.init.eye_(wsd_model[0].weight)
                torch.nn.init.zeros_(wsd_model[0].bias)

            # if hparams['init_wn']:
            #     init_output_embs_from_wn_()

            tqdm.tqdm.write("model:\n" + str(wsd_model))
            max_acc = -float('inf')
            optimizer = optim.Adam(wsd_model.parameters(), betas=sorted([hparams["beta_a"], hparams["beta_b"]]), lr=hparams["lr"], weight_decay=hparams["weight_decay"])
            criterion = nn.CrossEntropyLoss()
            if hparams['use_scheduler']:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, threshold=0.005)
            for epoch in tqdm.trange(EPOCHS, desc="Epochs", leave=False, disable=None, position=3):
                pbar = tqdm.tqdm(train, desc="Train", leave=False, disable=None, position=4)
                wsd_model.train()
                losses = collections.deque(maxlen=100)
                accs = collections.deque(maxlen=100)
                for batch in pbar:
                    optimizer.zero_grad()
                    with torch.no_grad():
                        all_ipts = sum(batch[key].to(DEVICE) for key in keys_to_sum)
                    mlp_output = wsd_model(all_ipts)
                    tgt = batch['tgt_idx'].view(-1).to(DEVICE)
                    # using a mask means we can use the same matrix for all preds, and zero out irrelevant items
                    lemma_specific_output = mlp_output.masked_fill(batch['lemma_mask'].to(DEVICE), -float('inf'))
                    loss = criterion(lemma_specific_output, tgt)
                    acc = (F.softmax(lemma_specific_output, dim=-1).argmax(dim=-1) ==  tgt).float().mean()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                    accs.append(acc.item())
                    pbar.set_description(f"Train (L={sum(losses)/len(losses):.4f}, A={sum(accs)/len(accs):.4f})")
                pbar.close()
                pbar = tqdm.tqdm(dev, desc="Val.", leave=False, disable=None)
                wsd_model.eval()
                running_loss, total_items = 0, 0
                total_acc = 0
                with torch.no_grad():
                    for batch in pbar:
                        all_ipts = sum(batch[key].to(DEVICE) for key in keys_to_sum)
                        mlp_output = wsd_model(all_ipts)
                        tgt = batch['tgt_idx'].view(-1).to(DEVICE)
                        lemma_specific_output = mlp_output.masked_fill(batch['lemma_mask'].to(DEVICE), -float('inf'))
                        loss = F.cross_entropy(lemma_specific_output, tgt, reduction='sum')
                        acc = (F.softmax(lemma_specific_output, dim=-1).argmax(dim=-1) == tgt).sum()
                        running_loss += loss.item()
                        total_acc += acc.item()
                        total_items += batch['tgt_idx'].numel()
                        pbar.set_description(f"Valid (L={running_loss/total_items:.4f}, A={total_acc/total_items:.4f})")
                tqdm.tqdm.write(f"Epoch {epoch}, loss: {running_loss/total_items}, acc.: {total_acc/total_items}")
                max_acc = max(max_acc, total_acc/total_items)
                best_tracker.try_dump(wsd_model, total_acc/total_items)
                if hparams['use_scheduler']:
                    scheduler.step(running_loss/total_items)
                pbar.close()
            return -max_acc

        if (MODELS_DIR / ("_".join(keys_to_sum) + f".{RUN}.pkl")).is_file():
            skopt_callback = None
            previous_dump = skopt.load(MODELS_DIR / ("_".join(keys_to_sum) + f".{RUN}.pkl"))
            if len(previous_dump['x_iters']) == 100:
                tqdm.tqdm.write(f"This config ({'+'.join(keys_to_sum)}) is already done. Continuing...")
                continue

        skopt_pbar = tqdm.trange(100, position=2, leave=False, desc=f"BayesOpt ({'+'.join(keys_to_sum)})", disable=None)
        def skopt_callback(partial_result):
            skopt.dump(partial_result, MODELS_DIR / ("_".join(keys_to_sum) + f".{RUN}.pkl"), store_objective=False)
            skopt_pbar.update()

        full_result = skopt.gp_minimize(fit, search_space, n_calls=100, n_initial_points=10, callback=skopt_callback)
        skopt_pbar.close()
        with open(f'semcor-devresults.{RUN}.txt', 'a') as ostr:
            print('+'.join(keys_to_sum), best_tracker.best, file=ostr)

orig_device = DEVICE
DEVICE = 'cpu'

for RUN in tqdm.trange(1,6, desc="runs", position=0):
    MODELS_DIR = MODELS_DIR_BASE / str(RUN)
    all_preds = {}
    with open(f'semcor-testresults.{RUN}.txt', 'w') as ostr:
        for keys_to_sum in powerset():
            wsd_model = torch.load(MODELS_DIR / ("_".join(keys_to_sum) + f'.{RUN}.pt'), map_location=torch.device(DEVICE))
            pbar = tqdm.tqdm(test, desc="Test", leave=False, disable=None)
            wsd_model.eval()
            running_loss, total_items = 0, 0
            total_acc = 0
            all_preds[keys_to_sum] = []
            all_targets = []
            with torch.no_grad():
                for batch in pbar:
                    all_ipts = sum(batch[key].to(DEVICE) for key in keys_to_sum)
                    mlp_output = wsd_model(all_ipts)
                    tgt = batch['tgt_idx'].view(-1).to(DEVICE)
                    all_targets.extend(tgt.tolist())
                    lemma_specific_output = mlp_output.masked_fill(batch['lemma_mask'].to(DEVICE), -float('inf'))
                    loss = F.cross_entropy(lemma_specific_output, tgt, reduction='sum')
                    lemma_preds = F.softmax(lemma_specific_output, dim=-1).argmax(dim=-1).view(-1)
                    all_preds[keys_to_sum].extend(lemma_preds.detach().tolist())
                    acc = (lemma_preds == tgt).sum()
                    running_loss += loss.item()
                    total_acc += acc.item()
                    total_items += batch['tgt_idx'].numel()
                    pbar.set_description(f"Valid (L={running_loss/total_items:.4f}, A={total_acc/total_items:.4f})")
            tqdm.tqdm.write(f"Model {'+'.join(keys_to_sum)}, loss: {running_loss/total_items}, acc.: {total_acc/total_items}, f1 (macro): {sklearn.metrics.f1_score(all_targets, all_preds[keys_to_sum], average='macro')}, f1 (micro): {sklearn.metrics.f1_score(all_targets, all_preds[keys_to_sum], average='micro')}")
            print('+'.join(keys_to_sum), total_acc/total_items, file=ostr)

    import numpy as np
    # compat with paper for figures
    keys_in_order = [
        ('ipt',), ('mha',), ('ff',), ('norm',),
        ('ipt', 'mha'), ('ipt', 'ff'), ('ipt', 'norm'), ('mha', 'ff'), ('norm', 'mha'), ('norm', 'ff'),
        ('ipt', 'mha', 'ff'), ('ipt', 'norm', 'mha'),  ('ipt', 'norm', 'ff'), ('norm', 'mha', 'ff'),
        ('ipt', 'norm', 'mha', 'ff')
    ]
    matrix_view = np.zeros((len(all_preds), len(all_preds)))
    for i, keys_1 in enumerate(keys_in_order):
        for j, keys_2 in enumerate(keys_in_order):
            matrix_view[i, j] = sklearn.metrics.f1_score(all_preds[keys_1], all_preds[keys_2], average='micro')
    # print(matrix_view.tolist())
    np.save(f'f1-classif-semcor.{RUN}.npy', matrix_view)
    DEVICE = orig_device
