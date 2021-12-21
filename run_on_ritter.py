import collections
import itertools
import functools
import math
import pathlib
import pprint
import random
import re

import more_itertools
from nltk.corpus import semcor
from nltk.corpus import wordnet as wn
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
import tqdm
import sklearn.metrics
import skopt
from transformers import AutoModel, AutoTokenizer
import linear_structure

# TUNED_MODEL_NAME = 'dslim/bert-base-NER-uncased'
# TUNED_MODEL = AutoModel.from_pretrained(TUNED_MODEL_NAME)
# TUNED_MODEL_TOK = AutoTokenizer.from_pretrained(TUNED_MODEL_NAME)

CACHE_DIR = pathlib.Path('data') / 'ner' / 'untuned'
CACHE_DIR.mkdir(exist_ok=True, parents=True)

IPT_SIZE = 768

BATCH_SIZE = 512
EPOCHS = 20
KEYS_TO_SUM = 'ipt', 'norm', 'mha', 'ff'
DEVICE = 'cuda'


def overlap(a_start, a_end, b_start, b_end):
    """determine whether a and b overlap"""
    return (a_start <= b_start and a_end > b_start) \
        or (a_start < b_end and a_end >= b_end) \
        or (a_start <= b_start and a_end >= b_end) \
        or (a_start >= b_start and a_end <= b_start)

def yield_file(file):
    with open(file) as istr:
        istr = map(str.strip, istr)
        words_acc, tags_acc = [], []
        for line in istr:
            if not line:
                sentence = " ".join(words_acc)
                tag_offsets = []
                start = -1
                for tag, word in zip(tags_acc, words_acc):
                    assert sentence[start + 1:start + 1 + len(word)] == word
                    tag_offsets.append([tag, (start + 1, start + 1 + len(word))])
                    start += 1 + len(word)
                yield sentence, tag_offsets
                words_acc, tags_acc = [], []
            else:
                word, tag = line.split('\t')
                words_acc.append(word)
                tags_acc.append(tag)


class BertFactorRiterDataset(Dataset):
    """torch object for convenience"""
    def __init__(self, file_name, tokenizer_=linear_structure.tokenizer, model_=linear_structure.model):
        self.items = []
        self.tag_vocab = set()
        for sentence, tag_offsets in tqdm.tqdm(list(yield_file(file_name)), desc='Building items', leave=False):
            embedded_toks = linear_structure.read_factors_last_layer(sentence, tokenizer_=tokenizer_, model_=model_)
            for tag, offset in tag_offsets:
                self.tag_vocab.add(tag)
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
                    'tag': tag,
                }
                self.items.append(dict_item)
        self.tag_vocab = sorted(self.tag_vocab)
        self.tag_stoi = {tag:i for i, tag in enumerate(self.tag_vocab)}
        for item in tqdm.tqdm(self.items, desc='mapping_tags', leave=False):
            item['tgt_idx'] = self.tag_stoi[item['tag']]


    def save(self, file):
        torch.save(self, file)

    @staticmethod
    def load(file):
        return torch.load(file)

    def __getitem__(self, idx):
        self.items[idx]['tgt_idx'] = self.tag_stoi[self.items[idx]['tag']]
        return self.items[idx]

    def __len__(self):
        return len(self.items)

RITTER_WNUT_SOURCE_DIR = pathlib.Path('data') / 'wnut16'

def get_dataloaders(cache_dir=CACHE_DIR, batch_size=BATCH_SIZE):
    """get (or build) Datasets, then convert them to DataLoader"""
    try:
        train_dataset = BertFactorRiterDataset.load(cache_dir / 'train.pt')
        dev_dataset = BertFactorRiterDataset.load(cache_dir / 'dev.pt')
        test_dataset = BertFactorRiterDataset.load(cache_dir / 'test.pt')
        tqdm.tqdm.write(f'Using cached dataset files at {cache_dir}.')
    except FileNotFoundError:
        tqdm.tqdm.write('Dataset files not cached, reading from scratch.')
        cache_dir.mkdir(exist_ok=True, parents=True)
        train_dataset = BertFactorRiterDataset(RITTER_WNUT_SOURCE_DIR / 'train')
        train_dataset.save(cache_dir / 'train.pt')
        dev_dataset = BertFactorRiterDataset(RITTER_WNUT_SOURCE_DIR / 'dev')
        dev_dataset.save(cache_dir / 'dev.pt')
        test_dataset = BertFactorRiterDataset(RITTER_WNUT_SOURCE_DIR / 'test')
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


MODELS_DIR = pathlib.Path('models-ner') / 'untuned'
MODELS_DIR.mkdir(exist_ok=True, parents=True)

train, dev, test = get_dataloaders(batch_size=BATCH_SIZE)
if not (train.dataset.tag_vocab == dev.dataset.tag_vocab == test.dataset.tag_vocab):
    assert all(t in train.dataset.tag_vocab for t in dev.dataset.tag_vocab)
    assert all(t in train.dataset.tag_vocab for t in test.dataset.tag_vocab)
    print('Rematching tag vocabs...')
    dev.dataset.tag_vocab = test.dataset.tag_vocab = train.dataset.tag_vocab
    dev.dataset.tag_stoi = test.dataset.tag_stoi = train.dataset.tag_stoi

for keys_to_sum in powerset():
    print("Using as input: " + " + ".join(keys_to_sum))
    MODEL_NAME = MODELS_DIR / ("_".join(keys_to_sum) + '.pt')
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
        ner_model = nn.Sequential(
            nn.Linear(IPT_SIZE, IPT_SIZE),
            nn.ReLU(),
            nn.Dropout(hparams['dropout_p']),
            nn.Linear(IPT_SIZE, len(train.dataset.tag_vocab), bias=False)
        ).to(DEVICE)

        @torch.no_grad()
        def init_output_embs_from_wn_():
            wn_views = get_or_compute_bert_wn_embs()
            pbar = tqdm.tqdm(wn_views.items(), total=len(wn_views), leave=False, desc="Init", disable=None)
            for wn_key, bert_emb in pbar:
                idx = tags_stoi[wn_key]
                ner_model[-1].weight[idx,:] = bert_emb.to(DEVICE)
                # wsd_model[-1].bias[:] = 0
            torch.nn.init.eye_(ner_model[0].weight)
            torch.nn.init.zeros_(ner_model[0].bias)

        # if hparams['init_wn']:
        #     init_output_embs_from_wn_()

        tqdm.tqdm.write("model:\n" + str(ner_model))
        max_f1 = -float('inf')
        optimizer = optim.Adam(ner_model.parameters(), betas=sorted([hparams["beta_a"], hparams["beta_b"]]), lr=hparams["lr"], weight_decay=hparams["weight_decay"])
        criterion = nn.CrossEntropyLoss()
        if hparams['use_scheduler']:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, threshold=0.005)
        for epoch in tqdm.trange(EPOCHS, desc="Epochs", leave=False, disable=None):
            pbar = tqdm.tqdm(train, desc="Train", leave=False, disable=None)
            ner_model.train()
            losses = collections.deque(maxlen=100)
            accs = collections.deque(maxlen=100)
            for batch in pbar:
                optimizer.zero_grad()
                with torch.no_grad():
                    all_ipts = sum(batch[key].to(DEVICE) for key in keys_to_sum)
                mlp_output = ner_model(all_ipts)
                tgt = batch['tgt_idx'].view(-1).to(DEVICE)
                # using a mask means we can use the same matrix for all preds, and zero out irrelevant items
                loss = criterion(mlp_output, tgt)
                acc = (F.softmax(mlp_output, dim=-1).argmax(dim=-1) ==  tgt).float().mean()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                accs.append(acc.item())
                pbar.set_description(f"Train (L={sum(losses)/len(losses):.4f}, A={sum(accs)/len(accs):.4f})")
            pbar.close()
            pbar = tqdm.tqdm(dev, desc="Val.", leave=False, disable=None)
            ner_model.eval()
            running_loss, total_items = 0, 0
            total_acc = 0
            with torch.no_grad():
                all_preds, all_true = [], []
                for batch in pbar:
                    all_ipts = sum(batch[key].to(DEVICE) for key in keys_to_sum)
                    mlp_output = ner_model(all_ipts)
                    tgt = batch['tgt_idx'].view(-1).to(DEVICE)
                    loss = F.cross_entropy(mlp_output, tgt, reduction='sum')
                    all_preds.extend(mlp_output.argmax(-1).view(-1).tolist())
                    all_true.extend(tgt.tolist())
                    acc = (F.softmax(mlp_output, dim=-1).argmax(dim=-1) == tgt).sum()
                    running_loss += loss.item()
                    total_acc += acc.item()
                    total_items += batch['tgt_idx'].numel()
                    pbar.set_description(f"Valid (L={running_loss/total_items:.4f}, A={total_acc/total_items:.4f})")
            new_f1 = sklearn.metrics.f1_score(all_true, all_preds, average='macro')
            tqdm.tqdm.write(f"Epoch {epoch}, loss: {running_loss/total_items}, acc.: {total_acc/total_items}, F1: {new_f1}")
            max_f1 = max(new_f1, max_f1)
            best_tracker.try_dump(ner_model, new_f1)
            if hparams['use_scheduler']:
                scheduler.step(new_f1)
            pbar.close()
        return -max_f1

    if (MODELS_DIR / ("_".join(keys_to_sum) + ".pkl")).is_file():
        skopt_callback = None
        previous_dump = skopt.load(MODELS_DIR / ("_".join(keys_to_sum) + ".pkl"))
        if len(previous_dump['x_iters']) == 100:
            print(f"This config ({'+'.join(keys_to_sum)}) is already done. Continuing...")
            continue

    skopt_pbar = tqdm.trange(100, position=2, leave=False, desc=f"BayesOpt ({'+'.join(keys_to_sum)})", disable=None)
    def skopt_callback(partial_result):
        skopt.dump(partial_result, MODELS_DIR / ("_".join(keys_to_sum) + ".pkl"), store_objective=False)
        skopt_pbar.update()

    full_result = skopt.gp_minimize(fit, search_space, n_calls=100, n_initial_points=10, callback=skopt_callback)
    skopt_pbar.close()
    with open('ritter-devresults.txt', 'a') as ostr:
        print('+'.join(keys_to_sum), best_tracker.best, file=ostr)

DEVICE = 'cpu'

all_preds = {}
with open('ritter-testresults-untuned.txt', 'w') as ostr:
    for keys_to_sum in powerset():
        ner_model = torch.load(MODELS_DIR / ("_".join(keys_to_sum) + '.pt'), map_location=torch.device(DEVICE))
        pbar = tqdm.tqdm(dev, desc="Test", leave=False, disable=None)
        ner_model.eval()
        running_loss, total_items = 0, 0
        total_acc = 0
        with torch.no_grad():
            all_true = []
            for batch in pbar:
                all_ipts = sum(batch[key].to(DEVICE) for key in keys_to_sum)
                mlp_output = ner_model(all_ipts)
                tgt = batch['tgt_idx'].view(-1).to(DEVICE)
                loss = F.cross_entropy(mlp_output, tgt, reduction='sum')
                all_preds[keys_to_sum].extend(mlp_output.argmax(-1).view(-1).tolist())
                all_true.extend(tgt.tolist())
                acc = (F.softmax(mlp_output, dim=-1).argmax(dim=-1) == tgt).sum()
                running_loss += loss.item()
                total_acc += acc.item()
                total_items += batch['tgt_idx'].numel()
                pbar.set_description(f"Valid (L={running_loss/total_items:.4f}, A={total_acc/total_items:.4f})")

        new_f1 = sklearn.metrics.f1_score(all_true, all_preds[keys_to_sum], average='macro')
        tqdm.tqdm.write(f"{'+'.join(keys_to_sum)}, loss: {running_loss/total_items}, acc.: {total_acc/total_items}, F1: {new_f1}")
        print('+'.join(keys_to_sum), total_acc/total_items, file=ostr)

import numpy as np
# compat with paper for figures
keys_in_order = [
    ('ipt',), ('mha',), ('ff',), ('norm',),
    ('ipt', 'mha'), ('ipt', 'ff'), ('ipt', 'norm'), ('mha', 'ff'), ('norm', 'mha'), ('norm', 'ff'),
    ('ipt', 'mha', 'ff'), ('ipt', 'norm', 'mha'),  ('ipt', 'norm', 'ff'), ('norm', 'mha', 'ff'),
    ('ipt', 'norm', 'mha', 'ff')
]
print(set(all_preds.keys()) - set(keys_in_order))
matrix_view = np.zeros((len(all_preds), len(all_preds)))
for i, keys_1 in enumerate(keys_in_order):
    for j, keys_2 in enumerate(keys_in_order):
        matrix_view[i, j] = sklearn.metrics.f1_score(all_preds[keys_1], all_preds[keys_2], average='macro')
# print(matrix_view.tolist())
np.save('f1-classif-ritter-untuned.npy', matrix_view)
