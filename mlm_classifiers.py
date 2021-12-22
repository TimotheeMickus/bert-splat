import itertools
import pathlib
import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections, tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import skopt

import linear_structure
torch.set_grad_enabled(False)

KEYS_TO_SUM = 'ipt', 'mha', 'norm', 'ff'

def powerset():
    combinations = (
        itertools.combinations(KEYS_TO_SUM, r)
        for r in range(1, len(KEYS_TO_SUM) + 1)
    )
    yield from itertools.chain.from_iterable(combinations)

first_true_idx = max(v for k,v in linear_structure.tokenizer.vocab.items() if k.startswith('[unused')) + 1

def mask_me(inputs):
    rd_sample = torch.rand(inputs.input_ids.size())
    sampled_mask = rd_sample <= 0.15
    target_ids = inputs.input_ids
    random_wordpiece = torch.randint(first_true_idx, linear_structure.tokenizer.vocab_size, inputs.input_ids.size())
    inputs['input_ids'] = inputs['input_ids'].masked_fill(rd_sample <= 0.135, 0) + random_wordpiece.masked_fill(rd_sample > 0.135, 0)
    inputs['input_ids'] = inputs['input_ids'].masked_fill(rd_sample <= 0.12, linear_structure.tokenizer.mask_token_id).detach()
    return target_ids, sampled_mask, inputs

PATH_TO_EUROPARL = "../data/europarl/europarl-sample.txt"
with open(PATH_TO_EUROPARL, 'r') as istr:
    data = map(str.strip, istr)
    data = sorted(data, key=len, reverse=True)

class MLMDataset():
    def __init__(self, items=None):
        if items:
            self.items = items
        else:
            self.items = []
            for sentence in tqdm.tqdm(data, desc='Building dataset...'):
                inputs = linear_structure.tokenizer([sentence], return_tensors="pt", truncation=True)
                target_ids, sampled_mask, inputs = mask_me(inputs)
                # targets_only = target_ids.masked_select(sampled_mask)
                _, keywords = linear_structure.run_bert_model(linear_structure.model, **inputs)
                factors = linear_structure.get_factors_last_layer(keywords)
                for idx, (is_sampled, tgt_id) in enumerate(zip(sampled_mask.view(-1), target_ids.view(-1))):
                    if is_sampled:
                        item = {
                            k: factors[k][idx] for k in factors
                        }
                        item['tgt_id'] = tgt_id.item()
                        self.items.append(item)

    def save(self, filename):
        return torch.save(self, filename)

    @staticmethod
    def load(file):
        return torch.load(file)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def split(self):
        random.shuffle(self.items)
        dev_split = (8 * len(self)) // 10
        test_split = (9 * len(self)) // 10
        train = MLMDataset(items=self.items[:dev_split])
        dev = MLMDataset(items=self.items[dev_split:test_split])
        test = MLMDataset(items=self.items[test_split:])
        return train, dev, test

import random
import joblib
import torch

BATCH_SIZE = 2048
def get_dataloaders(run):
    try:
        train = MLMDataset.load(f'data/mlm-dataset-run-{run}-train.pt')
        dev = MLMDataset.load(f'data/mlm-dataset-run-{run}-dev.pt')
        test = MLMDataset.load(f'data/mlm-dataset-run-{run}-test.pt')
    except FileNotFoundError:
        if pathlib.Path(f'data/mlm-dataset-run-{run}.pt').is_file():
            dataset = MLMDataset.load(f'data/mlm-dataset-run-{run}.pt')
        else:
            dataset = MLMDataset()
        train, dev, test = dataset.split()
        train.save(f'data/mlm-dataset-run-{run}-train.pt')
        dev.save(f'data/mlm-dataset-run-{run}-dev.pt')
        test.save(f'data/mlm-dataset-run-{run}-test.pt')
    return DataLoader(train, batch_size=BATCH_SIZE, shuffle=True), DataLoader(dev, batch_size=BATCH_SIZE), DataLoader(test, batch_size=BATCH_SIZE)

MODELS_DIR = pathlib.Path('models-mlm')
MODELS_DIR.mkdir(exist_ok=True, parents=True)

class Tracker():
    def __init__(self):
        self.best_acc = 0

search_space = [
    skopt.space.Real(1.e-5, 1.0, "log-uniform", name="lr"),
    skopt.space.Real(.9, 1. - 1.e-3, "log-uniform", name="beta_a"),
    skopt.space.Real(.9, 1. - 1.e-3, "log-uniform", name="beta_b"),
    skopt.space.Real(0., 1., "uniform", name="weight_decay"),
    skopt.space.Real(0., .5, "uniform", name="dropout_p"),
]

run = 1
train, dev, test = get_dataloaders(run)

for keys in tqdm.tqdm(list(powerset()), desc='terms'):
    tracker = Tracker()
    @skopt.utils.use_named_args(search_space)
    def fit(**hparams):
        tqdm.tqdm.write(f'fit with: {pprint.pformat(hparams)}')
        torch_model = nn.Sequential(
            nn.Dropout(hparams['dropout_p']),
            nn.Linear(768, linear_structure.tokenizer.vocab_size)
        ).to('cuda')
        optimizer = optim.AdamW(
            torch_model.parameters(),
            lr=hparams['lr'],
            betas=sorted([hparams["beta_a"], hparams["beta_b"]]),
            weight_decay=hparams['weight_decay'],
        )
        criterion = nn.CrossEntropyLoss()
        losses = collections.deque(maxlen=100)
        accs = collections.deque(maxlen=100)
        best_acc = 0
        for EPOCH in tqdm.trange(20, position=2, desc='Epochs', leave=False):
            with torch.set_grad_enabled(True):
                pbar = tqdm.tqdm(train, position=3, desc="Train", leave=False)
                for batch in pbar:
                    optimizer.zero_grad()
                    reps = sum(batch[k].to('cuda') for k in keys).detach()
                    raw_logits = torch_model(reps)
                    loss = criterion(raw_logits, batch['tgt_id'].to('cuda'))
                    loss.backward()
                    losses.append(loss.item())
                    optimizer.step()
                    acc = (raw_logits.argmax(-1) == batch['tgt_id'].to('cuda')).float().mean().item()
                    accs.append(acc)
                    pbar.set_description(f"Train L={sum(losses)/len(losses):.4f} A={sum(accs)/len(accs):.4f}")
            with torch.set_grad_enabled(False):
                val_accs, val_losses = [], []
                pbar = tqdm.tqdm(dev, position=3, desc="Val", leave=False)
                for batch in pbar:
                    reps = sum(batch[k].to('cuda') for k in keys).detach()
                    raw_logits = torch_model(reps)
                    loss = criterion(raw_logits, batch['tgt_id'].to('cuda'))
                    val_losses.append(loss.item())
                    acc = (raw_logits.argmax(-1) == batch['tgt_id'].to('cuda')).float().mean().item()
                    val_accs.append(acc)
                    pbar.set_description(f"Val L={sum(val_losses)/len(val_losses):.4f} A={sum(val_accs)/len(val_accs):.4f}")
                tqdm.tqdm.write(f"[Run {run} epoch {EPOCH}] model `{'+'.join(keys)}`: L={sum(val_losses)/len(val_losses)} A={sum(val_accs)/len(val_accs)}")
                if (sum(val_accs) / len(val_accs)) > tracker.best_acc:
                    tqdm.tqdm.write('Dumping model.')
                    tracker.best_acc = sum(val_accs) / len(val_accs)
                    best_acc = tracker.best_acc
                    torch.save(torch_model, MODELS_DIR / ('+'.join(keys) + f'.run-{run}.pt'))
        return -best_acc

    skopt_pbar = tqdm.trange(50, position=1, leave=False, desc=f"BayesOpt ({'+'.join(keys)})", disable=None)
    def skopt_callback(partial_result):
        skopt_pbar.update()
    skopt.gp_minimize(fit, search_space, n_calls=50, n_initial_points=10, callback=skopt_callback)
    skopt_pbar.close()
