import base64
from collections import OrderedDict
from random import randrange

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from peewee import *
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

db = SqliteDatabase('2021-07-31-lichess-evaluations-37MM.db')

class Evaluations(Model):
  id = IntegerField()
  fen = TextField()
  binary = BlobField()
  eval = FloatField()

  class Meta:
    database = db

  def binary_base64(self):
    return base64.b64encode(self.binary)
  
LABEL_COUNT = 37164639

class EvaluationDataset(IterableDataset):
  def __init__(self, count, subset_count=None):
    self.count = count
    self.subset_count = subset_count or count
  def __iter__(self):
    self.seen = 0
    return self
  def __next__(self):
    if self.seen >= self.subset_count:
        raise StopIteration
    idx = randrange(self.count)
    self.seen += 1
    return self[idx]
  def __len__(self):
    return self.subset_count
  def __getitem__(self, idx):
    eval = Evaluations.get(Evaluations.id == idx+1)
    bin = np.frombuffer(eval.binary, dtype=np.uint8)
    bin = np.unpackbits(bin, axis=0).astype(np.single)
    eval.eval = max(eval.eval, -15)
    eval.eval = min(eval.eval, 15)
    ev = np.array([eval.eval]).astype(np.single)
    return {'binary':bin, 'eval':ev}

class EvaluationModel(pl.LightningModule):
  def __init__(self,learning_rate=1e-3,batch_size=1024,layer_count=4, subset_count=LABEL_COUNT):
    super().__init__()
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.subset_count = subset_count
    
    layers = []
    for i in range(layer_count-1):
      layers.append((f"linear-{i}", nn.Linear(808, 808)))
      layers.append((f"relu-{i}", nn.ReLU()))
    layers.append((f"linear-{layer_count-1}", nn.Linear(808, 1)))
    
    self.seq = nn.Sequential(OrderedDict(layers))

  def forward(self, x):
    return self.seq(x)

  def training_step(self, batch, batch_idx):
    x, y = batch['binary'], batch['eval']
    y_hat = self(x)
    loss = F.l1_loss(y_hat, y)
    self.log("train_loss", loss)
    return loss

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

  def train_dataloader(self):
    dataset = EvaluationDataset(count=LABEL_COUNT, subset_count=self.subset_count)
    return DataLoader(dataset, batch_size=self.batch_size, num_workers=0, pin_memory=True)