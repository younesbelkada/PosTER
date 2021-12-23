import json

from PosTER.training import Trainer
from PosTER.utils_train import get_dataset
from PosTER.TITAN.titan_dataset import Sequence, Frame, Person

with open('config.json', 'r') as f:
    config = json.load(f)

train_dataloader, val_dataloader = get_dataset(config)

trainer = Trainer(config)
trainer.train(train_dataloader, val_dataloader)