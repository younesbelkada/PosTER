import json

from PosTER.Agents.main_agent import Trainer_Agent
from PosTER.Agents.utils_agent import get_dataset
from PosTER.Datasets.titan_dataset import Sequence, Frame, Person

with open('config.json', 'r') as f:
    config = json.load(f)

train_dataloader, val_dataloader = get_dataset(config)

trainer = Trainer_Agent(config).trainer
trainer.train(train_dataloader, val_dataloader)