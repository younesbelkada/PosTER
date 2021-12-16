import json
from torch.utils.data import Dataset, DataLoader

from PosTER.training import Trainer
from PosTER.dataset import DynamicDataset, StaticDataset, my_collate

with open('config.json', 'r') as f:
    config = json.load(f)

data = StaticDataset(config)
dataloader = DataLoader(data, batch_size=2, collate_fn=my_collate)

trainer = Trainer(config)
trainer.train(dataloader, dataloader)