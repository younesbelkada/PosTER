import json
from torch.utils.data import Dataset, DataLoader

from PosTER.training import Trainer
from PosTER.dataset import DynamicDataset, StaticDataset, my_collate

with open('config.json', 'r') as f:
    config = json.load(f)

train_data = StaticDataset(config, 'train')
train_dataloader = DataLoader(train_data, batch_size=config['Training']['batch_size'], collate_fn=my_collate)

val_data = StaticDataset(config, 'val')
val_dataloader = DataLoader(val_data, batch_size=config['Training']['batch_size'], collate_fn=my_collate)

trainer = Trainer(config)
trainer.train(train_dataloader, val_dataloader)