import torch
import wandb
from tqdm.notebook import tqdm
from model import PosTER 
from dataset import DynamicDataset, StaticDataset



def train_epoch(train_loader, val_loader, model, optimizer, criterion, device, scaler):  
  
  model.train()
  loop = tqdm(train_loader)
  avg_loss = 0
  for batch_idx, (x , labels) in enumerate(loop):   
    x = x.to(device)
    labels = torch.tensor(labels,dtype=torch.long)
    labels = labels.to(device)
    
    with torch.cuda.amp.autocast():
      out = model(x)
      loss = criterion(out, labels)

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    avg_loss += loss

  avg_loss = avg_loss/len(train_loader)
  

  #Evaluate model
  avg_val_loss = 0
  with torch.no_grad():
    model.eval()

    for val_batch_idx, (x_val , labels_val) in enumerate(val_loader):      
      x_val = x_val.to(device)
      labels_val = torch.tensor(labels_val, dtype=torch.long).to(device)
      
      with torch.cuda.amp.autocast():   
        out_val = model(x_val)
        loss_val = criterion(out_val, labels_val)
        

      avg_val_loss += loss_val
  
    
 

  avg_val_loss = avg_val_loss/len(val_loader)

  return avg_loss, avg_val_loss