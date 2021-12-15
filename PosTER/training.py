from dataset import DynamicDataset, StaticDataset
from model import PosTER 
from utils import save_checkpoint, load_checkpoint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm.notebook import tqdm
import wandb



def train_epoch(train_loader, val_loader, model, optimizer, criterion, device):  
  
  model.train()
  loop = tqdm(train_loader)
  avg_loss = 0

  for batch_idx, (x , labels) in enumerate(loop):   
    x = x.to(device)
    labels = torch.tensor(labels,dtype=torch.long)
    labels = labels.to(device)
    
    out = model(x)
    loss = criterion(out, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_loss += loss

  avg_loss = avg_loss/len(train_loader)
  

  #Evaluate model
  avg_val_loss = 0
  with torch.no_grad():
    model.eval()

    for val_batch_idx, (x_val , labels_val) in enumerate(val_loader):      
      x_val = x_val.to(device)
      labels_val = torch.tensor(labels_val, dtype=torch.long).to(device)  

      out_val = model(x_val)
      loss_val = criterion(out_val, labels_val)
        
      avg_val_loss += loss_val

  avg_val_loss = avg_val_loss/len(val_loader)

  return avg_loss, avg_val_loss


if __name__ == "__main__":
    
    #All the following is meant to be improved with config file 
    ################################################################################################
    '''
    Training setup
    '''
    dim_tokens = 3 # the size of vocabulary
    dim_embed = 100  # hidden dimension
    dim_ff = 2048
    nlayers = 4  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 5  # the number of heads in the multiheadattention models
    dropout = 0.1  # the dropout value  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #Get datasets and dataloaders. Ideally dataset[0]= masked sequence, and dataset[1] = full seq
    #train_data = 
    #val_data = 
    train_loader = DataLoader(train_data, batch_size = config.BATCH_SIZE, shuffle= True )
    val_loader = DataLoader(val_data, batch_size = config.BATCH_SIZE, shuffle= True )


    model = PosTER(dim_tokens, dim_embed, dim_ff, nhead, nlayers, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = config.LR)

    #Start the run on wandb. Here the entity should be your wandb name
    wandb_entity = "tronchonleo"
    wandb.init(project=config.PROJECT_NAME, entity=wandb_entity)
    wandb.watch(model, criterion, log="all", log_freq=10)

    '''
    Training loop
    '''
    best_loss = 0
    for epoch in range(config.NUM_EPOCHS):

        #Train epoch and return losses
        loss, val_loss = train_epoch(train_loader, val_loader, model, optimizer, criterion, config.DEVICE)

        #Display results
        print(f"Loss epoch {epoch}: ", loss.item())
        print(f"Validation Loss epoch {epoch}: ", val_loss.item())

        #Log results in wandb
        wandb.log({
            "loss": loss.item(), 
            "epoch": epoch,
            "val_loss": val_loss.item()
            })
        
        #Save best model
        if (config.SAVE_CHECKPOINT and val_loss > best_loss):
            best_loss = val_loss
            save_checkpoint(model, optimizer, filename= config.CHECKPOINT_FILENAME)