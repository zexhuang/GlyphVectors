import torch
from torch.utils.data import DataLoader

from data.torch_dataset import LetterVectors
from data.transform import ToFixedTensor
from config.config_loader import load_config
from model.net import CNN, Deepset
from utils.metric import Metrics
from utils.pytorchtools import EarlyStopping

config = load_config()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_model(nn, load_state=False, name=None):
    model = nn
    epoch = 1
    monitor = None
    
    if load_state:
        checkpoint=torch.load(config['save'] + f'{name}.pth')
        model.load_state_dict(checkpoint['model_state'])
        monitor = checkpoint['monitor']
        epoch = checkpoint['epoch']

    model.to(torch.device(device))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           mode='min',
                                                           factor=config['gamma'],
                                                           patience=config['step'],
                                                           verbose=True)
    return model, criterion, optimizer, scheduler, monitor, epoch


def train(loader, model, criterion, optimizer):
    metr = Metrics(config['out_channels'])
    model.train()
    for idx, data in enumerate(loader):
        optimizer.zero_grad()         # Clear gradients

        x = data['geom'].to(torch.device(device), dtype=torch.float)
        y = data['value'].to(torch.device(device))
        
        logits = model(x)             # Feedforward
        loss = criterion(logits, y)   # Compute gradients

        loss.backward()               # Backward pass
        optimizer.step()              # Update model parameters
        metr.update_cm(logits, y)     # Update confusion matrix

        print(f"Batch: {idx + 1}/{len(loader)}:\n"
              f"Training Loss: {loss:.4f} \n"
              f"Training Acc: {metr.accuracy():.4f} \n"
              f"{'-' * 80}")


@torch.no_grad()
def validation(loader, model, criterion):
    metr = Metrics(config['out_channels'])
    model.eval()
    for idx, data in enumerate(loader):
        x = data['geom'].to(torch.device(device), dtype=torch.float)
        y = data['value'].to(torch.device(device))

        logits = model(x)
        loss = criterion(logits, y)

        metr.update_cm(logits, y)
        metr.update_loss(loss)
    
    print(f"Avg Valid Loss: {metr.avg_loss():.4f} \n"
          f"Overall Valid Acc: {metr.accuracy():.4f} \n"
          f"{'-' * 80}") 
    return metr, metr.avg_loss()


if __name__ == "__main__": 
    training_data = LetterVectors(root_dir=config['data_folder']
                                  +config['train_set'], 
                                  transform=ToFixedTensor(config['embedding']))
    val_data = LetterVectors(root_dir=config['data_folder']
                             +config['val_set'], 
                             transform=ToFixedTensor(config['embedding']))
        
    train_loader = DataLoader(training_data, 
                              batch_size=config['batch_size'], 
                              shuffle=True)
    val_loader = DataLoader(val_data, 
                            batch_size=config['batch_size'], 
                            shuffle=True)
    # build model
    model, criterion, \
    optimizer, scheduler, \
    monitor, epoch = build_model(Deepset(in_channels=config['in_channels'], 
                                         out_channels=config['out_channels'],
                                         embedding=config['latent_space']),
                                         config['load_state'],
                                         config['checkpoint'])
    
    early_stopping = EarlyStopping(path=config['save']
                                   +f'{config["checkpoint"]}.pth', 
                                   best_score=monitor,
                                   patience=config['patience'], 
                                   verbose=True)
    # training loop
    EPOCH = config['epoch']
    for epoch in range(1, EPOCH+1):
        if config['train']:
            print(f'At Epoch [{epoch}/{EPOCH}]:')
            # Training & Validation
            train(train_loader, model, criterion, optimizer)
            metr, loss = validation(val_loader, model, criterion)
            scheduler.step(loss)
            # Early-stopping    
            early_stopping(loss, metr.cm, model, optimizer, epoch)
            if early_stopping.early_stop:
                print("Early stopping")
                break 
        else:
            validation(val_loader, model, criterion)
            break