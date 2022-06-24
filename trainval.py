import torch
from torch.utils.data import DataLoader

from data.torch_dataset import GlyphGeom
from data.transform import ToFixedTensor
from config.config_loader import load_config
from model.net import CNN, Deepset, GCNN, SetTransformer
from utils.metric import Metrics
from utils.pytorchtools import EarlyStopping

config = load_config()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_model(nn, load_state=False, checkpoint_name=None):
    if nn == 'cnn':
        model = CNN(config['in_channels'],
                    config['out_channels'])
    elif nn == 'deepset':
        model = Deepset(config['in_channels'],
                        config['out_channels'],
                        config['latent_dim'])
    elif nn == 'gcnn':
        model = GCNN(config['in_channels'],
                     config['out_channels'],
                     config['latent_dim'])
    elif nn == 'transformer':
        model = SetTransformer(config['in_channels'],
                               config['out_channels'],
                               config['latent_dim'])
    epoch = 1
    monitor = None
    if load_state:
        checkpoint = torch.load(config['save'] + f'{checkpoint_name}.pth')
        model.load_state_dict(checkpoint['model_state'])
        monitor = checkpoint['monitor']
        epoch = checkpoint['epoch']

    model.to(torch.device(device))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                            optimizer=optimizer,
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
    # Dataset
    training_data = GlyphGeom(data_dir=config['data_dir']+config['train_set'],
                              transform=ToFixedTensor(config['set_size']))
    val_data = GlyphGeom(data_dir=config['data_dir']+config['val_set'],
                         transform=ToFixedTensor(config['set_size']))

    train_loader = DataLoader(training_data,
                              batch_size=config['batch_size'],
                              shuffle=True)
    val_loader = DataLoader(val_data,
                            batch_size=config['batch_size'],
                            shuffle=True)
    # Build model
    model, criterion, optimizer, scheduler, monitor, epoch = build_model(
                                                        config['model'],
                                                        config['load_state'],
                                                        config['checkpoint'])
    early_stopping = EarlyStopping(path=config['save']
                                   + f'{config["checkpoint"]}.pth',
                                   best_score=monitor,
                                   patience=config['patience'],
                                   verbose=True)

    # Training loop
    if config['train']:
        EPOCH = config['epoch']
        for ep in range(epoch, EPOCH+1):
            print(f'At Epoch [{ep}/{EPOCH}]:')
            # Training & Validation
            train(train_loader, model, criterion, optimizer)
            metr, loss = validation(val_loader, model, criterion)
            scheduler.step(loss)
            # Early-stopping
            early_stopping(loss, metr.cm, model, optimizer, ep)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    else:
        validation(val_loader, model, criterion)
