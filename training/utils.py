import torch
import tqdm
import numpy as np
from pathlib import Path


class EarlyStopper:
    def __init__(self, model, save_path, patience=5, min_delta=0):
        self.model = model
        self.save_path = save_path
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.epochs_counter = 0
        self.best_epoch = None
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        self.epochs_counter += 1
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            print('best model!')
            torch.save(self.model.state_dict(), self.save_path)
            self.best_epoch = self.epochs_counter
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def opt_schedule(model, config, current_optimizer, drop_counter):
    # adam to sgd
    if current_optimizer == 'adam' and drop_counter == 80:
        model.load_state_dict(torch.load(config['save_path'],
                                         map_location=torch.device(config['device'])))
        config['optimizer'] = torch.optim.SGD(model.parameters(), 0.001,
                                              momentum=0.9, weight_decay=0.0001,
                                              nesterov=True)
        current_optimizer = 'sgd_1'
        drop_counter = 0
        print('sgd 1e-3')
    # first drop
    if current_optimizer == 'sgd_1' and drop_counter == 20:
        model.load_state_dict(torch.load(config['save_path'],
                                         map_location=torch.device(config['device'])))
        for pg in config['optimizer'].param_groups:
            pg['lr'] = 0.0001
        current_optimizer = 'sgd_2'
        drop_counter = 0
        print('sgd 1e-4')
    # second drop
    if current_optimizer == 'sgd_2' and drop_counter == 20:
        model.load_state_dict(torch.load(config['save_path'],
                                         map_location=torch.device(config['device'])))
        for pg in config['optimizer'].param_groups:
            pg['lr'] = 0.00001
        current_optimizer = 'sgd_3'
        print('sgd 1e-5')
    return config, current_optimizer, drop_counter


def get_warmup_lr(lr, step, max_step=1000, change_period=50):
    if step <= max_step and step % change_period == 0:
        lr = (step / max_step) * lr
    return lr


def train_one_epoch(model, train_loader, config):
    model.train()
    total_loss = 0
    for x, y in tqdm.tqdm(train_loader):
        
        if 'normalize_input' in config and config['normalize_input']:
            # normalize the input audio spectrogram so that the dataset mean 
            # and standard deviation are 0 and 0.5 respectively
            x = (x - config['norm_mean']) / (config['norm_std'] * 2)

        logits = model(x.float().to(config['device']))
        
        if config['optimizer_type'] == 'adam_lr_scheduler_warmup':
            config['global_step'] += 1
            lr = get_warmup_lr(config['LR'], config['global_step'])
            for param_group in config['optimizer'].param_groups:
                param_group['lr'] = lr
        
        # compute loss
        loss = config['loss_function'](logits, y.float().to(config['device']))
        # prepare
        config['optimizer'].zero_grad()
        # backward
        loss.backward()
        # optimizer step
        config['optimizer'].step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate_one_epoch(model, val_loader, config):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            
            if 'normalize_input' in config and config['normalize_input']:
                # normalize the input audio spectrogram so that the dataset mean 
                # and standard deviation are 0 and 0.5 respectively
                x = (x - config['norm_mean']) / (config['norm_std'] * 2)            
            
            logits = model(x.float().to(config['device']))
            loss = config['loss_function'](
                logits, y.float().to(config['device']))
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss


def train_model(model, train_loader, val_loader, config):
    print(
        f'Training started for model "{Path(config["save_path"]).parent.name}/{Path(config["save_path"]).stem}"...')

    if not config['early_stopping_patience']:
        # no early-stopping; EarlyStopper will just save the best model
        config['early_stopping_patience'] = config['epochs']
    early_stopper = EarlyStopper(
        model, config['save_path'], patience=config['early_stopping_patience'])

    if config['optimizer_type'] == 'scheduler':
        current_optimizer = 'adam'
        drop_counter = 0

    for epoch in range(config['epochs']):
        print(f'Epoch {epoch+1}/{config["epochs"]}')
        train_loss = train_one_epoch(model, train_loader, config)
        validation_loss = validate_one_epoch(model, val_loader, config)
        print(
            f'Loss at training set: {train_loss}\n\tLoss at validation set: {validation_loss}')

        if config['optimizer_type'] == 'scheduler':
            config, current_optimizer, drop_counter = opt_schedule(
                model, config, current_optimizer, drop_counter)
        
        elif config['optimizer_type'] == 'adam_lr_scheduler_warmup':
            config['lr_scheduler'].step()

        if early_stopper.early_stop(validation_loss):
            print('Early Stopping was activated.')
            print(
                f'Epoch {epoch+1}/{config["epochs"]}, Loss at training set: {train_loss}\n\tLoss at validation set: {validation_loss}')
            print('Training has been completed.\n')
            break
    
    print(f'\nDone. Epoch with the best model: {early_stopper.best_epoch}/{config["epochs"]}')
