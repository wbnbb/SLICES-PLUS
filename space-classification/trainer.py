"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler

from utils import *
import re
import pandas as pd


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, config, stoi, itos):
        self.model = model
        self.train_dataset = train_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        self.stoi = stoi
        self.itos = itos

        if torch.cuda.is_available():
            self.device = 'cuda:0'
            # self.model = torch.nn.DataParallel(self.model).to(self.device)
            self.model = self.model.to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        scaler = GradScaler()

        def run_epoch():
            model.train()
            loader = DataLoader(self.train_dataset, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader))
            for it, (x, p) in pbar:
                # place data on the correct device
                x = x.to(self.device)
                p = p.to(self.device)

                # forward the model
                with torch.cuda.amp.autocast():
                    logits, loss, _, _ = model(x, p)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                # backprop and update the parameters
                model.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                scaler.step(optimizer)
                scaler.update()

                # decay the learning rate based on our progress
                if config.lr_decay:
                    self.tokens += (p >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                    if self.tokens < config.warmup_tokens:
                        # linear warmup
                        lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                    else:
                        # cosine learning rate decay
                        progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = config.learning_rate * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = config.learning_rate

                # report progress
                pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            return float(np.mean(losses))

        best_loss = float('inf')
        self.tokens = 0  # counter used for learning rate decay
        train_losses = []

        for epoch in range(config.max_epochs):
            train_loss = run_epoch()
            train_losses.append(train_loss)  # record the training loss
            print(f'Epoch {epoch + 1} completed with train loss {train_loss:.5f}')

            # Save the model if the current epoch's loss is the best we've seen so far
            if train_loss < best_loss:
                best_loss = train_loss
                print(f'Saving model at epoch {epoch + 1} with train loss {best_loss:.5f}')
                self.save_checkpoint()

        with open('loss_epoch.txt', 'w') as f:
            for epoch, loss in enumerate(train_losses, 1):
                f.write(f'Epoch {epoch}: {loss:.5f}\n')

        # Plot the training loss
        plt.figure()
        plt.plot(range(1, config.max_epochs + 1), train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss vs. Epoch')
        plt.legend()
        plt.savefig('loss_curves.png')
        plt.show()

        return None
