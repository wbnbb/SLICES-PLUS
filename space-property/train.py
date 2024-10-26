import pandas as pd
import argparse
from utils import *
import numpy as np
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler
import os
from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from dataset import SliceDataset

from sklearn.model_selection import train_test_split
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import math

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str,
                        help="name for wandb run", required=False)
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug')
    parser.add_argument('--lstm', action='store_true',
                        default=False, help='use lstm for transforming scaffold')
    parser.add_argument('--num_props', type=int, default = 1, help="number of properties to use for condition", required=False)
    parser.add_argument('--n_layer', type=int, default=8,
                        help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8,
                        help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=512,
                        help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=30,
                        help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=3,
                        help="batch size", required=False)
    parser.add_argument('--learning_rate', type=float,
                        default=3e-4, help="learning rate", required=False)
    parser.add_argument('--lstm_layers', type=int, default=0,
                        help="number of layers in lstm", required=False)
    parser.add_argument('--num_classes', type=int, default=6,
                        help="number of space systems", required=False)

    args = parser.parse_args()

    set_seed(42)


    slice_train_file = "train-formation-slices.sli"
    print("Reading slices...")
    slices_train = read_slices_from_file(slice_train_file)  
    print(slices_train[0])

    bandgap_train_file = "train-formation.sli"
    print("Reading bandgaps...")
    bandgaps_train = read_bandgap_from_file(bandgap_train_file)  
    print(bandgaps_train[0])

    space_train_file = "train-Ef-spaceid.sli"
    print("Reading Spaceid...")
    space_train = read_bandgap_from_file(space_train_file)  
    print(space_train[0])

    train_ = list(zip(slices_train, bandgaps_train, space_train))
    random.shuffle(train_)
    slices_train[:], bandgaps_train[:], space_train[:] = zip(*train_)
    
    
    slice_test_file = "test-formation-slices.sli"
    print("Reading slices...")
    slices_test = read_slices_from_file(slice_test_file)  
    print(slices_test[0])

    bandgap_test_file = "test-formation.sli"
    print("Reading bandgaps...")
    bandgaps_test = read_bandgap_from_file(bandgap_test_file)  
    print(bandgaps_test[0])

    space_test_file = "test-Ef-spaceid.sli"
    print("Reading Spaceid...")
    space_test = read_bandgap_from_file(space_test_file)  
    print(space_test[0])

    test_ = list(zip(slices_test, bandgaps_test, space_test))
    random.shuffle(test_)
    slices_test[:], bandgaps_test[:], space_test[:] = zip(*test_)
    
    
    lens = [len(sli.strip().split(' ')) for sli in (slices_train+slices_test)]
    max_len = max(lens)
    print('Max length of slices: ', max_len)  #360


    print("Constructing vocabulary...")
    voc_chars = construct_vocabulary(slices_train+slices_test)
    char_list = sorted(list(voc_chars)+['<','>'])
    vocab_size = len(char_list)
    print("vocab_size: {}\n".format(vocab_size))
    print(char_list)
    print(len(char_list))

    # prepare datasets
    device = 'cuda:0'
    train_datasets = SliceDataset(args, slices_train, char_list, max_len, bandgaps_train, space_train) # save for check_novelty
    valid_datasets = SliceDataset(args, slices_test, char_list, max_len, bandgaps_test, space_test)

    
    print(len(train_datasets),len(valid_datasets))
    
    mconf = GPTConfig(vocab_size, max_len, num_props=args.num_props,
                        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                        lstm=args.lstm, lstm_layers=args.lstm_layers, num_classes=args.num_classes)
    model = GPT(mconf)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model.load_state_dict(torch.load('./cond_gpt/weights/weights_zxb.ckpt'))
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Model architecture:")
    print(model)
    print("The number of trainable parameters is: {}".format(params))

    
    tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                            lr_decay=True, warmup_tokens=0.1*len(slices_train+slices_test)*max_len,
                            final_tokens=args.max_epochs*len(slices_train+slices_test)*max_len,num_workers=10, 
                            ckpt_path=f'./cond_gpt/weights/weights_zxb.ckpt', block_size=max_len, generate=False)

    trainer = Trainer(model, train_datasets, valid_datasets,
                        tconf, train_datasets.stoi, train_datasets.itos)

    df = trainer.train()
    #df.to_csv(f'{args.run_name}.csv', index=False)


