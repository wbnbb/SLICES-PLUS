# -*- coding: utf-8 -*-
from utils import *
from dataset import SliceDataset
from model import GPT, GPTConfig
import math
from tqdm import tqdm
import argparse
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import json
import os
import sys
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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import math
#from invcryrep.invcryrep import InvCryRep

# python generate.py --model_weight bandgap_reverse.pt --data_name bandgap --csv_name slices --gen_size 128 --batch_size 128 --vocab_size 97 --block_size 360 --n_props 1

# python generate.py --model_weight fenegy_reverse_Aug100.pt --data_name fenergy --csv_name slices --gen_size 30000 --batch_size 128 --vocab_size 97 --block_size 360 --n_props 1


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_weight', type=str, help="path of model weights", required=True)
        parser.add_argument('--lstm', action='store_true', default=False, help='use lstm for transforming scaffold')
        #parser.add_argument('--csv_name', type=str, help="name to save the generated mols in csv format", required=True)
        parser.add_argument('--data_name', type=str, default = 'bandgap', help="name of the dataset to train on", required=False)
        parser.add_argument('--batch_size', type=int, default = 8, help="batch size", required=False)
        parser.add_argument('--gen_size', type=int, default = 100, help="number of times to generate from a batch", required=False)
        parser.add_argument('--vocab_size', type=int, default = 114, help="size of vocabs", required=False)  # 97 including padding code "<"
        parser.add_argument('--block_size', type=int, default = 1728, help="size of slices", required=False)   # 360 for slices with atom number smaller than 10.
        parser.add_argument('--debug', action='store_true',
                        default=False, help='debug')
        parser.add_argument('--n_layer', type=int, default = 8, help="number of layers", required=False)
        parser.add_argument('--n_head', type=int, default = 8, help="number of heads", required=False)
        parser.add_argument('--n_embd', type=int, default = 256, help="embedding dimension", required=False)
        parser.add_argument('--n_props', type=int, default = 0, help="dimension of desired properties", required=False)
        parser.add_argument('--lstm_layers', type=int, default = 2, help="number of layers in lstm", required=False)

        args = parser.parse_args()


        nprops = args.n_props
        
        
        slice_test_file = "../1_data/test_slice.sli"
        print("Reading slices...")
        slices_test = read_slices_from_file(slice_test_file)  
        print(slices_test[0])
    
        bandgap_test_file = "../1_data/test_space_id.sli"
        print("Reading bandgaps...")
        bandgaps_test = read_bandgap_from_file(bandgap_test_file)  
        print(bandgaps_test[0])
    
    
        test_ = list(zip(slices_test, bandgaps_test))
        random.shuffle(test_)
        slices_test[:], bandgaps_test[:] = zip(*test_)


        char_list = sorted(set(read_vocab(fname="Voc_prior")+['<']))
        stoi = { ch:i for i,ch in enumerate(char_list) }
        itos = { i:ch for i,ch in enumerate(char_list) }
        
        #lens = [len(sli.strip().split(' ')) for sli in (slices_test)]
        #max_len = max(lens)
        valid_datasets = SliceDataset(args, slices_test, char_list, args.block_size, bandgaps_test)###
        

        # ÅäÖÃÄ£ÐÍ
        mconf = GPTConfig(args.vocab_size, args.block_size, num_props=args.n_props,
                          n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                          lstm=args.lstm, lstm_layers=args.lstm_layers)
        model = GPT(mconf)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print('Loading model')
        model.load_state_dict(torch.load('./cond_gpt/weights/' + args.model_weight, map_location=torch.device(device)))
        model.to(device)
        print('Model loaded')
        
        model.eval()
        loader = DataLoader(valid_datasets, shuffle=False, pin_memory=True,
                            batch_size=args.batch_size, num_workers=10)
        
        # ³õÊ¼»¯¼ÆÊýÆ÷
        correct_counts = {0: 0, 1: 0, 2: 0}
        label_counts = {0: 0, 1: 0, 2: 0}
        
        total = 0
        correct = 0
        
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for x, y in tqdm(loader):
                x = x.to(device)
                y = y.to(device)
                logits, _, _, _ = model(x, y)
                
                probabilities = F.softmax(logits, dim=-1)
                _, predicted = torch.max(probabilities, 2)  # [8, 720, 3] -> [8, 720]
                
                predicted = predicted.mode(dim=1).values  # [8, 720] -> [8]
                predicted = predicted.unsqueeze(1)  # [8] -> [8, 1]
                
                print(f"Shape of logits: {logits.shape}")
                print(f"Shape of y: {y.shape}")
                print(f"Shape of predicted: {predicted.shape}")
                
                total += y.size(0)
                correct += (predicted == y).sum().item()
                
                all_labels.extend(y.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                
                for label, prediction in zip(y, predicted):
                    label_counts[label.item()] += 1
                    if label.item() == prediction.item():
                        correct_counts[label.item()] += 1
        
        accuracy = correct / total
        print(f'Test Accuracy: {accuracy * 100:.2f}%')
        
        label_accuracies = {label: correct_counts[label] / label_counts[label] if label_counts[label] > 0 else 0 
                            for label in label_counts}
        
        for label, accuracy in label_accuracies.items():
            print(f'Label {label} accuracy: {accuracy * 100:.2f}%')
        
        with open('label_accuracies_1.txt', 'w') as f:
            f.write(f'Test Accuracy: {accuracy * 100:.2f}%\n')
            for label, accuracy in label_accuracies.items():
                f.write(f'Label {label} accuracy: {accuracy * 100:.2f}%\n')
        
        # Éú³É»ìÏý¾ØÕó
        cm = confusion_matrix(all_labels, all_predictions)
        
        # ½«»ìÏý¾ØÕó×ª»»Îª¸ÅÂÊ¾ØÕó
        cm_prob = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # »æÖÆ»ìÏý¾ØÕó
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_prob, annot=True, fmt='.2f', cmap='Blue', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2], annot_kws={"size": 14})
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix (Probabilities)')

        # ±£´æ»ìÏý¾ØÕóÍ¼Æ¬
        plt.savefig('Mix_picture_1.png')

        # ÏÔÊ¾»ìÏý¾ØÕó
        plt.show()
                        
                        
