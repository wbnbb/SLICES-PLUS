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
#from invcryrep.invcryrep import InvCryRep

# python generate.py --model_weight bandgap_reverse.pt --data_name bandgap --csv_name slices --gen_size 128 --batch_size 128 --vocab_size 97 --block_size 360 --n_props 1

# python generate.py --model_weight fenegy_reverse_Aug100.pt --data_name fenergy --csv_name slices --gen_size 30000 --batch_size 128 --vocab_size 97 --block_size 360 --n_props 1



if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_weight', type=str, help="path of model weights", required=True)
        parser.add_argument('--lstm', action='store_true', default=False, help='use lstm for transforming scaffold')
        parser.add_argument('--csv_name', type=str, help="name to save the generated mols in csv format", required=True)
        parser.add_argument('--data_name', type=str, default = 'bandgap', help="name of the dataset to train on", required=False)
        parser.add_argument('--batch_size', type=int, default = 5, help="batch size", required=False)
        parser.add_argument('--gen_size', type=int, default = 800, help="number of times to generate from a batch", required=False)
        parser.add_argument('--vocab_size', type=int, default = 122, help="size of vocabs", required=False)  # 97 including padding code "<"
        parser.add_argument('--block_size', type=int, default = 1728, help="size of slices", required=False)   # 360 for slices with atom number smaller than 10.
        parser.add_argument('--n_layer', type=int, default = 8, help="number of layers", required=False)
        parser.add_argument('--n_head', type=int, default = 8, help="number of heads", required=False)
        parser.add_argument('--n_embd', type=int, default = 512, help="embedding dimension", required=False)
        parser.add_argument('--n_props', type=int, default = 1, help="dimension of desired properties", required=False)
        parser.add_argument('--lstm_layers', type=int, default = 2, help="number of layers in lstm", required=False)
        parser.add_argument('--num_classes', type=int, default = 6, help="number of space systems", required=False)

        args = parser.parse_args()

        nprops = args.n_props
         
        char_list = sorted(set(read_vocab(fname="Voc_prior")+['<','>']))
        stoi = { ch:i for i,ch in enumerate(char_list) }
        itos = { i:ch for i,ch in enumerate(char_list) }

        # load model
        mconf = GPTConfig(args.vocab_size, args.block_size, num_props = nprops,
                       n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                       lstm = args.lstm, lstm_layers = args.lstm_layers, num_classes = args.num_classes)
        model = GPT(mconf)
        
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'



        print('Loading model')
        model.load_state_dict(torch.load('./cond_gpt/weights/' + args.model_weight, map_location=torch.device(device)))
        model.to(device) #model.to('cuda')
        print('Model loaded')

        prop_condition = [-0.5] # desired band gap
        crystal_system = 0  #Ŀ�꾧ϵ
       
        gen_iter = math.ceil(args.gen_size / args.batch_size)
      
        all_dfs = []
        all_metrics = []

        #count = 0
        model.eval()
        if (prop_condition is not None):
            #count = 0
            for c in prop_condition:
                slices = []
                #count += 1
                for i in tqdm(range(gen_iter)):
                        context = '>'
                        x = torch.tensor([stoi[context]], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to(device)
                        p = None
                        if nprops == 1:
                                p = torch.tensor([[c]]).repeat(args.batch_size, 1).to(device)   # for single condition
                        else:
                                p = torch.tensor([c]).repeat(args.batch_size, 1).unsqueeze(1).to(device)    # for multiple conditions
                        
                        crystal = torch.tensor([int(crystal_system)]).repeat(args.batch_size, 1).to(device) 

                        y = model.sample(x, args.block_size, temperature=1.2, do_sample=True, top_k=0.0, top_p=0.9, prop=p, crystal=crystal)  
                        for gen_mol in y:
                                #CG = InvCryRep(graph_method='crystalnn')
                                completion = " ".join([itos[int(i)] for i in gen_mol])
                                completion = completion.replace("<", "").strip()
                                #is_crystal = CG.check_SLICES(completion,4)
                                #reconstructed_structure,final_energy_per_atom_IAP = CG.from_SLICES(completion).to_structures()
                                #print(reconstructed_structure[-1])
                                #if is_crystal:
                                slices.append(completion)
                                #else:print(completion)

                "Valid slices % = {}".format(len(slices))

                cry_dict = []
                for i in slices:
                        cry_dict.append({'crystal_system': crystal_system, 'formation': c, 'slices': i})

                results = pd.DataFrame(cry_dict)

                # with open(f'gen_csv/moses_metrics_7_top10.json', 'w') as file:
                #       json.dump(metrics, file)
                print(results)
                
                unique_slices = list(set([s for s in results['slices']]))
                
                print(f'Condition bandgaps: {c}')
                print('Valid ratio: ', np.round(len(results)/(args.batch_size*gen_iter), 3))
                print('Unique ratio: ', np.round(len(unique_slices)/len(results), 3))                
                
                all_dfs.append(results)

        results = pd.concat(all_dfs)
        results.to_csv(args.csv_name + '.csv', index = False)

        unique_slices = list(set(results['slices']))

        # print('Valid ratio: ', np.round(len(results)/(args.batch_size*gen_iter*count), 3))
        # print('Unique ratio: ', np.round(len(unique_slices)/len(results), 3))
