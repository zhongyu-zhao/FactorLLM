from ast import arg
from re import template
import os
import sys
import numpy as np
import torch
import argparse
import tqdm
from transformers import AutoModelForCausalLM

sys.path.append('moefication')
import utils

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default='TinyLlama/TinyLlama-1.1B-step-50K-105b', help='model name in huggingface model hub')
parser.add_argument('--res_path', type=str, default='results/tinyllama-1.1b/', help='path to store the results of moefication')
parser.add_argument('--num-layer', type=int, default=22, help='number of layers')
parser.add_argument('--num-expert', type=int, default=4, help='number of experts')
parser.add_argument(
    '--templates', 
    type=str, 
    default='model.layers.{}.mlp.gate_proj.weight,model.layers.{}.mlp.up_proj.weight,model.layers.{}.mlp.down_proj.weight', 
    help='weight names of the first linear layer in each FFN (use comma to separate multiple templates)'
)

args = parser.parse_args()
if not os.path.exists(args.res_path):
    os.makedirs(args.res_path)

model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
torch.save(model.state_dict(), os.path.join(args.res_path, 'model.pt'))

config = utils.ModelConfig(os.path.join(args.res_path, 'model.pt'), args.res_path, split_num=args.num_expert)

templates = args.templates.split(',')

for template in templates:
    for i in range(args.num_layer):
        center = utils.MLPCenter(config, template, '{}/param_split/{}'.format(args.res_path, template.format(i)), i)
        center.cal_center()
