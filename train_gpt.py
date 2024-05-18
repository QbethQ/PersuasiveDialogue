import torch
import numpy as np
from tqdm import tqdm
from util import get_gpt_model_and_data, get_E2E_gpt_model_and_data
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--e2e', action="store_true")
parser.add_argument('--gpu', type=str, help='GPU ID', default="0")

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("working on gpu")
else:
    device = torch.device("cpu")
    print("working on cpu")

if not args.e2e:
    (model, tokenizer, train_dataloader) = get_gpt_model_and_data(path='./data/train.csv')
    model_dir = './models/PersuaderGPT2_{}/'
    tokenizer_dir = './models/PersuaderGPT2_tokenizer/'
else:
    (model, tokenizer, train_dataloader) = get_E2E_gpt_model_and_data(path='./data/train.csv')
    model_dir = './e2e_models/E2E_PersuaderGPT2_{}/'
    tokenizer_dir = './e2e_models/E2E_PersuaderGPT2_tokenizer/'

tokenizer.save_pretrained(tokenizer_dir)
model.to(device=device)
epochs = 50
optimizer =  torch.optim.Adam(params=model.parameters(), lr = 0.0001)
losses = []
model.train()
for i in range(epochs):
    print("Epoch {} Started".format(i+1))
    for batch_idx, (input_ids, labels) in tqdm(enumerate(train_dataloader), mininterval=10):
        optimizer.zero_grad()
        outputs = model.forward(input_ids.to(device), labels=labels.to(device))
        # logits = outputs.logits
        loss = outputs.loss
        losses.append(loss.mean().item())
        loss.backward()
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        if batch_idx % 100 == 0:
            print("Epoch {} batch {}: loss {}".format(i+1, batch_idx+1, np.mean(losses)))
    print("Epoch {} completed, loss {}\n".format(i+1, np.mean(losses)))

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(model_dir.format(i+1))