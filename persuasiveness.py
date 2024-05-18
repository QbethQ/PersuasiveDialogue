import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from model import Persuasiveness, Persuasivenes_wo_LSTM
import csv
from tqdm import tqdm
from collections import OrderedDict
import pandas as pd
import os

torch.backends.cudnn.enabled = False
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="2"

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        label = self.labels[idx]
        return data_item, label
    
def get_data_label(dataset_type, donation_threshold=5):
    data = []
    labels = []
    with open("./data/"+dataset_type+"_p.csv", encoding="utf-8-sig", mode="r") as f:
        reader = csv.DictReader(f)

        dial = []
        label = (0, 0)
        turn = "-1"
        first_flag = True
        for row in reader:
            if row['Turn'] == "0" and turn != "0":
                if first_flag:
                    first_flag = False
                else:
                    data.append(dial)
                    labels.append(label)
                    dial = []
                dial.append(row['Unit'])
                label = (min(float(row['B5']), donation_threshold), 
                         min(float(row['B6']), donation_threshold), 
                         min(max(float(row['B5']), float(row['B6'])), donation_threshold),
                         )
            else:
                # if row['Turn'] != turn:
                #     dial.append(row['history'])
                #     dial.append(row['Unit'])
                # else:
                #     dial[-1] += " " + row['Unit']
                dial.append(row['Unit'])
            turn = row['Turn']

        data.append(dial)
        labels.append(label)

        assert(len(data) == len(labels))

    return data, labels

def get_dataloader(dataset_type):
    data, labels = get_data_label(dataset_type)
    my_dataset = MyDataset(data, labels)
    batch_size = 1
    dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def test(model, dataloader, loss_function, label_id):
    model.eval()
    test_losses = []
    for batch_id, (dial, labels) in tqdm(enumerate(dataloader)):
        for u in dial:
            print(u)
        x_ids_list = [torch.tensor(model.tokenizer.encode(x[0])[-512:]).long().unsqueeze(0) for x in dial]
        output = model(x_ids_list, labels[0].unsqueeze(0).float())
        print(output)
        print(labels)
        loss = loss_function(output.float(), labels[label_id].unsqueeze(0).float())
        test_losses.append(loss)
    print(f'\tAverage Test Loss: {torch.mean(torch.stack(test_losses))}\n')

def main():
    train_loader = get_dataloader("train")
    val_loader = get_dataloader("val")
    test_loader = get_dataloader("test")

    # settings
    LSTM_num_layers = 2
    # model = Persuasiveness(n_layers = LSTM_num_layers, donation_context=False)
    model = Persuasivenes_wo_LSTM(n_layers = LSTM_num_layers, donation_context=False)
    num_epochs = 30
    loss_function = nn.MSELoss()
    label_id = 1 # 0: intend donation; 1: actual donation; 2: max
    optimizer = optim.Adam(model.parameters(), lr=5e-5)  # Optimizer
    update_steps = 15
    device = "cuda:0"
    model.to(device=device)

    print("BiLSTM num layers:", LSTM_num_layers)

    # training
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        optimizer.zero_grad()
        losses = []

        for batch_id, (dial, labels) in tqdm(enumerate(train_loader)):
            x_ids_list = [torch.tensor(model.tokenizer.encode(x[0])[-512:]).long().unsqueeze(0).to(device=device) for x in dial]
            output = model(x_ids_list, labels[0].to(device=device).unsqueeze(0).float())
            loss = loss_function(output.float(), labels[label_id].to(device=device).unsqueeze(0).float())
            losses.append(loss)
            train_losses.append(loss)

            if (batch_id + 1) % update_steps == 0:
                loss = torch.mean(torch.stack(losses))
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_id+1}/{len(train_loader)}], Loss: {loss.item()}')
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses = []

        # validation and test
        model.eval()
        with torch.no_grad():
            val_losses = []
            val_dict = {'label': [], 'predict': []}
            for batch_id, (dial, labels) in tqdm(enumerate(val_loader)):
                x_ids_list = [torch.tensor(model.tokenizer.encode(x[0])[-512:]).long().unsqueeze(0).to(device=device) for x in dial]
                output = model(x_ids_list, labels[0].to(device=device).unsqueeze(0).float())
                loss = loss_function(output.float(), labels[label_id].to(device=device).unsqueeze(0).float())
                val_dict['label'].append(labels[label_id].float().tolist()[0])
                val_dict['predict'].append(output.float().tolist()[0][0])
                val_losses.append(loss)

            test_losses = []
            test_dict = {'label': [], 'predict': []}
            for batch_id, (dial, labels) in tqdm(enumerate(test_loader)):
                x_ids_list = [torch.tensor(model.tokenizer.encode(x[0])[-512:]).long().unsqueeze(0).to(device=device) for x in dial]
                output = model(x_ids_list, labels[0].to(device=device).unsqueeze(0).float())
                loss = loss_function(output.float(), labels[label_id].to(device=device).unsqueeze(0).float())
                test_dict['label'].append(labels[label_id].float().tolist()[0])
                test_dict['predict'].append(output.float().tolist()[0][0])
                test_losses.append(loss)

            print(f'Epoch [{epoch+1}/{num_epochs}] Done\n', 
                f'\tAverage Training Loss: {torch.mean(torch.stack(train_losses))}\n',
                f'\tAverage Validation Loss: {torch.mean(torch.stack(val_losses))}\n',
                f'\tAverage Test Loss: {torch.mean(torch.stack(test_losses))}\n')
        
        torch.save(model.state_dict(), f'./models/pssness_layer_{LSTM_num_layers}_{label_id}_{epoch+1}.pkl')

if __name__ == '__main__':
    main()