import csv
import random
import argparse
import copy

def preprocess_single_label():
    '''
    all_history: all conversations before
    this turn: generated text in this turn.
    '''
    with open("./data/data.csv", encoding="utf-8-sig", mode="r") as f:
        reader = csv.DictReader(f)
        dias = set()
        temp_list = list()
        for row in reader:
            dias.add(row['B2'])

            for unused_key in ['B4', 'ee_label_1', 'neg', 'neu', 'pos', 'his_stem', 'Unit_char']:
                row.pop(unused_key)

            # If this is the first sentence in this turn
            if len(temp_list) > 0 and row['Turn'] == temp_list[-1]['Turn']:
                row['ifFirst'] = 0
            else:
                row['ifFirst'] = 1

            # calculate all history context
            if row['ifFirst'] == 1:
                if row['history'] == "<Start>":
                    row['all_his'] = "<Start>"
                else:
                    if temp_list[-1]['all_his'] == "<Start>":
                        row['all_his'] = temp_list[-1]['this_turn'] + ' ' + temp_list[-1]['Unit'] + "\nPersuadee: " + row['history']
                    else:
                        row['all_his'] = temp_list[-1]['all_his'] + "\n" + temp_list[-1]['this_turn'] + ' ' + temp_list[-1]['Unit'] + "\nPersuadee: " + row['history']
                row['this_turn'] = "Persuader:"
            else:
                row['all_his'] = temp_list[-1]['all_his']
                row['this_turn'] = temp_list[-1]['this_turn'] + ' ' + temp_list[-1]['Unit']

            row.pop('ifFirst')
            temp_list.append(row)
        dias = list(dias)

        random.shuffle(dias)

        train_size = int(0.7 * len(dias))
        val_size = int(0.1 * len(dias))
        test_size = int(0.2 * len(dias))
        train_set = dias[0 : train_size]
        val_set = dias[train_size : train_size + val_size]
        test_set = dias[train_size + val_size : train_size + val_size + test_size]
        print(len(train_set), len(val_set), len(test_set))

        with open("./data/train.csv", encoding="utf-8-sig", mode="w") as f1:
            with open("./data/val.csv", encoding="utf-8-sig", mode="w") as f2:
                with open("./data/test.csv", encoding="utf-8-sig", mode="w") as f3:
                    writer1 = csv.DictWriter(f1, temp_list[0].keys())
                    writer2 = csv.DictWriter(f2, temp_list[0].keys())
                    writer3 = csv.DictWriter(f3, temp_list[0].keys())
                    writer1.writeheader()
                    writer2.writeheader()
                    writer3.writeheader()
                    for row in temp_list:
                        if row['B2'] in train_set:
                            writer1.writerow(row)
                        if row['B2'] in val_set:
                            writer2.writerow(row)
                        if row['B2'] in test_set:
                            writer3.writerow(row)

def preprocess_multi_label():
    '''
    multilabel classification
    '''
    with open("./data/data.csv", encoding="utf-8-sig", mode="r") as f:
        reader = csv.DictReader(f)
        dias = set()
        temp_list = list()
        temp_clist = list()
        crow = None
        for row in reader:
            dias.add(row['B2'])

            for unused_key in ['B4', 'ee_label_1', 'neg', 'neu', 'pos', 'his_stem', 'Unit_char']:
                row.pop(unused_key)

            # If this is the first sentence in this turn
            if len(temp_list) > 0 and row['Turn'] == temp_list[-1]['Turn']:
                row['ifFirst'] = 0
            else:
                row['ifFirst'] = 1

            # calculate all history context
            if row['ifFirst'] == 1:
                if crow != None:
                    crow.pop('label_inter')
                    crow.pop('label')
                    crow.pop('er_label_1')
                    last_turn = str()
                    last_labels = None
                    idx = 1
                    while idx <= len(temp_list) and temp_list[-idx]['Turn'] == temp_list[-1]['Turn']:
                        last_turn = temp_list[-idx]['Unit'] + " " + last_turn
                        if last_labels != None:
                            if str(temp_list[-idx]['label_inter']) not in last_labels:
                                last_labels = str(temp_list[-idx]['label_inter']) + "," + last_labels
                        else:
                            last_labels = str(temp_list[-idx]['label_inter'])
                        idx += 1
                    crow['labels'] = last_labels
                    crow['Unit'] = last_turn
                    temp_clist.append(crow)

                if row['history'] == "<Start>":
                    row['all_his'] = "<Start>"
                else:
                    # calculate sentences in the last turn
                    last_turn = str()
                    idx = 1
                    while idx <= len(temp_list) and temp_list[-idx]['Turn'] == temp_list[-1]['Turn']:
                        last_turn = temp_list[-idx]['Unit'] + " " + last_turn
                        idx += 1

                    if temp_list[-1]['all_his'] == "<Start>":
                        row['all_his'] = "Persuader: " + last_turn + "\nPersuadee: " + row['history']
                    else:
                        row['all_his'] = temp_list[-1]['all_his'] + "\nPersuader: " + last_turn + "\nPersuadee: " + row['history']
            else:
                row['all_his'] = temp_list[-1]['all_his']
            row.pop('ifFirst')
            crow = copy.copy(row)
            temp_list.append(row)
        dias = list(dias)

        random.shuffle(dias)

        train_size = int(0.7 * len(dias))
        val_size = int(0.1 * len(dias))
        test_size = int(0.2 * len(dias))
        train_set = dias[0 : train_size]
        val_set = dias[train_size : train_size + val_size]
        test_set = dias[train_size + val_size : train_size + val_size + test_size]
        print(len(train_set), len(val_set), len(test_set))

        with open("./data/train_ml.csv", encoding="utf-8-sig", mode="w") as f1:
            with open("./data/val_ml.csv", encoding="utf-8-sig", mode="w") as f2:
                with open("./data/test_ml.csv", encoding="utf-8-sig", mode="w") as f3:
                    writer1 = csv.DictWriter(f1, temp_clist[0].keys())
                    writer2 = csv.DictWriter(f2, temp_clist[0].keys())
                    writer3 = csv.DictWriter(f3, temp_clist[0].keys())
                    writer1.writeheader()
                    writer2.writeheader()
                    writer3.writeheader()
                    for row in temp_clist:
                        if row['B2'] in train_set:
                            writer1.writerow(row)
                        if row['B2'] in val_set:
                            writer2.writerow(row)
                        if row['B2'] in test_set:
                            writer3.writerow(row)

def get_donate_dict(filepath):
    with open(filepath, encoding="utf-8-sig", mode="r") as persuasivenessf:
        reader = csv.DictReader(persuasivenessf)
        dialid2intdon = dict()
        dialid2actudon = dict()
        for row in reader:
            if row['B4'] == "1":
                if row["B5"] == "":
                    dialid2intdon[row['B2']] = "0"
                else:
                    dialid2intdon[row['B2']] = row["B5"]
                dialid2actudon[row['B2']] = row["B6"]
    return dialid2intdon, dialid2actudon

def write_donate_to_data_file(dataset, dialid2intdon, dialid2actudon):
    with open("./data/"+dataset+"_ml.csv", encoding="utf-8-sig", mode="r") as f:
        reader = csv.DictReader(f)
        temp_list = list()
        for row in reader:
            row['B5'] = dialid2intdon[row['B2']]
            row['B6'] = dialid2actudon[row['B2']]
            temp_list.append(row)

    with open("./data/"+dataset+"_p.csv", encoding="utf-8-sig", mode="w") as f:
        writer = csv.DictWriter(f, temp_list[0].keys())
        writer.writeheader()
        for row in temp_list:
            writer.writerow(row)

def persuasiveness():
    dialid2intdon, dialid2actudon = get_donate_dict("./300_info.csv")
    for dataset in ["train", "val", "test"]:
        write_donate_to_data_file(dataset, dialid2intdon, dialid2actudon)
    

parser = argparse.ArgumentParser()
parser.add_argument('--mode', dest='mode', type=int, help='Preprocess Mode', default=0)
args = parser.parse_args()
mode = args.mode
if mode == 0:
    preprocess_single_label()
elif mode == 1:
    preprocess_multi_label()
elif mode == 2:
    persuasiveness()
else:
    print("Wrong Mode!")