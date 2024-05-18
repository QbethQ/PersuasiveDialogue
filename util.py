import pandas as pd
import numpy as np
import numpy as np
import re
import logging
import re
import sys
import spacy
from sklearn.metrics import f1_score,precision_score,classification_report,hamming_loss,recall_score
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertTokenizer, GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
import json
from tqdm import tqdm

spacy_en = spacy.load('en_core_web_sm')
import torch
import torch.nn as nn
from torch.nn import init
from torchtext import vocab
from torchtext import data
from torchtext.data import Iterator, BucketIterator
def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    
    # Begin CHANGES
    fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "
    
    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES
    
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
        
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.2f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if cm[i, j] != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()
class AverageMeter(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def l2_matrix_norm(m):
    """
    Frobenius norm calculation

    Args:
       m: {Variable} ||AAT - I||

    Returns:
        regularized value

    """
    sum_res=torch.sum(torch.sum(torch.sum(m**2,1),1)**0.5)
    sum_res=sum_res.float().cuda() 
    return sum_res

def eval_(logits, labels,binary=False):
    """
    calculate the accuracy
    :param logits: Variable [batch_size, ]
    :param labels: Variable [batch_size]
    :return:
    """
    if binary is False:
        _, predicted = torch.max(logits.data, 1)
        acc=(predicted == labels).sum().item()/labels.size(0)
        
        macro_f1 = f1_score(labels.cpu(),predicted.cpu(), average='macro')
        micro_f1 = f1_score(labels.cpu(),predicted.cpu(), average='micro')
        
        return acc,macro_f1,micro_f1
    else:
        l=torch.ones(logits.size()).cuda()
        
        l[logits <= 0] = 0

        return (l== labels).sum().item()/labels.size(0)
    
def tokenizer(text): # create a tokenizer function
    text = [tok.text for tok in spacy_en.tokenizer(text) if tok.text != ' ']
    tokenized_text = []
    auxiliary_verbs = ['am', 'is', 'are', 'was', 'were', "'s","'m","'re"]
    punctuation='.,:!?/\\-*&#\'\"'
    num=set('1234567890')
    stop_words=['a','the','of','in','on','to']
    for token in text:
        if token == "n't":
            tmp = 'not'
        elif token == 'US':
            tmp = 'America'
        elif token == "'ll":
            tmp = 'will'
        elif token == "'m":
            tmp = 'am'
        elif token == "'s":
            tmp = 'is'
        elif len(set(token) & num)>0:
            continue
        elif token == "'re":
            tmp = 'are'

        elif token in punctuation:
            continue
        elif token in stop_words:
            continue
        else:
            tmp = token
        tmp=tmp.lower()
        tokenized_text.append(tmp)

    # if len(tokenized_text) >= 256:
    #     tokenized_text = tokenized_text[-255:]
    return tokenized_text

class BatchWrapper:
    def __init__(self, b_iter, x_var, all_his_var, turn_var, y_var, index, this_turn=False, this_turn_var=None):
        self.b_iter, self.x_var, self.all_his_var, self.t_var, self.y_var, self.index, self.this_turn, self.this_turn_var = b_iter, x_var, all_his_var, turn_var, y_var, index, this_turn, this_turn_var

    def __iter__(self):
        for batch in self.b_iter:
            x = getattr(batch, self.x_var) # we assume only one input in this wrapper
            t = getattr(batch, self.t_var)
            all_his = getattr(batch, self.all_his_var)
            index = getattr(batch, self.index)

            if self.y_var is not None:
                y = getattr(batch, self.y_var)
            else:
                y = torch.zeros((1))

            if self.this_turn:
                tt = getattr(batch, self.this_turn_var)
                yield (x,all_his,t,y,index,tt)
            else:
                yield (x,all_his,t,y,index)
            
    
    def __len__(self):
        return len(self.b_iter)
    
class GPTDataset(Dataset):
    """
    GPT model dataset
    """
    def __init__(self, dialogue_list, max_len=1000):
        self.dialogue_list = dialogue_list
        self.max_len = max_len

    def __getitem__(self, index):
        dialogue = self.dialogue_list[index]
        dialogue['input_ids'] = torch.tensor(dialogue['input_ids'][:self.max_len], dtype=torch.long)
        dialogue['labels'] = torch.tensor(dialogue['labels'][:self.max_len], dtype=torch.long)
        return dialogue

    def __len__(self):
        return len(self.dialogue_list)
    
def collate_fn(batch, pad_token_id, pad_label_id=-100):
    input_ids_batch = torch.nn.utils.rnn.pad_sequence([utterance['input_ids'] for utterance in batch], batch_first=True, padding_value=pad_token_id)
    labels_batch = torch.nn.utils.rnn.pad_sequence([utterance['labels'] for utterance in batch], batch_first=True, padding_value=pad_label_id)
    return input_ids_batch, labels_batch
    
class custom_Field(data.Field):
    def __init__(self,model, **kwargs):
        self.model = model
        super(custom_Field, self).__init__(**kwargs)
    
    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a Variable.
        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.
        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (str or torch.device): A string or instance of `torch.device`
                specifying which device the Variables are going to be created on.
                If left as default, the tensors will be created on cpu. Default: None.
        """
        text_feat = self.model.transform(arr)
        text_feat = torch.from_numpy(text_feat).float().cuda()
        return text_feat
        
def load_data_model(model,n_layers=2,drop=0.5,use_gpu=False,num_classes= 6,f_path='./data/',abla_list=None,his_mode='rnn',char_model=None,cnt=0):
    df = pd.read_csv(f_path+'train.csv')
    print(df.head(2))
    datafields=[]
    df.label.value_counts()
    weight=np.array([df.shape[0]/v for v in df.label.value_counts().sort_index().tolist()])
    weight = np.log(weight)
    # Fit tf-idf vector
    # from sklearn.feature_extraction.text import TfidfVectorizer
    # df_tf_idf=df
    # corpus = df_tf_idf.his_stem.tolist()
    # vectorizer = TfidfVectorizer(max_features= 1000)
    # vectorizer = vectorizer.fit(corpus)

    # Define field
    TEXT =data.Field(sequential=True,tokenize=tokenizer, use_vocab=True, lower=True,eos_token='<EOS>',batch_first=True, truncate_first=True,include_lengths=True)
    LABEL = data.Field(sequential=False, use_vocab=False)
    TURN = data.Field(sequential=False, use_vocab=False)
    INDEX= data.Field(sequential=False, use_vocab=False)
    HIS = data.Field(sequential=True,tokenize=tokenizer, use_vocab=True, lower=True,eos_token='<EOS>',batch_first=True, truncate_first=True,include_lengths=True)
   
    for col in df.columns.tolist():
        # if col =='Unit':
        #     datafields.append((col,TEXT))
        # elif col =='Unit_char':
        #     datafields.append((col,CHAR_FEAT))
        if col == 'Index':
            datafields.append((col,INDEX))
        elif col=="label":
            datafields.append((col,LABEL))
        elif col=='Turn':
            datafields.append((col,TURN))
            
        elif col=='history':
            datafields.append((col,TEXT))
        elif col == "all_his":
            datafields.append((col, HIS))
        else:
            datafields.append((col,None))

    # train,valid=dataset.split(split_ratio=0.8,random_state=np.random.seed(42))
    train,valid= data.TabularDataset.splits(   
        format='csv',
        skip_header=True,
        path=f_path,
        train='train.csv',
        validation= 'val.csv',
        fields=datafields,
        )
    test = data.TabularDataset(
        path=f_path+'test.csv',    
        format='csv',
        skip_header=True,
        fields=datafields,
        )

    # using the training corpus to create the vocabulary
    TEXT.build_vocab(train,valid,test)
    TEXT.vocab.load_vectors(vectors='fasttext.en.300d')
    HIS.build_vocab(train,valid,test)#, vectors=vectors, max_size=300000)
    HIS.vocab=TEXT.vocab
    
    print('num of tokens', len(TEXT.vocab.itos))
    print(HIS.vocab.freqs.most_common(5))
    
    print('len(train)', len(train))
    print('len(val)', len(valid))
    print('len(test)', len(test))
    

    train_iter = data.Iterator(dataset=train, batch_size=32 ,train=True, sort_key=lambda x: len(x.Index), sort_within_batch=False,repeat=False)#,device=torch.device('cuda:0') if use_gpu else -1)
    
    val_iter = data.Iterator(dataset=valid, batch_size=32, train=False, sort_key=lambda x: len(x.Index), sort_within_batch=False,repeat=False)#,device=torch.device('cuda:0') if use_gpu else -1)
    
    test_iter = data.Iterator(dataset=test, batch_size=32, train=False, sort_key=lambda x: len(x.Index), sort_within_batch=False,repeat=False)#,device=torch.device('cuda:0') if use_gpu else -1)


    num_tokens = len(HIS.vocab.itos)


    hidden_dim=200
    
    print("No.class",num_classes)
    nets = model(vocab_size=num_tokens, embedding=TEXT.vocab.vectors, hidden_dim=hidden_dim, output_dim=num_classes, n_layers=n_layers, bidirectional=True, dropout=drop,his_mode=his_mode,**abla_list)
    
    train_iter = BatchWrapper(train_iter, 'history', 'all_his', 'Turn', "label", 'Index')
    valid_iter = BatchWrapper(val_iter, 'history', 'all_his', 'Turn', "label", 'Index')
    test_iter = BatchWrapper(test_iter, 'history', 'all_his', 'Turn', "label", 'Index')
  
    weight=torch.from_numpy(weight).float().cuda()
    
    if use_gpu:
        cuda1 = torch.device('cuda:0')
        nets.cuda(device=cuda1)
        return train_iter, valid_iter,test_iter,nets,weight
    else:
        return train_iter, valid_iter,test_iter,nets,weight
    
def load_data_bertcls(num_classes=11, f_path='./data/', this_turn=False):    
    df = pd.read_csv(f_path+'train'+'.csv')
    print(df.head(2))
    datafields=[]
    df.label.value_counts()
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def simple_tokenizer(text):
        tokens = bert_tokenizer.tokenize(text)
        # print("simple_tokenizer Step 1:", tokens)
        tokens = tokens[-510:]
        # print("simple_tokenizer Step 2:", tokens)
        return tokens

    # Define field
    TEXT =data.Field(sequential=True, use_vocab=False, lower=False, batch_first=True, truncate_first=True, include_lengths=True, 
                     tokenize=simple_tokenizer, preprocessing=bert_tokenizer.convert_tokens_to_ids,
                     init_token=bert_tokenizer.cls_token_id, eos_token=bert_tokenizer.sep_token_id, 
                     pad_token=bert_tokenizer.pad_token_id, unk_token=bert_tokenizer.unk_token_id)
    LABEL = data.Field(sequential=False, use_vocab=False)
    TURN = data.Field(sequential=False, use_vocab=False)
    INDEX= data.Field(sequential=False, use_vocab=False)
    HIS = data.Field(sequential=True, use_vocab=False, lower=False, batch_first=True, truncate_first=True, include_lengths=True, 
                     tokenize=simple_tokenizer, preprocessing=bert_tokenizer.convert_tokens_to_ids,
                     init_token=bert_tokenizer.cls_token_id, eos_token=bert_tokenizer.sep_token_id, 
                     pad_token=bert_tokenizer.pad_token_id, unk_token=bert_tokenizer.unk_token_id)
    if this_turn:
        THISTURN = data.Field(sequential=True, use_vocab=False, lower=False, batch_first=True, truncate_first=True, include_lengths=True, 
                              tokenize=simple_tokenizer, preprocessing=bert_tokenizer.convert_tokens_to_ids,
                              init_token=bert_tokenizer.cls_token_id, eos_token=bert_tokenizer.sep_token_id, 
                              pad_token=bert_tokenizer.pad_token_id, unk_token=bert_tokenizer.unk_token_id)
   
    for col in df.columns.tolist():
        # if col =='Unit':
        #     datafields.append((col,TEXT))
        # elif col =='Unit_char':
        #     datafields.append((col,CHAR_FEAT))
        if col == "Index":
            datafields.append((col, INDEX))
        elif col == "label":
            datafields.append((col, LABEL))
        elif col == 'Turn':
            datafields.append((col, TURN))
        elif col == 'history':
            datafields.append((col, TEXT))
        elif col == "all_his":
            datafields.append((col, HIS))
        elif col == "this_turn" and this_turn:
            datafields.append((col, THISTURN))
        else:
            datafields.append((col, None))

    # train,valid=dataset.split(split_ratio=0.8,random_state=np.random.seed(42))
    train,valid= data.TabularDataset.splits(   
        format='csv',
        skip_header=True,
        path=f_path,
        train='train'+'.csv',
        validation= 'val'+'.csv',
        fields=datafields,
        )
    test = data.TabularDataset(
        path=f_path+'test'+'.csv',    
        format='csv',
        skip_header=True,
        fields=datafields,
        )
    
    print('len(train)', len(train))
    print('len(val)', len(valid))
    print('len(test)', len(test))
    

    train_iter = data.Iterator(dataset=train, batch_size=32 ,train=True, sort_key=lambda x: len(x.Index), sort_within_batch=False,repeat=False)#,device=torch.device('cuda:0') if use_gpu else -1)
    
    val_iter = data.Iterator(dataset=valid, batch_size=32, train=False, sort_key=lambda x: len(x.Index), sort_within_batch=False,repeat=False)#,device=torch.device('cuda:0') if use_gpu else -1)
    
    test_iter = data.Iterator(dataset=test, batch_size=32, train=False, sort_key=lambda x: len(x.Index), sort_within_batch=False,repeat=False)#,device=torch.device('cuda:0') if use_gpu else -1)
    
    print("No.class",num_classes)

    train_iter = BatchWrapper(train_iter, 'history', 'all_his', 'Turn', "label", 'Index', this_turn=this_turn, this_turn_var="this_turn")
    valid_iter = BatchWrapper(val_iter, 'history', 'all_his', 'Turn', "label", 'Index', this_turn=this_turn, this_turn_var="this_turn")
    test_iter = BatchWrapper(test_iter, 'history', 'all_his', 'Turn', "label", 'Index', this_turn=this_turn, this_turn_var="this_turn")
  
    return train_iter, valid_iter, test_iter

def load_data_bertmulticls(num_classes=11, f_path='./data/', this_turn=False):    
    df = pd.read_csv(f_path+'train'+'_ml.csv')
    print(df.head(2))
    datafields=[]
    df.labels.value_counts()
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def simple_tokenizer(text):
        tokens = bert_tokenizer.tokenize(text)
        # print("simple_tokenizer Step 1:", tokens)
        tokens = tokens[-510:]
        # print("simple_tokenizer Step 2:", tokens)
        return tokens

    # Define field
    TEXT =data.Field(sequential=True, use_vocab=False, lower=False, batch_first=True, truncate_first=True, include_lengths=True, 
                     tokenize=simple_tokenizer, preprocessing=bert_tokenizer.convert_tokens_to_ids,
                     init_token=bert_tokenizer.cls_token_id, eos_token=bert_tokenizer.sep_token_id, 
                     pad_token=bert_tokenizer.pad_token_id, unk_token=bert_tokenizer.unk_token_id)
    LABEL = data.Field(tokenize=lambda x: x.split(","), unk_token=None)
    TURN = data.Field(sequential=False, use_vocab=False)
    INDEX= data.Field(sequential=False, use_vocab=False)
    HIS = data.Field(sequential=True, use_vocab=False, lower=False, batch_first=True, truncate_first=True, include_lengths=True, 
                     tokenize=simple_tokenizer, preprocessing=bert_tokenizer.convert_tokens_to_ids,
                     init_token=bert_tokenizer.cls_token_id, eos_token=bert_tokenizer.sep_token_id, 
                     pad_token=bert_tokenizer.pad_token_id, unk_token=bert_tokenizer.unk_token_id)
    if this_turn:
        THISTURN = data.Field(sequential=True, use_vocab=False, lower=False, batch_first=True, truncate_first=True, include_lengths=True, 
                              tokenize=simple_tokenizer, preprocessing=bert_tokenizer.convert_tokens_to_ids,
                              init_token=bert_tokenizer.cls_token_id, eos_token=bert_tokenizer.sep_token_id, 
                              pad_token=bert_tokenizer.pad_token_id, unk_token=bert_tokenizer.unk_token_id)
   
    for col in df.columns.tolist():
        # if col =='Unit':
        #     datafields.append((col,TEXT))
        # elif col =='Unit_char':
        #     datafields.append((col,CHAR_FEAT))
        if col == "Index":
            datafields.append((col, INDEX))
        elif col == "labels":
            datafields.append((col, LABEL))
        elif col == 'Turn':
            datafields.append((col, TURN))
        elif col == 'history':
            datafields.append((col, TEXT))
        elif col == "all_his":
            datafields.append((col, HIS))
        elif col == "this_turn" and this_turn:
            datafields.append((col, THISTURN))
        else:
            datafields.append((col, None))

    # train,valid=dataset.split(split_ratio=0.8,random_state=np.random.seed(42))
    train,valid= data.TabularDataset.splits(   
        format='csv',
        skip_header=True,
        path=f_path,
        train='train'+'_ml.csv',
        validation= 'val'+'_ml.csv',
        fields=datafields,
        )
    test = data.TabularDataset(
        path=f_path+'test'+'_ml.csv',    
        format='csv',
        skip_header=True,
        fields=datafields,
        )
    LABEL.build_vocab(train)

    with open("multilabel_dict.json", "w") as mld_f:
        json.dump(
            dict(LABEL.vocab.stoi),
            mld_f,
            indent=4,
            ensure_ascii=False
        )
    print(LABEL.vocab.stoi)
    
    print('len(train)', len(train))
    print('len(val)', len(valid))
    print('len(test)', len(test))
    

    train_iter = data.Iterator(dataset=train, batch_size=32 ,train=True, sort_key=lambda x: len(x.Index), sort_within_batch=False,repeat=False)#,device=torch.device('cuda:0') if use_gpu else -1)
    
    val_iter = data.Iterator(dataset=valid, batch_size=32, train=False, sort_key=lambda x: len(x.Index), sort_within_batch=False,repeat=False)#,device=torch.device('cuda:0') if use_gpu else -1)
    
    test_iter = data.Iterator(dataset=test, batch_size=32, train=False, sort_key=lambda x: len(x.Index), sort_within_batch=False,repeat=False)#,device=torch.device('cuda:0') if use_gpu else -1)
    
    print("No.class",num_classes)

    train_iter = BatchWrapper(train_iter, 'history', 'all_his', 'Turn', "labels", 'Index', this_turn=this_turn, this_turn_var="this_turn")
    valid_iter = BatchWrapper(val_iter, 'history', 'all_his', 'Turn', "labels", 'Index', this_turn=this_turn, this_turn_var="this_turn")
    test_iter = BatchWrapper(test_iter, 'history', 'all_his', 'Turn', "labels", 'Index', this_turn=this_turn, this_turn_var="this_turn")
  
    return train_iter, valid_iter,test_iter

def add_special_tokens_(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, attr_to_special_token):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(attr_to_special_token) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

def get_gpt_model_and_data(path='./data/data.csv', model='gpt2'):
    SPECIAL_TOKENS = ["<cls>", "<sep>", "<persuadee>", "<persuader>", "<pad>", "<persona_begin>", "<persona_end>", "<sentiment_begin>", "<sentiment_end>", "<policy_begin>", "<policy_end>"]
    ATTR_TO_SPECIAL_TOKEN = {'cls_token': '<cls>', 'sep_token': '<sep>', 'pad_token': '<pad>', 'additional_special_tokens': ['<persuadee>', '<persuader>', '<persona_begin>', '<persona_end>', '<sentiment_begin>', '<sentiment_end>', "<policy_begin>", "<policy_end>"]}

    gpt_tokenizer = GPT2Tokenizer.from_pretrained(model)
    gpt_model = GPT2LMHeadModel.from_pretrained(model)
    add_special_tokens_(gpt_model, gpt_tokenizer, ATTR_TO_SPECIAL_TOKEN)
    
    gpt_tokenizer.save_vocabulary(save_directory="./")
    cls_id, sep_id, ee, er, pad_id, persona_begin, persona_end, sentiment_begin, sentiment_end, policy_begin, policy_end = gpt_tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:])
    df = pd.read_csv(path)
    dialogue_list = []
    input_ids = [] + [cls_id]
    labels = [] + [-100] # labels -100 are ignored
    for i in tqdm(range(len(df['Index']))):
        if i > 0 and df['B2'][i] != df['B2'][i-1]:
            dialogue_list.append({"input_ids": input_ids, "labels": labels})
            input_ids = [] + [cls_id]
            labels = [] + [-100] # labels -100 are ignored

        if df['history'][i] != '<Start>' and df['history'][i] != df['history'][i-1]:
            # print(df['history'][i])
            # print(gpt_tokenizer.tokenize(df['history'][i]))
            history_ids = gpt_tokenizer.convert_tokens_to_ids(gpt_tokenizer.tokenize(df['history'][i]))
            # print(history_ids)
            input_ids += [ee] + history_ids + [sep_id]
            labels += [-100] * (len(history_ids) + 2)

        unit_ids = gpt_tokenizer.convert_tokens_to_ids(gpt_tokenizer.tokenize(df['Unit'][i]))

        if df['label_inter'][i] != 'non-strategy':
            policy_ids = gpt_tokenizer.convert_tokens_to_ids(gpt_tokenizer.tokenize(df['label_inter'][i]))

            input_ids += [er, policy_begin] + policy_ids + [policy_end] + unit_ids + [sep_id]
            labels += [-100] * (2 + len(policy_ids) + 1) + unit_ids + [sep_id]
            
        else:
            input_ids += [er] + unit_ids + [sep_id]
            labels += [-100] + unit_ids + [sep_id]

    dialogue_list.append({"input_ids": input_ids, "labels": labels})

    train_dataset = GPTDataset(dialogue_list)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, 
                                  collate_fn=lambda x: collate_fn(x, pad_token_id=pad_id, pad_label_id=-100), drop_last=True)
    return (gpt_model, gpt_tokenizer, train_dataloader)
            
def get_E2E_gpt_model_and_data(path='./data/data.csv', model='gpt2'):
    SPECIAL_TOKENS = ["<cls>", "<sep>", "<persuadee>", "<persuader>", "<pad>", "<persona_begin>", "<persona_end>", "<sentiment_begin>", "<sentiment_end>", "<policy_begin>", "<policy_end>"]
    ATTR_TO_SPECIAL_TOKEN = {'cls_token': '<cls>', 'sep_token': '<sep>', 'pad_token': '<pad>', 'additional_special_tokens': ['<persuadee>', '<persuader>', '<persona_begin>', '<persona_end>', '<sentiment_begin>', '<sentiment_end>', "<policy_begin>", "<policy_end>"]}

    gpt_tokenizer = GPT2Tokenizer.from_pretrained(model)
    gpt_model = GPT2LMHeadModel.from_pretrained(model)
    add_special_tokens_(gpt_model, gpt_tokenizer, ATTR_TO_SPECIAL_TOKEN)
    
    gpt_tokenizer.save_vocabulary(save_directory="./")
    cls_id, sep_id, ee, er, pad_id, persona_begin, persona_end, sentiment_begin, sentiment_end, policy_begin, policy_end = gpt_tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:])
    df = pd.read_csv(path)
    dialogue_list = []
    input_ids = [] + [cls_id]
    labels = [] + [-100] # labels -100 are ignored
    for i in tqdm(range(len(df['Index']))):
        if i > 0 and df['B2'][i] != df['B2'][i-1]:
            dialogue_list.append({"input_ids": input_ids, "labels": labels})
            input_ids = [] + [cls_id]
            labels = [] + [-100] # labels -100 are ignored

        if df['history'][i] != '<Start>' and df['history'][i] != df['history'][i-1]:
            # print(df['history'][i])
            # print(gpt_tokenizer.tokenize(df['history'][i]))
            history_ids = gpt_tokenizer.convert_tokens_to_ids(gpt_tokenizer.tokenize(df['history'][i]))
            # print(history_ids)
            input_ids += [ee] + history_ids + [sep_id]
            labels += [-100] * (len(history_ids) + 2)

        unit_ids = gpt_tokenizer.convert_tokens_to_ids(gpt_tokenizer.tokenize(df['Unit'][i]))

        # if df['label_inter'][i] != 'non-strategy':
        #     policy_ids = gpt_tokenizer.convert_tokens_to_ids(gpt_tokenizer.tokenize(df['label_inter'][i]))

        #     input_ids += [er, policy_begin] + policy_ids + [policy_end] + unit_ids + [sep_id]
        #     labels += [-100] * (2 + len(policy_ids) + 1) + unit_ids + [sep_id]
            
        # else:
        input_ids += [er] + unit_ids + [sep_id]
        labels += [-100] + unit_ids + [sep_id]

    dialogue_list.append({"input_ids": input_ids, "labels": labels})

    train_dataset = GPTDataset(dialogue_list)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, 
                                  collate_fn=lambda x: collate_fn(x, pad_token_id=pad_id, pad_label_id=-100), drop_last=True)
    return (gpt_model, gpt_tokenizer, train_dataloader)

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
    
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num=8, alpha=None, gamma=5, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
    
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)
        

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        
        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)
        
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss    
