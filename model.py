import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from torch.nn import Embedding
import numpy as np
import random
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel, BertOnlyMLMHead


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding,self.embedding_dim=self._load_embeddings(vocab_size,use_pretrained_embeddings=True,embeddings=embedding)
        #self.hidden_state = self._init_hidden()
        self.rnn = nn.LSTM(self.embedding_dim, hidden_dim, num_layers=n_layers,batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.bool_fc =  nn.Linear(hidden_dim*2, 1)
    def _load_embeddings(self,vocab_size,emb_dim=None,use_pretrained_embeddings=False,embeddings=None):
        """Load the embeddings based on flag"""
       
        if use_pretrained_embeddings is True and embeddings is None:
            raise Exception("Send a pretrained word embedding as an argument")
           
        if not use_pretrained_embeddings and vocab_size is None:
            raise Exception("Vocab size cannot be empty")
   
        if not use_pretrained_embeddings:
            word_embeddings = torch.nn.Embedding(vocab_size,emb_dim,padding_idx=0)
            
        elif use_pretrained_embeddings:
            word_embeddings = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
            word_embeddings.weight = torch.nn.Parameter(embeddings)
            emb_dim = embeddings.size(1)
                
        return word_embeddings,emb_dim
       
    def forward(self, x,*args,**kwargs):
        
        #x = [sent len, batch size]
        #embedded = [sent len, batch size, emb dim]
        embedded = self.dropout(self.embedding(x))
        #output = [sent len, batch size, hid dim * num directions]
        #(hidden,cell) = ([num layers * num directions, batch size, hid dim]*2)
        
        outputs, (hidden,cell) = self.rnn(embedded)
        #hidden [batch size, hid. dim * num directions]
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
#         sf=nn.Softmax(dim=1)
        output = self.fc(hidden)         
        bool_l=self.bool_fc(hidden)    
        return output,bool_l,bool_l
        return output
    
class BERTCls(nn.Module):
    def __init__(self, dropout=0.1, num_class=11, turn_emb_dim=12, this_turn=False, freeze_bert=True):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.this_turn = this_turn
        if self.this_turn:
            self.cls = nn.Linear(3 * self.bert.config.hidden_size + turn_emb_dim, num_class, bias=False)
        else:
            self.cls = nn.Linear(2 * self.bert.config.hidden_size + turn_emb_dim, num_class, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_class))
        self.cls.bias = self.bias
        self.turn_embeddings=torch.nn.Embedding(30,turn_emb_dim)
        self.dropout = nn.Dropout(dropout)
        if freeze_bert:
            for name, params in self.named_parameters():
                if name.startswith("bert") and (not name.startswith("bert.pooler")):
                    params.requires_grad = False
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x_ids = None, his_ids = None, this_turn_ids = None, turn = None):
        '''input_id of size (batch_size, seq_length)'''
        x_outputs = self.bert(input_ids = x_ids)
        his_outputs = self.bert(input_ids = his_ids)
        x_pooled_output = x_outputs[1]
        his_pooled_output = his_outputs[1]
        t_embed=self.turn_embeddings(turn)
        
        if self.this_turn:
            this_turn_outputs = self.bert(input_ids = this_turn_ids)
            tt_pooled_output = this_turn_outputs[1]
            pooled_output = torch.cat((x_pooled_output, his_pooled_output, tt_pooled_output, t_embed), 1)
        else:
            pooled_output = torch.cat((x_pooled_output, his_pooled_output, t_embed), 1)
            
        scores = self.cls(pooled_output)
        return scores

class BERTMultiCls(nn.Module):
    def __init__(self, dropout, num_class=11, turn_emb_dim=12, freeze_bert=True):
        '''
        num_class: not include label <pad> 
        '''
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.output_feature_dim = num_class + 1
        self.cls = nn.Linear(2 * self.bert.config.hidden_size + turn_emb_dim, self.output_feature_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.output_feature_dim))
        self.cls.bias = self.bias
        self.turn_embeddings=torch.nn.Embedding(30,turn_emb_dim)
        self.dropout = nn.Dropout(dropout)
        if freeze_bert:
            for name, params in self.named_parameters():
                if name.startswith("bert") and (not name.startswith("bert.pooler")):
                    params.requires_grad = False

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x_ids = None, his_ids = None, this_turn_ids = None, turn = None):
        '''input_id of size (batch_size, seq_length)'''
        x_outputs = self.bert(input_ids = x_ids)
        his_outputs = self.bert(input_ids = his_ids)
        x_pooled_output = x_outputs[1]
        his_pooled_output = his_outputs[1]
        t_embed=self.turn_embeddings(turn)
        
        pooled_output = torch.cat((x_pooled_output, his_pooled_output, t_embed), 1)
            
        scores = self.cls(pooled_output)
        return scores
    
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding,self.embedding_dim=self._load_embeddings(vocab_size,use_pretrained_embeddings=True,embeddings=embedding)
        #self.hidden_state = self._init_hidden()
        self.rnn = nn.LSTM(self.embedding_dim, hidden_dim, num_layers=n_layers,batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.bool_fc =  nn.Linear(hidden_dim*2, 1)
    def _load_embeddings(self,vocab_size,emb_dim=None,use_pretrained_embeddings=False,embeddings=None):
        """Load the embeddings based on flag"""
       
        if use_pretrained_embeddings is True and embeddings is None:
            raise Exception("Send a pretrained word embedding as an argument")
           
        if not use_pretrained_embeddings and vocab_size is None:
            raise Exception("Vocab size cannot be empty")
   
        if not use_pretrained_embeddings:
            word_embeddings = torch.nn.Embedding(vocab_size,emb_dim,padding_idx=0)
            
        elif use_pretrained_embeddings:
            word_embeddings = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
            word_embeddings.weight = torch.nn.Parameter(embeddings)
            emb_dim = embeddings.size(1)
                
        return word_embeddings,emb_dim
       
    def forward(self, x,*args,**kwargs):
        
        #x = [sent len, batch size]
        #embedded = [sent len, batch size, emb dim]
        embedded = self.dropout(self.embedding(x))
        #output = [sent len, batch size, hid dim * num directions]
        #(hidden,cell) = ([num layers * num directions, batch size, hid dim]*2)
        
        outputs, (hidden,cell) = self.rnn(embedded)
        #hidden [batch size, hid. dim * num directions]
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
#         sf=nn.Softmax(dim=1)
        output = self.fc(hidden)         
        bool_l=self.bool_fc(hidden)    
        return output,bool_l,bool_l
        return output
      
class Persuasiveness(nn.Module):
    def __init__(self, n_layers=2, dropout=0.15, bidirectional=True, freeze_bert=True, donation_context=False):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.donation_context = donation_context
        self.rnn = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, num_layers=n_layers, batch_first=False, bidirectional=bidirectional, dropout=dropout)
        # self.attn = nn.MultiheadAttention(embed_dim=self.bert.config.hidden_size*2, num_heads=8, dropout=dropout)
        self.fc1 = nn.Linear(self.bert.config.hidden_size*2, 32, bias=False)
        self.act_fn = nn.Tanh()
        self.fc2 = nn.Linear(32 + (1 if self.donation_context else 0), 8, bias=False)
        self.fc3 = nn.Linear(8, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        # self.bool_fc =  nn.Linear(self.bert.config.hidden_size*2, 1)

    def forward(self, x_ids_list = None, c = torch.tensor([[0]]), *args,**kwargs):
        x_outputs_list = [self.bert(input_ids = x_ids) for x_ids in x_ids_list]
        x_pooled_output = torch.stack([x_outputs[1] for x_outputs in x_outputs_list])
        x_pooled_output = self.dropout(x_pooled_output)
        # x = [dialogue_len, batch size, hidden_size]
        # output = [dialogue_len, batch size, hidden_size * num directions]
        #(hidden,cell) = ([num layers * num directions, batch size, hid dim]*2)
        
        outputs, (hidden,cell) = self.rnn(x_pooled_output)
        #hidden [batch size, hid. dim * num directions]
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
#         sf=nn.Softmax(dim=1)
        output = self.fc1(hidden)
        output = self.act_fn(output)
        output = self.dropout(output)
        if self.donation_context:
            output = torch.cat((output, c), dim=1)
        output = self.fc2(output)
        output = self.act_fn(output)
        output = self.dropout(output)

        output = self.fc3(output)
        # bool_l=self.bool_fc(hidden)
        # print("bool_l", bool_l.shape)
        return output
    
class Persuasivenes_wo_LSTM(nn.Module):
    def __init__(self, n_layers=2, dropout=0.15, bidirectional=True, freeze_bert=True, donation_context=False):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.donation_context = donation_context
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 32, bias=False)
        self.act_fn = nn.Tanh()
        self.fc2 = nn.Linear(32 + (1 if self.donation_context else 0), 8, bias=False)
        self.fc3 = nn.Linear(8, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        # self.bool_fc =  nn.Linear(self.bert.config.hidden_size*2, 1)

    def forward(self, x_ids_list = None, c = torch.tensor([[0]]), *args,**kwargs):
        x_outputs_list = [self.bert(input_ids = x_ids) for x_ids in x_ids_list]
        x_pooled_output = torch.stack([x_outputs[1] for x_outputs in x_outputs_list])
        x_pooled_output = self.dropout(x_pooled_output)
        # x = [dialogue_len, batch size, hidden_size]
        hidden = torch.mean(x_pooled_output, dim=0)
        
        output = self.fc1(hidden)
        output = self.act_fn(output)
        output = self.dropout(output)
        if self.donation_context:
            output = torch.cat((output, c), dim=1)
        output = self.fc2(output)
        output = self.act_fn(output)
        output = self.dropout(output)

        output = self.fc3(output)
        # bool_l=self.bool_fc(hidden)
        # print("bool_l", bool_l.shape)
        return output