from nltk.translate.bleu_score import sentence_bleu
from interact import PersuaderGPT, E2E_PersuaderGPT
import pandas as pd
from tqdm import tqdm
from util import AverageMeter
import copy
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel
import torch
import math
import pickle as pkl
import argparse
import os
import random
import numpy as np

class stat():
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.var = 0
        self.sd = 0
        self.sum = 0
        self.square_sum = 0
        self.cnt = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.square_sum += val * val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.var = self.square_sum / self.cnt - self.avg * self.avg
        self.sd = math.sqrt(self.var)

def get_eval_data(data_path, model_path, if_e2e, ref_hypo_f=None, sentence_list_cache=None):
    if if_e2e:
        model = E2E_PersuaderGPT(model_path)
    else:
        model = PersuaderGPT(model_path)
    SPECIAL_TOKENS = ["<cls>", "<sep>", "<persuadee>", "<persuader>", "<pad>", "<persona_begin>", "<persona_end>", "<sentiment_begin>", "<sentiment_end>", "<policy_begin>", "<policy_end>"]
    cls_id, sep_id, ee, er, pad_id, persona_begin, persona_end, sentiment_begin, sentiment_end, policy_begin, policy_end = model.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:])

    if ref_hypo_f:
        ref_hypo_f.write("[val]" if "val" in data_path else "[test]" if "test" in data_path else data_path)
        ref_hypo_f.write(" " + model_path + "\n")
        ref_hypo_f.write("{\n")
    
    df = pd.read_csv(data_path)
    sentence_list = []
    input_ids = [] + [cls_id]
    for i in tqdm(range(len(df['Index'])), mininterval=10):
        if i > 0 and df['B2'][i] != df['B2'][i-1]:
            input_ids = [] + [cls_id]

        if df['history'][i] != '<Start>' and df['history'][i] != df['history'][i-1]:
            # print(df['history'][i])
            # print(model.tokenizer.tokenize(df['history'][i]))
            history_ids = model.tokenizer.convert_tokens_to_ids(model.tokenizer.tokenize(df['history'][i]))
            # print(history_ids)
            input_ids += [ee] + history_ids + [sep_id]

        model.input_ids = copy.deepcopy(input_ids)

        if if_e2e:
            model_text = model.predict()
        else:
            model_text = model.predict(df['label_inter'][i])
        ref = df['Unit'][i]
        sentence_list.append({"reference": ref.lower(), "hypothesis": model_text.lower()})

        # print(model.check())
        if ref_hypo_f:
            ref_hypo_f.write("\tReference: " + ref + "\n")
            ref_hypo_f.write("\tHypothesis: " + model_text + "\n\n")

        unit_ids = model.tokenizer.convert_tokens_to_ids(model.tokenizer.tokenize(df['Unit'][i]))

        if df['label_inter'][i] != 'non-strategy' and (not if_e2e):
            policy_ids = model.tokenizer.convert_tokens_to_ids(model.tokenizer.tokenize(df['label_inter'][i]))
            input_ids += [er, policy_begin] + policy_ids + [policy_end] + unit_ids + [sep_id]
        else:
            input_ids += [er] + unit_ids + [sep_id]

    if ref_hypo_f:
        ref_hypo_f.write("}\n")

    if sentence_list_cache:
        with open(sentence_list_cache, "wb") as fsl:
            pkl.dump(sentence_list, fsl)

    return sentence_list

def count_BLEU(sentence_list):
    bleu_scores = [AverageMeter() for i in range(8)] # Average, BLEU-1, BLEU-2, BLEU-3, BLEU-4
    weights = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (1, 0, 0, 0), (0.5, 0.5, 0, 0), (1.0/3.0, 1.0/3.0, 1.0/3.0, 0), (0.25, 0.25, 0.25, 0.25)]
    for pair in tqdm(sentence_list):
        for i in range(8):
            bleu_scores[i].update(sentence_bleu([pair["reference"].split()], pair["hypothesis"].split(), weights[i]))
    return bleu_scores

@torch.no_grad()
def count_sentence_sim(sentence_list, mode="cls"):
    """mode: 'cls' or 'pool'"""
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_max_len = 512
    cos_sims = stat()
    random_cos_sims = stat()
    euclid_dists = stat()
    random_euclid_dists = stat()
    vectors = AverageMeter()
    ref_list = list()
    hypo_list = list()

    for pair in tqdm(sentence_list):
        ref = pair['reference']
        ref_ids = bert_tokenizer.encode(ref)
        ref_ids = ref_ids[-bert_max_len:]
        ref_ids = torch.tensor(ref_ids).long().unsqueeze(0)
        ref_output = bert_model(input_ids = ref_ids)
        hypo = pair['hypothesis']
        hypo_ids = bert_tokenizer.encode(hypo)
        hypo_ids = hypo_ids[-bert_max_len:]
        hypo_ids = torch.tensor(hypo_ids).long().unsqueeze(0)
        hypo_output = bert_model(input_ids = hypo_ids)

        if mode == "cls":
            ref_output = (ref_output.last_hidden_state)[0][0]
            hypo_output = (hypo_output.last_hidden_state)[0][0]

        elif mode == "pool":
            ref_output = (ref_output.pooler_output)[0]
            hypo_output = (hypo_output.pooler_output)[0]

        vectors.update(ref_output)
        vectors.update(hypo_output)
        ref_list.append(ref_output)
        hypo_list.append(hypo_output)

    length = len(ref_list)

    for i in range(length):
        cos_sim = torch.cosine_similarity(ref_list[i] - vectors.avg, hypo_list[i] - vectors.avg, dim=0).item()
        cos_sims.update(cos_sim)
        dist = torch.dist(ref_list[i], hypo_list[i]).item()
        euclid_dists.update(dist)

    n_sample = 2000
    for i in range(n_sample):
        i1 = random.randint(0, length - 1)
        i2 = random.randint(0, length - 1)
        cos_sim = torch.cosine_similarity(ref_list[i1] - vectors.avg, hypo_list[i2] - vectors.avg, dim=0).item()
        random_cos_sims.update(cos_sim)
        dist = torch.dist(ref_list[i1], hypo_list[i2]).item()
        random_euclid_dists.update(dist)

    return cos_sims, euclid_dists, random_cos_sims, random_euclid_dists

def sentence_embedding(x_ebds):
    '''
    computing sentence embedding by computing average of all word embeddings of sentence.\
    :param x_ebds: list[list1, list2,...,listn], listk(k=1...n) is word vector which from sentence x
    '''
    sen_embed = np.array([0 for _ in range(len(x_ebds[0]))]) # 存放句向量
 
    for x_v in x_ebds:
        sen_embed = np.add(x_v, sen_embed)
    sen_embed = sen_embed / math.sqrt(sum(np.square(sen_embed)))
    return sen_embed
 
 
def embedding_average(x_ebds, y_ebds, norm=False):
    x = sentence_embedding(x_ebds)
    y = sentence_embedding(y_ebds)
    return cosine_similarity(x, y, norm)

def vector_extrema(x_ebds):
    '''
    computing vector extrema by compapring maximun value of all word embeddings in same dimension.
    :param x_ebds: list[list1, list2,...,listn], listk(k=1...n) is word vector which from sentence x,
    :return: a vector extrema
    '''
    vec_extre = np.max(x_ebds, axis=0)
    return vec_extre

def embedding_extreme(x_ebds, y_ebds, norm=False):
    x = vector_extrema(x_ebds)
    y = vector_extrema(y_ebds)
    return cosine_similarity(x, y, norm)

def greedy(x_ebds, y_ebds):
    '''
    :param x: a sentence, type is string.
    :param x_ebds: list[list1, list2,...,listn], listk(k=1...n) is word vector which from sentence x,
    :param y_ebds: list[list1, list2,..., listn], listk(k=1...n) is word vector which from sentence y,
    :return: a scalar, it's value is in [0, 1]
    '''
    cosine = []
    sum_x = 0
    for x_v in x_ebds:
        for y_v in y_ebds:
            cosine.append(cosine_similarity(x_v, y_v))
        if cosine:
            sum_x += max(cosine)
            cosine = []
    sum_x = sum_x / x_ebds.shape[0]
    return sum_x

def embedding_greedy(x, y):
    '''
    :param lines: english word embbeding list, like[['-','0.345',...,'0.3123'],...]
    :param x: a sentence, here is a candidate answer
    :param y: a sentence, here is reference answer
    :return: a scalar in [0,1]
    '''
    # greedy match
    sum_x = greedy(x, y)
    sum_y = greedy(y, x)
    score = (sum_x+sum_y)/2
    return score


def cosine_similarity(x, y, norm=False):
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = np.array([0 for _ in range(len(x))])
    if x.all() == zero_list.all() or y.all() == zero_list.all():
        return float(1) if x == y else float(0)
 
    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
 
    return 0.5 * cos + 0.5 if norm else cos

@torch.no_grad()
def embedding_sim(sentence_list):
    average = stat()
    extreme = stat()
    greedy = stat()

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_max_len = 512
    
    for pair in tqdm(sentence_list):
        ref = pair['reference']
        ref_ids = bert_tokenizer.encode(ref)
        ref_ids = ref_ids[1:-1]
        ref_ids = ref_ids[-bert_max_len:]
        ref_ids = torch.tensor(ref_ids).long().unsqueeze(0)
        ref_embeddings = bert_model.embeddings.word_embeddings(ref_ids)[0].numpy()
        hypo = pair['hypothesis']
        hypo_ids = bert_tokenizer.encode(hypo)
        hypo_ids = hypo_ids[1:-1]
        hypo_ids = hypo_ids[-bert_max_len:]
        hypo_ids = torch.tensor(hypo_ids).long().unsqueeze(0)
        hypo_embeddings = bert_model.embeddings.word_embeddings(hypo_ids)[0].numpy()

        average.update(embedding_average(ref_embeddings, hypo_embeddings))
        extreme.update(embedding_extreme(ref_embeddings, hypo_embeddings))
        greedy.update(embedding_greedy(ref_embeddings, hypo_embeddings))

    return average, extreme, greedy

def n_grams(s, n):
    symbols = ",.?!;:"
    for i in symbols:
        s = s.replace(i, "")
    s = s.split(" ")
    result = []
    for i in range(len(s)-n+1):
        res = " ".join(s[i:i+n])
        result.append(res)
    return result

def distinct_n(sentence_list, n):
    all_n_grams = []
    for pair in tqdm(sentence_list):
        all_n_grams += n_grams(pair['hypothesis'], n)
    return len(set(all_n_grams)) / len(all_n_grams)

def eval(data_path, model_path, if_e2e, ref_hypo_f=None, metric="bleu", sentence_list_cache=None):
    if isinstance(metric, list):
        for m in metric:
            eval(data_path, model_path, if_e2e, ref_hypo_f, m, sentence_list_cache)
        return

    if sentence_list_cache and os.path.exists(sentence_list_cache):
        with open(sentence_list_cache, "rb") as fsl:
            sentence_list = pkl.load(fsl, encoding='iso-8859-1')
            print("Loading sentence_list from cache file {}".format(sentence_list_cache))
    else:
        sentence_list = get_eval_data(data_path, model_path, if_e2e, ref_hypo_f, sentence_list_cache)
        
    print("[val]" if "val" in data_path else "[test]" if "test" in data_path else data_path, model_path)

    print(metric)
    
    if metric.lower() == "bleu":
        bleu_scores = count_BLEU(sentence_list)
        metrics = ["1-gram", "2-gram", "3-gram", "4-gram", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]
        for i in range(8):
            print("\t" + metrics[i] + ":", bleu_scores[i].avg)
    
    elif metric.lower() == "sentence":
        cos_sims, euclid_dists, rd_cos_sims, rd_euclid_dists = count_sentence_sim(sentence_list)
        print("Cosine similarity:")
        print("\tAverage:", cos_sims.avg)
        print("\tVariance:", cos_sims.var)
        print("\tStandard Deviation:", cos_sims.sd)
        print("Euclid distance:")
        print("\tAverage:", euclid_dists.avg)
        print("\tVariance:", euclid_dists.var)
        print("\tStandard Deviation:", euclid_dists.sd)
        print("Random cosine similarity:")
        print("\tAverage:", rd_cos_sims.avg)
        print("\tVariance:", rd_cos_sims.var)
        print("\tStandard Deviation:", rd_cos_sims.sd)
        print("Random Euclid distance:")
        print("\tAverage:", rd_euclid_dists.avg)
        print("\tVariance:", rd_euclid_dists.var)
        print("\tStandard Deviation:", rd_euclid_dists.sd)

    elif metric.lower() == "embedding":
        embedding_average, embedding_extreme, embedding_greedy = embedding_sim(sentence_list)
        print("Embedding Average:")
        print("\tAverage:", embedding_average.avg)
        print("\tVariance:", embedding_average.var)
        print("\tStandard Deviation:", embedding_average.sd)
        print("Embedding Extreme:")
        print("\tAverage:", embedding_extreme.avg)
        print("\tVariance:", embedding_extreme.var)
        print("\tStandard Deviation:", embedding_extreme.sd)
        print("Embedding Greedy:")
        print("\tAverage:", embedding_greedy.avg)
        print("\tVariance:", embedding_greedy.var)
        print("\tStandard Deviation:", embedding_greedy.sd)

    elif metric.lower() == "distinct":
        distinct1 = distinct_n(sentence_list, 1)
        distinct2 = distinct_n(sentence_list, 2)
        print("Distinct-1:", distinct1)
        print("Distinct-2:", distinct2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--e2e', action="store_true")
    parser.add_argument('--gpu', dest='gpu', type=str, help="GPU ID", default="0")
    parser.add_argument('--metric', dest='metric', type=str, help="Evaluation metric", default="all")
    parser.add_argument('--pair_path', dest='pair_path', type=str, help="hypo-ref pair file path, default None Type", default=None)
    parser.add_argument('--baseline', action="store_true")
    parser.add_argument('--eval_data_cache', dest='eval_data_cache', type=str, help="evaluation data cache pickle file", default=None)
    args = parser.parse_args()
    gpu_id = args.gpu
    metric = args.metric
    if metric.lower() == "all":
        metric = ["BLEU", "sentence", "embedding", "distinct"]
    hypo_ref_pair_file_path = args.pair_path
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id

    
    if hypo_ref_pair_file_path:
        f = open(hypo_ref_pair_file_path, "w")
    else:
        f = None

    for model_idx in range(50, 0, -1):
        for dataset in ("val", "test"):
            model_path = "./models/PersuaderGPT2_{}".format(model_idx)
            sentence_list_cache = "./eval_data/sentence_list_{}_{}.pkl".format(dataset, model_idx)
            eval("./data/{}.csv".format(dataset), model_path, args.e2e, f, metric, sentence_list_cache)
    
    if f:
        f.close()
