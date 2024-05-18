import torch
from transformers import BertTokenizer, GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
import torch.nn.functional as F
from util import add_special_tokens_
import random
from model import BERTCls
import os
import argparse

label_name=[
    'provide-org-facts',
    'personal-story',
    "task-related-inquiry",
    'provide-donation-procedure',
    'non-strategy',
    "have-you-heard-of-the-org",
    "logical-appeal",
    "foot-in-the-door",
    'personal-related-inquiry',
    "emotion-appeal",
    "example-donation",
]

def get_policy():
    idx = random.randint(0,10)
    return label_name[idx]

class PolicyClassifier:
    def __init__(self, model_name_or_path, mode=0, max_len=512):
        '''mode: 0/1/2/3 -> this_turn / all_his including this turn / all_his not including this turn / multilabel'''
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BERTCls(this_turn=(mode == 0))
        state_dict = torch.load(model_name_or_path)
        self.model.load_state_dict(state_dict=state_dict)
        self.model.eval()
        self.mode = mode
        self.max_len = max_len

        self.turn = 0
        self.history = str()
        if mode == 0:
            self.this_turn = "Persuader:"

        print("<CLS>:", self.tokenizer.cls_token_id)
        print("<SEP>:", self.tokenizer.sep_token_id)

    def predict(self, input_text):
        x_ids = self.tokenizer.encode(input_text)
        x_ids = torch.tensor(x_ids).long()
        x_ids = x_ids.unsqueeze(0)
        # print(x_ids)

        his_ids = self.tokenizer.encode(self.history)
        his_ids = his_ids[-self.max_len : ]
        his_ids = torch.tensor(his_ids).long()
        his_ids = his_ids.unsqueeze(0)
        # print(his_ids.shape)

        turn = torch.tensor(self.turn).long()
        turn = turn.unsqueeze(0)
        # print(turn)

        if self.mode == 0:
            tt_ids = self.tokenizer.encode(self.this_turn)
            tt_ids = torch.tensor(tt_ids).long()
            tt_ids = tt_ids.unsqueeze(0)
            # print(tt_ids.shape)

            logits = self.model(x_ids=x_ids, turn=turn, his_ids=his_ids, this_turn_ids=tt_ids)
        else:
            logits = self.model(x_ids=x_ids, turn=turn, his_ids=his_ids)

        # print(logits.shape)

        logits[0][4] = -float('Inf')

        policy_id = torch.argmax(logits[0], dim = 0)
        print("policy:", policy_id, label_name[policy_id])

        return label_name[policy_id]
    
    def update(self, input_text=None, generated_text=None, add_turn=False):
        if self.mode == 0:
            if input_text:
                self.history += "\nPersuadee: " + input_text
            if generated_text:
                self.this_turn += " " + generated_text
            if add_turn:
                self.turn += 1
                self.history += self.this_turn
                self.this_turn = "\nPersuader:"

        # print("Policy Model Update:")
        # print("\t[Turn]", self.turn)
        # print("\t[history]", self.history)
        # print("\t[this turn]", self.this_turn)
        

class PersuaderGPT:
    def __init__(self, model_name_or_path, max_len=1000):
        self.tokenizer = GPT2Tokenizer.from_pretrained("./models/PersuaderGPT2_tokenizer")
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

        self.SPECIAL_TOKENS = ["<cls>", "<sep>", "<persuadee>", "<persuader>", "<pad>", "<persona_begin>", "<persona_end>", "<sentiment_begin>", "<sentiment_end>", "<policy_begin>", "<policy_end>"]
        # ATTR_TO_SPECIAL_TOKEN = {'cls_token': '<cls>', 'sep_token': '<sep>', 'pad_token': '<pad>', 'additional_special_tokens': ['<persuadee>', '<persuader>', '<persona_begin>', '<persona_end>', '<sentiment_begin>', '<sentiment_end>', "<policy_begin>", "<policy_end>"]}
        # add_special_tokens_(self.model, self.tokenizer, ATTR_TO_SPECIAL_TOKEN)
        
        # self.model.load_state_dict(torch.load(model_name_or_path))
        self.model.eval()
        self.max_len = max_len

        self.cls_id, self.sep_id, self.ee, self.er, self.pad_id, self.persona_begin, self.persona_end, self.sentiment_begin, self.sentiment_end, self.policy_begin, self.policy_end = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS[:])
        self.input_ids = []

    def generate(self):
        if len(self.input_ids) >= self.max_len:
            self.input_ids = self.input_ids[-(self.max_len-100):]

        # print("length:", len(self.input_ids))
        input = torch.tensor(self.input_ids).long()
        input = input.unsqueeze(0)
        response = []

        for _ in range(self.max_len):
            outputs = self.model(input)
            logits = outputs.logits
            # print(logits.shape)
            next_token_logits = logits[0, -1, :]
            for token in self.SPECIAL_TOKENS:
                if token != '<sep>':
                    next_token_logits[self.tokenizer.convert_tokens_to_ids(token)] = -float('Inf')
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            if next_token == self.sep_id:
                break
            response.append(next_token.item())
            input = torch.cat((input, next_token.unsqueeze(0)), dim=1)
            if input.shape[1] >= self.max_len:
                input = input[:, -(self.max_len-100):]

        self.input_ids += (response + [self.sep_id])
        if len(self.input_ids) >= self.max_len:
            self.input_ids = self.input_ids[-(self.max_len-100):]
        response_tokens = self.tokenizer.convert_ids_to_tokens(response)
        return "".join(response_tokens).replace("Ġ", " ")
        

    def greeting(self):
        self.input_ids += [self.cls_id, self.er]

        return self.generate()
    
    def update_user_input(self, input_text):
        text_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(input_text))
        self.input_ids += [self.ee] + text_ids + [self.sep_id]

    def predict(self, policy='non-strategy'):
        self.input_ids += [self.er]

        if policy != 'non-strategy':
            policy_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(policy))
            self.input_ids += [self.policy_begin] + policy_ids + [self.policy_end]

        return self.generate()
    
    def check(self):
        dialogue_tokens = self.tokenizer.convert_ids_to_tokens(self.input_ids)
        text = "".join(dialogue_tokens).replace("Ġ", " ")
        text = text.replace("<persuader><policy_begin>", "\n[Model (policy: ").replace("<policy_end>", ")] ")
        text = text.replace("<persuader>", "\n[Model] ").replace("<persuadee>", "\n[User] ").replace("<cls>", "").replace("<sep>", "")
        return text
    
    def clear(self):
        self.input_ids = []
    

class PersuaderForGoodModel():
    def __init__(self, policy_model_name_or_path, gpt_model_name_or_path, policy_model_mode=0, gpt_max_len=1024, policy_max_len=512):
        self.policy_model = PolicyClassifier(policy_model_name_or_path, policy_model_mode, policy_max_len)
        self.gpt_model = PersuaderGPT(gpt_model_name_or_path, gpt_max_len)

    def conversation(self):
        thx_count = 0

        model_text = self.gpt_model.greeting()
        print("[Model]", model_text)
        self.policy_model.update(generated_text=model_text, add_turn=True)
        for turn in range(1, 15):
            input_text = input("[User] ")
            if input_text in ('break', 'end'):
                break
            self.policy_model.update(input_text=input_text)
            self.gpt_model.update_user_input(input_text)

            model_text1 = self.gpt_model.predict()
            self.policy_model.update(generated_text=model_text1)

            policy = self.policy_model.predict(input_text)
            model_text2 = self.gpt_model.predict(policy=policy)
            
            self.policy_model.update(generated_text=model_text2, add_turn=True)

            print("[Model]", (model_text1 + ". " + model_text2).replace("..", ".").replace("?.", "?").replace("!.", "!").replace("  ", " "))

            end_dialogue = False
            for s in ("Thank you again", "Thanks again"):
                if s in (model_text1 + model_text2):
                    end_dialogue = True
                    break
            if end_dialogue:
                break

        print(self.gpt_model.check())

class E2E_PersuaderGPT:
    def __init__(self, model_name_or_path, max_len=1000):
        self.tokenizer = GPT2Tokenizer.from_pretrained("./models/PersuaderGPT2_tokenizer")
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

        self.SPECIAL_TOKENS = ["<cls>", "<sep>", "<persuadee>", "<persuader>", "<pad>", "<persona_begin>", "<persona_end>", "<sentiment_begin>", "<sentiment_end>", "<policy_begin>", "<policy_end>"]
        # ATTR_TO_SPECIAL_TOKEN = {'cls_token': '<cls>', 'sep_token': '<sep>', 'pad_token': '<pad>', 'additional_special_tokens': ['<persuadee>', '<persuader>', '<persona_begin>', '<persona_end>', '<sentiment_begin>', '<sentiment_end>', "<policy_begin>", "<policy_end>"]}
        # add_special_tokens_(self.model, self.tokenizer, ATTR_TO_SPECIAL_TOKEN)
        
        # self.model.load_state_dict(torch.load(model_name_or_path))
        self.model.eval()
        self.max_len = max_len

        self.cls_id, self.sep_id, self.ee, self.er, self.pad_id, self.persona_begin, self.persona_end, self.sentiment_begin, self.sentiment_end, self.policy_begin, self.policy_end = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS[:])
        self.input_ids = []

    def generate(self):
        if len(self.input_ids) >= self.max_len:
            self.input_ids = self.input_ids[-(self.max_len-100):]

        # print("length:", len(self.input_ids))
        input = torch.tensor(self.input_ids).long()
        input = input.unsqueeze(0)
        response = []

        for _ in range(self.max_len):
            outputs = self.model(input)
            logits = outputs.logits
            # print(logits.shape)
            next_token_logits = logits[0, -1, :]
            for token in self.SPECIAL_TOKENS:
                if token != '<sep>':
                    next_token_logits[self.tokenizer.convert_tokens_to_ids(token)] = -float('Inf')
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            if next_token == self.sep_id:
                break
            response.append(next_token.item())
            input = torch.cat((input, next_token.unsqueeze(0)), dim=1)
            if input.shape[1] >= self.max_len:
                input = input[:, -(self.max_len-100):]

        self.input_ids += (response + [self.sep_id])
        if len(self.input_ids) >= self.max_len:
            self.input_ids = self.input_ids[-(self.max_len-100):]
        response_tokens = self.tokenizer.convert_ids_to_tokens(response)
        return "".join(response_tokens).replace("Ġ", " ")
        

    def greeting(self):
        self.input_ids += [self.cls_id, self.er]

        return self.generate()
    
    def update_user_input(self, input_text):
        text_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(input_text))
        self.input_ids += [self.ee] + text_ids + [self.sep_id]

    def predict(self):
        self.input_ids += [self.er]

        return self.generate()
    
    def check(self):
        dialogue_tokens = self.tokenizer.convert_ids_to_tokens(self.input_ids)
        text = "".join(dialogue_tokens).replace("Ġ", " ")
        text = text.replace("<persuader><policy_begin>", "\n[Model (policy: ").replace("<policy_end>", ")] ")
        text = text.replace("<persuader>", "\n[Model] ").replace("<persuadee>", "\n[User] ").replace("<cls>", "").replace("<sep>", "")
        return text
    
    def clear(self):
        self.input_ids = []

class E2E_PersuaderForGoodModel():
    def __init__(self, gpt_model_name_or_path, gpt_max_len=1024):
        self.gpt_model = E2E_PersuaderGPT(gpt_model_name_or_path, gpt_max_len)

    def conversation(self):
        thx_count = 0

        model_text = self.gpt_model.greeting()
        print("[Model]", model_text)
        for turn in range(1, 15):
            input_text = input("[User] ")
            if input_text in ('break', 'end'):
                break
            self.gpt_model.update_user_input(input_text)

            model_text1 = self.gpt_model.predict()

            model_text2 = self.gpt_model.predict()

            print("[Model]", (model_text1 + ". " + model_text2).replace("..", ".").replace("?.", "?").replace("!.", "!").replace("  ", " "))

            end_dialogue = False
            for s in ("Thank you again", "Thanks again"):
                if s in (model_text1 + model_text2):
                    end_dialogue = True
                    break
            if end_dialogue:
                break

        print(self.gpt_model.check())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e", action="store_true")
    parser.add_argument('--p', dest='p', type=str, help="Policy model name or path", default="./models/policy_classifier_0-ce.pkl")
    parser.add_argument("--g", dest='g', type=str, help="GPT model name or path", default="./models/PersuaderGPT2_20_all")
    parser.add_argument('--gpu', dest='gpu', type=str, help="GPU ID", default="0")
    args = parser.parse_args()
    gpu_id = args.gpu
    policy_model_name_or_path = args.p
    gpt_model_name_or_path = args.g
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id
    if args.e2e:
        persuader = E2E_PersuaderForGoodModel(gpt_model_name_or_path)
    else:
        persuader = PersuaderForGoodModel(policy_model_name_or_path, gpt_model_name_or_path)
    persuader.conversation()
    