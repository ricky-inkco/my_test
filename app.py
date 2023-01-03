#!/usr/bin/env python

import argparse
#import connexion
import os
import yaml
from flask import send_from_directory, redirect,request,Flask
#from flask_cors import CORS

# from backend.Project import Project # TODO !!
from backend import AVAILABLE_MODELS
import numpy as np
import torch
import time
from flask import request
import json
from transformers import (GPT2LMHeadModel, GPT2Tokenizer,
                          BertTokenizer, BertForMaskedLM)
from flask import jsonify, make_response
import re

__author__ = 'Hendrik Strobelt, Sebastian Gehrmann'

CONFIG_FILE_NAME = 'lmf.yml'
projects = {}

#app = connexion.App(__name__, debug=False)
app = Flask(__name__)


class Project:
    def __init__(self, LM, config):
        self.config = config
        self.lm = LM()


def get_all_projects():
    res = {}
    for k in projects.keys():
        res[k] = projects[k].config
    return res


def analyze(analyze_request):
    project = analyze_request.get('project')
    text = analyze_request.get('text')

    res = {}
    if project in projects:
        p = projects[project] # type: Project
        res = p.lm.check_probabilities(text, topk=20)

    return {
        "request": {'project': project, 'text': text},
        "result": res
    }

class AbstractLanguageChecker:
    """
    Abstract Class that defines the Backend API of GLTR.

    To extend the GLTR interface, you need to inherit this and
    fill in the defined functions.
    """

    def __init__(self):
        """
        In the subclass, you need to load all necessary components
        for the other functions.
        Typically, this will comprise a tokenizer and a model.
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def check_probabilities(self, in_text, topk=40):
        """
        Function that GLTR interacts with to check the probabilities of words

        Params:
        - in_text: str -- The text that you want to check
        - topk: int -- Your desired truncation of the head of the distribution

        Output:
        - payload: dict -- The wrapper for results in this function, described below

        Payload values
        ==============
        bpe_strings: list of str -- Each individual token in the text
        real_topk: list of tuples -- (ranking, prob) of each token
        pred_topk: list of list of tuple -- (word, prob) for all topk
        """
        raise NotImplementedError

    def postprocess(self, token):
        """
        clean up the tokens from any special chars and encode
        leading space by UTF-8 code '\u0120', linebreak with UTF-8 code 266 '\u010A'
        :param token:  str -- raw token text
        :return: str -- cleaned and re-encoded token text
        """
        raise NotImplementedError


def top_k_logits(logits, k):
    """
    Filters logits to only the top k choices
    from https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_gpt2.py
    """
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values,
                       torch.ones_like(logits, dtype=logits.dtype) * -1e10,
                       logits)

class LM(AbstractLanguageChecker):
    def __init__(self, model_name_or_path="gpt2"):
        super(LM, self).__init__()
        self.enc = GPT2Tokenizer.from_pretrained(model_name_or_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()
        self.start_token = self.enc(self.enc.bos_token, return_tensors='pt').data['input_ids'][0]
        print("Loaded GPT-2 model!")

    def check_probabilities(self, in_text, topk=40):
        # Process input
        token_ids = self.enc(in_text, return_tensors='pt').data['input_ids'][0]
        token_ids = torch.concat([self.start_token, token_ids])
        # Forward through the model
        output = self.model(token_ids.to(self.device))
        all_logits = output.logits[:-1].detach().squeeze()
        # construct target and pred
        # yhat = torch.softmax(logits[0, :-1], dim=-1)
        all_probs = torch.softmax(all_logits, dim=1)

        y = token_ids[1:]
        # Sort the predictions for each timestep
        sorted_preds = torch.argsort(all_probs, dim=1, descending=True).cpu()
        # [(pos, prob), ...]
        real_topk_pos = list(
            [int(np.where(sorted_preds[i] == y[i].item())[0][0])
             for i in range(y.shape[0])])
        real_topk_probs = all_probs[np.arange(
            0, y.shape[0], 1), y].data.cpu().numpy().tolist()
        real_topk_probs = list(map(lambda x: round(x, 5), real_topk_probs))

        real_topk = list(zip(real_topk_pos, real_topk_probs))
        # [str, str, ...]
        bpe_strings = self.enc.convert_ids_to_tokens(token_ids[:])

        bpe_strings = [self.postprocess(s) for s in bpe_strings]

        topk_prob_values, topk_prob_inds = torch.topk(all_probs, k=topk, dim=1)

        pred_topk = [list(zip(self.enc.convert_ids_to_tokens(topk_prob_inds[i]),
                              topk_prob_values[i].data.cpu().numpy().tolist()
                              )) for i in range(y.shape[0])]
        pred_topk = [[(self.postprocess(t[0]), t[1]) for t in pred] for pred in pred_topk]


        # pred_topk = []
        payload = {'bpe_strings': bpe_strings,
                   'real_topk': real_topk,
                   'pred_topk': pred_topk}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return payload

    def sample_unconditional(self, length=100, topk=5, temperature=1.0):
        '''
        Sample `length` words from the model.
        Code strongly inspired by
        https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_gpt2.py

        '''
        context = torch.full((1, 1),
                             self.enc.encoder[self.start_token],
                             device=self.device,
                             dtype=torch.long)
        prev = context
        output = context
        past = None
        # Forward through the model
        with torch.no_grad():
            for i in range(length):
                logits, past = self.model(prev, past=past)
                logits = logits[:, -1, :] / temperature
                # Filter predictions to topk and softmax
                probs = torch.softmax(top_k_logits(logits, k=topk),
                                      dim=-1)
                # Sample
                prev = torch.multinomial(probs, num_samples=1)
                # Construct output
                output = torch.cat((output, prev), dim=1)

        output_text = self.enc.decode(output[0].tolist())
        return output_text

    def postprocess(self, token):
        with_space = False
        with_break = False
        if token.startswith('Ġ'):
            with_space = True
            token = token[1:]
            # print(token)
        elif token.startswith('â'):
            token = ' '
        elif token.startswith('Ċ'):
            token = ' '
        elif token.startswith('Ġ'):
            token= ' '
            with_break = True

        token = '-' if token.startswith('â') else token
        token = '“' if token.startswith('ľ') else token
        token = '”' if token.startswith('Ŀ') else token
        token = "'" if token.startswith('Ļ') else token
        token = " " if token.startswith('Ġ') else token

        if with_space:
            token = '\u0120' + token
        if with_break:
            token = '\u010A' + token

        return token

#########################
#  some non-logic routes
#########################


@app.route('/')
def redir():
    return redirect('client/index.html')

@app.route('/predict',methods=['GET','POST'])
def my_test():
    content=request.json
    input_text=content['text']
    input_text=input_text.strip()
    testing_text = input_text
    
    lm = LM()
    payload = lm.check_probabilities(testing_text, topk=5)  
    result=payload['pred_topk']
    topk_data=payload['real_topk']
    

    #convert the tuple into list
    list_data = []
    # Iterate over the elements in the data list
    for element in result:
        # Create a new list for each element
        inner_list = []
        # Iterate over the tuples in the element
        for tuple_ in element:
            # Append the tuple to the inner list as a list
            inner_list.append(list(tuple_))
        # Append the inner list to the list_data list
        list_data.append(inner_list)
    
    #remove unnecessay character in result
    for outer_lst in list_data:
        for inner_lst in outer_lst:
            inner_lst[0] = re.sub(r'Ġ', '', inner_lst[0])
    
    #calculate the max_values foreach result
    max_values = []
    
    for lst in list_data:
        max_value = max(lst, key=lambda x: x[1])
        max_values.append(max_value[1])
    #print(max_values)
    # Use a regular expression to split the string
    words = re.split(r'[\s.]', testing_text)


    #words.pop()
    if input_text.endswith('.'):
        words.append('.')

    #get topk prediction
    topk=[]
    for i in range(len(topk_data)):
        topk.append(topk_data[i][1])
    missing=len(result)-len(testing_text.split())
    for i in range(0,missing):
        words.append("")
    
    #calculate fracp
    st=0
    data_frac=[]
    a,b,c=np.shape(list_data)
    for i in range(a):
        for j in range(b):
            d=list_data[i][j][0]
            #print(d)
            if words[st]==d:
                frac=topk[st]/max_values[st]
                data_frac.append(frac)
            elif words[st]!=d and topk[st]!=max_values[st]:
                frac=topk[st]/max_values[st]
                data_frac.append(frac)
            elif words[st]!=d and topk[st]==max_values[st]:
                frac=topk[st]
                #data_frac.append(topk[st])
                data_frac.append(frac)
                
        st=st+1
    fracp = []

    # Iterate over the numbers in the list
    for number in data_frac:
        # If the number is not already in the unique numbers list, add it
        if number not in fracp:
            fracp.append(number)
    final_fracp=np.median(fracp)
    return make_response(jsonify({'fracp':final_fracp}), 200)
    


if __name__ == '__main__':
    #args = parser.parse_args()
    #app.run(port=int(args.port), debug=not args.nodebug, host=args.address)
    app.run(debug=True,host="0.0.0.0",port=5000)
