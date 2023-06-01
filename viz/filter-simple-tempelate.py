#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

index = int(os.environ['INDEX'])
slice_name = os.environ['SLICE_NAME']
top_k = int(os.environ['TOPK'])


# In[2]:


import transformers.models.gpt_neox.modeling_gpt_neox

from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from  transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig


# In[3]:


from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
import json


# In[4]:


tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-160m-deduped')


# In[5]:


model = AutoModelForCausalLM.from_pretrained(
    'EleutherAI/pythia-160m-deduped',
)


# In[6]:


model = model.to(f'cuda:{index}')


# In[7]:


model = model.half()


# In[8]:


# from functorch import make_functional_with_buffers, vmap, grad

# fmodel, params, buffers = make_functional_with_buffers(model)

# def compute_loss_stateless_model (params, buffers, input_ids):
#     input_ids = input_ids.unsqueeze(0)
#     labels = input_ids
#     loss = fmodel(params, buffers, input_ids, labels=labels).loss
#     return loss

# inputs = tokenizer('hello world', return_tensors='pt')
# input_ids = inputs['input_ids'][0, :2048].to(model.device)

# compute_loss_stateless_model(params, buffers, input_ids)

# ft_compute_grad = grad(compute_loss_stateless_model)

# %%time
# ft_compute_grad(params, buffers, input_ids)
# torch.cuda.synchronize()

# ft_compute_sample_grad = vmap(ft_compute_grad)

# tokenizer.pad_token = tokenizer.eos_token

# inputs = tokenizer(['hello world', '1,2,3'], return_tensors='pt', padding=True)
# input_ids = inputs['input_ids'][:, :2048].to(model.device)

# ft_per_sample_grads = ft_compute_sample_grad(params, buffers, input_ids, input_ids)


# In[9]:


selected_p = None
for n, p in model.named_parameters():
    if 'layers.11.attention.query_key_value.weight' in n:
        p.requires_grad_(True)
        print(n)
        selected_p = p
    else:
        p.requires_grad_(False)


# In[10]:


def compute_loss_model (text):
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids'][:, :2048].to(model.device)
    labels = input_ids
    loss = model(input_ids, labels=labels).loss
    return loss


# In[11]:


data = load_dataset('json', data_files=f'to_target/{slice_name}_to_target_{top_k}.jsonl', split='train')


# In[12]:


q_data = load_dataset('json', data_files='target-train.jsonl', split='train')


# In[13]:


len(q_data)


# In[14]:


q_data = q_data.shard(8, index)


# In[15]:


q_id_to_t_items = {}


# In[16]:


for item in data:
    q_id = item['meta']['query_meta']['id']
    if q_id not in q_id_to_t_items:
        q_id_to_t_items[q_id] = []
    q_id_to_t_items[q_id].append(item)


# In[17]:


def grad_trial(model, para_grad, text):
    model.zero_grad()
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids'][:, :2048].to(model.device)
    loss = model(input_ids, labels=input_ids).loss
    grad = torch.autograd.grad(loss, para_grad)[0]
    # loss.backward()
    # grad = para_grad.grad.clone()
    return grad


# In[18]:


import tqdm


# In[ ]:


with open(f'to_target/rerank_{slice_name}_to_target_{index}-8.jsonl', 'w') as f:
    
    # with torch.autocast(device_type="cuda", enabled=False):

        for q_item in tqdm.tqdm(q_data):
            q_id = q_item['meta']['id']
            q_grad = grad_trial(model, selected_p, q_item['text'])

            # print(q_grad)

            t_items = q_id_to_t_items[q_id]
            t_scores = []
            for t_item in t_items:
                t_grad = grad_trial(model, selected_p, t_item['text'])
                # print(t_grad)
                t_score = (q_grad * t_grad).sum().item()
                t_scores.append(t_score)

            sorted_t_items = sorted(zip(t_scores, t_items), key=lambda x: -x[0])

            for _i, (_s, item) in enumerate(sorted_t_items):
                item['meta']['grad_rank'] = _i
                item['meta']['grad_rank_score'] = _s

                f.write(json.dumps(item) + '\n')


# In[ ]:




