'''
Description     : VisualBertResMLP Classification Model
Author          : Lalithkumar Seenivasan, Mobarakol Islam, Adithya Krishna, Hongliang Ren
Paper           : Surgical-VQA: Visual Question Answering in Surgical Scenes Using Transformers
Lab             : MMLAB, National University of Singapore
Acknowledgement : Code adopted from the official implementation of VisualBertModel from 
                  huggingface/transformers (https://github.com/huggingface/transformers.git) and modified.
'''

import torch
from torch import nn
from transformers import VisualBertConfig
from models.VisualBertResMLP import VisualBertResMLPModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
VisualBertResMLP Classification Model
'''
class VisualBertResMLPClassification(nn.Module):
    def __init__(self, vocab_size, layers, n_heads, num_class = 10, token_size = 26):
        super(VisualBertResMLPClassification, self).__init__()
        VBconfig = VisualBertConfig(vocab_size= vocab_size, visual_embedding_dim = 512, num_hidden_layers = layers, num_attention_heads = n_heads, hidden_size = 2048)
        self.VisualBertResMLPEncoder = VisualBertResMLPModel(VBconfig, token_size = token_size)
        self.classifier = nn.Linear(VBconfig.hidden_size, num_class)


    def forward(self, inputs, visual_embeds):
        # prepare visual embedding
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long).to(device)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float).to(device)

        # append visual features to text
        inputs.update({
                        "visual_embeds": visual_embeds,
                        "visual_token_type_ids": visual_token_type_ids,
                        "visual_attention_mask": visual_attention_mask,
                        "output_attentions": True
                        })
                        
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        inputs['visual_token_type_ids'] = inputs['visual_token_type_ids'].to(device)
        inputs['visual_attention_mask'] = inputs['visual_attention_mask'].to(device)

        # Encoder output
        outputs = self.VisualBertResMLPEncoder(**inputs)
        
        # Classification layer
        outputs = self.classifier(outputs['pooler_output'])
        return outputs