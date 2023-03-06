import torch
from torch import nn
from transformers import VisualBertModel, VisualBertConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
VisualBert Classification Model
'''
class VisualBertClassification(nn.Module):
    '''
    VisualBert Classification Model
    vocab_size    = tokenizer length
    encoder_layer = 6
    n_heads       = 8
    num_class     = number of class in dataset
    '''
    def __init__(self, vocab_size, layers, n_heads, num_class = 10):
        super(VisualBertClassification, self).__init__()
        VBconfig = VisualBertConfig(vocab_size= vocab_size, visual_embedding_dim = 512, num_hidden_layers = layers, num_attention_heads = n_heads, hidden_size = 2048)
        self.VisualBertEncoder = VisualBertModel(VBconfig)
        self.classifier = nn.Linear(VBconfig.hidden_size, num_class)
        
        '--------------------- VQA ---------------------------'
        self.dropout = nn.Dropout(VBconfig.hidden_dropout_prob)
        self.num_labels = num_class
        '--------------------- VQA ---------------------------'


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

        '----------------- VQA -----------------'
        index_to_gather = inputs['attention_mask'].sum(1) - 2  # as in original code # 6
        
        outputs = self.VisualBertEncoder(**inputs)
        sequence_output = outputs[0] # [1, 33, 2048]

        # TO-CHECK: From the original code
        index_to_gather = (index_to_gather.unsqueeze(-1).unsqueeze(-1).expand(index_to_gather.size(0), 1, sequence_output.size(-1))) #  [1, 1, 2048]

        pooled_output = torch.gather(sequence_output, 1, index_to_gather) # [1, 33, 2048]
        
        pooled_output = self.dropout(pooled_output) # [1, 1, 2048]
        logits = self.classifier(pooled_output) # [1, 1, 8]
        reshaped_logits = logits.view(-1, self.num_labels) # [1, 8]
        return reshaped_logits
        '----------------- VQA -----------------'
        
        '----------------- our VQA -----------------'
        # Encoder output
        # outputs = self.VisualBertEncoder(**inputs)
        
        # classification layer
        # outputs = self.classifier(outputs['pooler_output'])
        # return outputs
        '----------------- our VQA -----------------'

