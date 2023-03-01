import torch
from torch import nn
from transformers import VisualBertConfig
from models.ViReVisualbert import ViReVisualBertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
VisualBert Classification Model
'''
class ViReVisualBertClassification(nn.Module):
    '''
    VisualBert Classification Model
    vocab_size    = tokenizer length
    encoder_layer = 6
    n_heads       = 8
    num_class     = number of class in dataset
    '''
    def __init__(self, vocab_size, layers, n_heads, num_class = 10, version='v1'):
        super(ViReVisualBertClassification, self).__init__()
        VBconfig = VisualBertConfig(vocab_size= vocab_size, visual_embedding_dim = 512, num_hidden_layers = layers, num_attention_heads = n_heads, hidden_size = 2048)
        
        self.ViReVisualBertEncoder = ViReVisualBertModel(VBconfig, version)
        
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
        
        outputs = self.ViReVisualBertEncoder(**inputs)
        sequence_output = outputs[0] # [1, 33, 2048]

        # TO-CHECK: From the original code
        index_to_gather = (index_to_gather.unsqueeze(-1).unsqueeze(-1).expand(index_to_gather.size(0), 1, sequence_output.size(-1))) #  [1, 1, 2048]

        pooled_output = torch.gather(sequence_output, 1, index_to_gather) # [1, 33, 2048]
        
        pooled_output = self.dropout(pooled_output) # [1, 1, 2048]
        logits = self.classifier(pooled_output) # [1, 1, 8]
        reshaped_logits = logits.view(-1, self.num_labels) # [1, 8]
        return reshaped_logits
        '----------------- VQA -----------------'
        
        # '----------------- our VQA -----------------'
        # # Encoder output
        # outputs = self.ViReVisualBertEncoder(**inputs)
        
        # # classification layer
        # outputs = self.classifier(outputs['pooler_output'])
        # return outputs
        # '----------------- our VQA -----------------'
        

        
# import torch
# from transformers import BertTokenizer

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = ViReVisualBertClassification(vocab_size=len(tokenizer), layers=6, n_heads=8, num_class = 8, visual_token_size=25)
# model = model.to(device)

# inputs = tokenizer("What is the man eating?", return_tensors="pt")
# # this is a custom function that returns the visual embeddings given the image path
# visual_embeds = torch.rand(1,25, 512).to(device)

# outputs = model(inputs, visual_embeds)




# class VisualBertForQuestionAnswering(VisualBertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels

#         self.visual_bert = VisualBertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.cls = nn.Linear(config.hidden_size, config.num_labels)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def forward(self, input_ids: Optional[torch.LongTensor] = None,
#                 attention_mask: Optional[torch.LongTensor] = None,
#                 token_type_ids: Optional[torch.LongTensor] = None, position_ids: Optional[torch.LongTensor] = None, head_mask: Optional[torch.LongTensor] = None,
#                 inputs_embeds: Optional[torch.FloatTensor] = None,
#                 visual_embeds: Optional[torch.FloatTensor] = None, visual_attention_mask: Optional[torch.LongTensor] = None, visual_token_type_ids: Optional[torch.LongTensor] = None,
#                 image_text_alignment: Optional[torch.LongTensor] = None,
#                 output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None,
#                 return_dict: Optional[bool] = None, labels: Optional[torch.LongTensor] = None,) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, total_sequence_length)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. A KLDivLoss is computed between the labels and the returned logits.

#         Returns:

#         Example:

#         ```python
#         # Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.
#         from transformers import BertTokenizer, VisualBertForQuestionAnswering
#         import torch

#         tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#         model = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa")

#         text = "Who is eating the apple?"
#         inputs = tokenizer(text, return_tensors="pt")
#         visual_embeds = get_visual_embeddings(image).unsqueeze(0)
#         visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
#         visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

#         inputs.update(
#             {
#                 "visual_embeds": visual_embeds,
#                 "visual_token_type_ids": visual_token_type_ids,
#                 "visual_attention_mask": visual_attention_mask,
#             }
#         )

#         labels = torch.tensor([[0.0, 1.0]]).unsqueeze(0)  # Batch size 1, Num labels 2

#         outputs = model(**inputs, labels=labels)
#         loss = outputs.loss
#         scores = outputs.logits
#         ```"""
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         # Get the index of the last text token
#         index_to_gather = attention_mask.sum(1) - 2  # as in original code

#         outputs = self.visual_bert(input_ids, attention_mask=attention_mask, 
#                                 token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
#                                 visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask, visual_token_type_ids=visual_token_type_ids,
#                                 image_text_alignment=image_text_alignment, 
#                                 output_attentions=output_attentions, output_hidden_states=output_hidden_states,
#                                 return_dict=return_dict,
#                             )

#         sequence_output = outputs[0]

#         # TO-CHECK: From the original code
#         index_to_gather = (index_to_gather.unsqueeze(-1).unsqueeze(-1).expand(index_to_gather.size(0), 1, sequence_output.size(-1)))
#         pooled_output = torch.gather(sequence_output, 1, index_to_gather)

#         pooled_output = self.dropout(pooled_output)
#         logits = self.cls(pooled_output)
#         reshaped_logits = logits.view(-1, self.num_labels)

#         loss = None
#         if labels is not None:
#             loss_fct = nn.KLDivLoss(reduction="batchmean")
#             log_softmax = nn.LogSoftmax(dim=-1)
#             reshaped_logits = log_softmax(reshaped_logits)
#             loss = loss_fct(reshaped_logits, labels.contiguous())
#         if not return_dict:
#             output = (reshaped_logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return SequenceClassifierOutput(loss=loss, logits=reshaped_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions,)

