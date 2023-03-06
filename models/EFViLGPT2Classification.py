import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from timm.models import create_model
from transformers import  VisualBertConfig, GPT2Config
from transformers import VisualBertModel, GPT2Model, ViTModel, SwinModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' Custom embedding GPT with CNN/Transformers (**** implementation)'''


class ViLEmbeddings(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        # For Visual Features
        # Token type and position embedding for image features
        self.visual_token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.visual_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        if config.special_visual_initialize:
            self.visual_token_type_embeddings.weight.data = nn.Parameter(self.token_type_embeddings.weight.data.clone(), requires_grad=True)
            self.visual_position_embeddings.weight.data = nn.Parameter(self.position_embeddings.weight.data.clone(), requires_grad=True)

        self.visual_projection = nn.Linear(config.visual_embedding_dim, config.hidden_size)


    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, visual_embeds=None, visual_token_type_ids=None, image_text_alignment=None,):

        input_shape = input_ids.size()
        seq_length = input_shape[1]
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings

        # Absolute Position Embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        visual_embeds = self.visual_projection(visual_embeds)
        visual_token_type_embeddings = self.visual_token_type_embeddings(visual_token_type_ids)
        visual_position_ids = torch.zeros(*visual_embeds.size()[:-1], dtype=torch.long, device=visual_embeds.device)
        visual_position_embeddings = self.visual_position_embeddings(visual_position_ids)
        visual_embeddings = visual_embeds + visual_position_embeddings + visual_token_type_embeddings
        embeddings = torch.cat((embeddings, visual_embeddings), dim=1)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings    


class ViLGPT2VQA(nn.Module):
    def __init__(self, num_class=2, config_emb=None):
        super(ViLGPT2VQA, self).__init__()
        
        # visual features
        model_visual_feat = models.resnet50(pretrained=True)
        model_visual_feat.avgpool = nn.Identity()
        model_visual_feat.fc = nn.Identity()
        self.model_visual_features = model_visual_feat.eval()


        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.config = GPT2Config.from_pretrained("gpt2")
        self.classifier = nn.Linear(100 * 768, num_class)

        self.config_emb = config_emb
        self.config_emb.visual_embedding_dim = 2048 #most right dim of the visual features
        self.config_emb.hidden_size = self.config.hidden_size
        self.config_emb.vocab_size = self.config.vocab_size 
        self.config_emb.pad_token_id = self.config.pad_token_id 

        self.embeddings = ViLEmbeddings(config=self.config_emb)
        # self.embeddings = self.visualbert.embeddings

    def forward(self, inputs, img):

        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)

        #visual feature
        with torch.no_grad():
            visual_embeds = self.model_visual_features(img)
            visual_embeds = visual_embeds.view(-1, 80, 2048)
            visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long, device=device)
            visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float, device=device)

        # inputs = tokenizer(questions, return_tensors="pt", padding="max_length",max_length=20,)#[2 20]
        token_type_ids = torch.zeros(inputs['input_ids'].shape, dtype=torch.long, device=device)

        inputs.update(
                {
                    "token_type_ids": token_type_ids,
                    "visual_embeds": visual_embeds, #[2, 80, 2048]
                    "visual_token_type_ids": visual_token_type_ids,
                    "visual_attention_mask": visual_attention_mask,
                }
            )


        hidden_states = self.embeddings(
            input_ids=inputs['input_ids'],
            token_type_ids=inputs['token_type_ids'],
            position_ids=None,
            inputs_embeds=None,
            visual_embeds=inputs['visual_embeds'],
            visual_token_type_ids=inputs['visual_token_type_ids'],
            image_text_alignment=None,
        )

        hidden_states = self.gpt2.drop(hidden_states)
        input_shape = inputs['input_ids'].size()
        visual_input_shape = inputs['visual_embeds'].size()[:-1]
        combined_attention_mask = torch.cat((inputs['attention_mask'], inputs['visual_attention_mask']), dim=-1)
        extended_attention_mask: torch.Tensor = self.gpt2.get_extended_attention_mask(combined_attention_mask, (input_shape[0], input_shape + visual_input_shape), device = device)
        output_attentions = self.config.output_attentions
        head_mask = self.gpt2.get_head_mask(None, self.config.n_layer)
        past_key_values = tuple([None] * len(self.gpt2.h))
        for i, (block, layer_past) in enumerate(zip(self.gpt2.h, past_key_values)):
            outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    use_cache=None,
                    output_attentions=output_attentions,
                )
            
            hidden_states = outputs[0]

        hidden_states = self.gpt2.ln_f(hidden_states) #[2, 59, 768]
        x = torch.flatten(hidden_states, 1) 
        # print(inputs['input_ids'].size(), visual_embeds.size(), hidden_states.size(), x.size(), )
        x = self.classifier(x)
        return x
