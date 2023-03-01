import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from transformers import  VisualBertConfig, GPT2Config
from transformers import VisualBertModel, GPT2Model, ViTModel, SwinModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GCN(nn.Module):
    """
    Graph Convolution network for Global interaction space 
    init    : 
        num_state, num_node, bias=False
        
    forward : x
    """
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, padding=0, stride=1, groups=1, bias=bias)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        # (n, num_state, num_node) -> (n, num_state, num_node)        
        h = self.conv2(self.relu(h))

        return h


class GloRe_Unit1D(nn.Module):
    """
    Global Reasoning Unit (GR/GloRe)
    init    : 
        num_in, num_mid, stride=(1, 1), kernel=1
        
    forward : x
    """    
    def __init__(self, num_in, num_mid, stride=1, kernel=1):
        super(GloRe_Unit1D, self).__init__()

        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)

        kernel_size = kernel
        padding = 1 if kernel == 3 else 0

        # Reduce dimension
        self.conv_state = nn.Conv1d(num_in, self.num_s, kernel_size=kernel_size, padding=padding)
        # generate graph transformation function
        self.conv_proj = nn.Conv1d(num_in, self.num_n, kernel_size=kernel_size, padding=padding)
        # ----------
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        # ----------
        # tail: extend dimension
        self.fc_2 = nn.Conv1d(self.num_s, num_in, kernel_size=kernel_size, padding=padding, stride=1,groups=1, bias=False)

        self.blocker = nn.BatchNorm1d(num_in)

    def forward(self, x):
        '''
        Parameter x dimension : (N, C, H, W)
        '''
        batch_size = x.size(0)
        
        x_token_state = x.permute(0, 2, 1)                  # (n, tokens, token state[2048]) --> (n, 2048, tokens)
        
        x_state_reshaped = self.conv_state(x_token_state)   # (n, nummid, tokens)
        
        x_proj_reshaped = self.conv_proj(x_token_state)     # (n, tokens, numin) --> (n, num_node, tokens)
        
        x_rproj_reshaped = x_proj_reshaped                  # (n, num_node, tokens)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # Projection: Coordinate space -> Interaction space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul( x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        
        # normalize
        x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        # reverse projection: interaction space -> coordinate space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_n_rel = self.gcn(x_n_state)

        # Reverse projection: Interaction space -> Coordinate space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped) 
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # (n, num_state, h*w) --> (n, num_state, h, w)
        # x_state = x_state_reshaped.view(batch_size, self.num_s, *x_token_state.size()[2:])
        
        # -----------------
        # (n, num_state, h, w) -> (n, num_in, h, w)
        x_gr = self.blocker(self.fc_2(x_state_reshaped))


        out = x_token_state + x_gr
        out = out.permute(0,2,1)

        return out




# class GPT2ViTViReClassification(nn.Module):
#     def __init__(self, num_class = 12):
#         super(GPT2ViTViReClassification, self).__init__()

#         # text processing
#         self.text_feature_extractor = GPT2Model.from_pretrained('gpt2')
 
#         # image processing
#         self.img_feature_extractor = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

#         #intermediate_layers
#         self.intermediate_layer = nn.Linear(1536, 512)  #(512+768)
#         self.LayerNorm = nn.BatchNorm1d(512)
#         self.dropout = nn.Dropout(0.1)

#         # classifier
#         self.classifier = nn.Linear(512, num_class)

#     def forward(self, input, img):
        
#         # image encoder features
#         img['pixel_values'] = img['pixel_values'].to(device)
#         img_feature = self.img_feature_extractor(**img)
        
#         # question tokenizer features
#         input['input_ids'] = input['input_ids'].to(device)
#         input['attention_mask'] = input['attention_mask'].to(device)

#         # GPT text encoder
#         text_feature = self.text_feature_extractor(**input)
#         text_feature = text_feature.last_hidden_state.swapaxes(1,2)
#         text_feature = F.adaptive_avg_pool1d(text_feature,1)
#         text_feature = text_feature.swapaxes(1,2).squeeze(1)        
        
#         # late visual-text fusion
#         # img_text_features = torch.cat((img_feature[0][:, 0, :], text_feature), dim=1)
#         img_text_features = torch.cat((img_feature[1], text_feature), dim=1)

#         # intermediate layers
#         out =self.intermediate_layer(img_text_features)
#         out = self.LayerNorm(out)
#         out = self.dropout(out)

#         # classifier
#         out = self.classifier(out)
#         # print(out.size())
#         return out

# class GPT2SwinGRClassification(nn.Module):
#     def __init__(self, num_class = 12):
#         super(GPT2SwinGRClassification, self).__init__()

#         # text processing
#         self.text_feature_extractor = GPT2Model.from_pretrained('gpt2')
 
#         # image processing
#         self.img_feature_extractor = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

#         #intermediate_layers
#         self.GloRe = GloRe_Unit1D(config.hidden_size, 128, stride=(1, 1), kernel=1)
#         self.intermediate_layer = nn.Linear(1536, 512)  #(512+768)
#         self.LayerNorm = nn.BatchNorm1d(512)
#         self.dropout = nn.Dropout(0.1)

#         ## 1D global reasoning
#         self.GloRe = GloRe_Unit1D(768, 128, stride=(1, 1), kernel=1)

#         # classifier
#         self.classifier = nn.Linear(512, num_class)

#     def forward(self, input, img):

#         # image encoder features
#         img['pixel_values'] = img['pixel_values'].to(device)
#         img_feature = self.img_feature_extractor(**img)
        
#         # question tokenizer features
#         input['input_ids'] = input['input_ids'].to(device)
#         input['attention_mask'] = input['attention_mask'].to(device)

#         # GPT text encoder
#         text_feature = self.text_feature_extractor(**input)
#         text_feature = text_feature.last_hidden_state.swapaxes(1,2)
#         text_feature = F.adaptive_avg_pool1d(text_feature,1)
#         text_feature = text_feature.swapaxes(1,2).squeeze(1)        
        
#         # late visual-text fusion
#         img_text_features = torch.cat((img_feature[0][:, 0, :], text_feature), dim=1)
#         # img_text_features = torch.cat((img_feature[1], text_feature), dim=1)

#         # intermediate layers
#         out =self.intermediate_layer(img_text_features)
#         out = self.LayerNorm(out)
#         out = self.dropout(out)

#         # classifier
#         out = self.classifier(out)
#         # print(out.size())
#         return out


class EFGPT2RS18GRClassification(nn.Module):
    def __init__(self, num_class = 12):
        super(EFGPT2RS18GRClassification, self).__init__()

        ## image processing
        self.img_feature_extractor = models.resnet18(pretrained=True)
        # default
        new_fc = nn.Sequential(*list(self.img_feature_extractor.fc.children())[:-1])
        self.img_feature_extractor.fc = new_fc
        # for visual patch
        # self.img_feature_extractor = torch.nn.Sequential(*(list(self.img_feature_extractor.children())[:-2]))

        ## Visual_embedding
        # visual bert embedding
        VB_config = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
        VB_config.visual_embedding_dim = 512
        visualbert = VisualBertModel(config=VB_config)
        self.visual_embedder = visualbert.embeddings.visual_projection
        # random initialiation
        # self.visual_embedder = nn.Linear(512, 768)

        ## Question_embedding
        question_embedder = GPT2Model.from_pretrained('gpt2')
        self.question_embedder = question_embedder.wte

        ## GPT2 visual_cotext_aware_decoder
        self.VCAdecoder = GPT2Model.from_pretrained('gpt2')

        ## 1D global reasoning
        self.GloRe = GloRe_Unit1D(768, 128, stride=(1, 1), kernel=1)
 
        ## intermediate_layers
        self.intermediate_layer = nn.Linear(768, 512)  #(512+768)
        self.LayerNorm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.1)

        ## classifier
        self.classifier = nn.Linear(512, num_class)

    def forward(self, input, img):
        
        ## image encoder features
        img_feature = self.img_feature_extractor(img)
        # default
        img_feature = torch.unsqueeze(img_feature, dim=1)
        # using visual patch
        # img_feature = torch.permute(torch.flatten(img_feature, start_dim=2),(0,2,1))
        
        ## visual Embedding
        visual_embeds = self.visual_embedder(img_feature)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
        visual_attention_mask = visual_attention_mask.to(device)
        
        ## question embedding
        input['input_ids'] = input['input_ids'].to(device)
        input['attention_mask'] = input['attention_mask'].to(device)

        question_embeds = self.question_embedder(input['input_ids'])
        question_attention_mask = input['attention_mask']   

        ## combine visual and question embeds
        inputs_embeds = torch.cat((visual_embeds, question_embeds), dim=1)
        attention_mask = torch.cat((visual_attention_mask, question_attention_mask), dim=1)

        ## VCA_GPT2 decoder
        decoder_output = self.VCAdecoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        decoder_output = self.GloRe(decoder_output['last_hidden_state'])
        decoder_output = decoder_output.swapaxes(1,2) #torch.Size([64, 21, 768]) -> torch.Size([64, 768, 21])
        decoder_output = F.adaptive_avg_pool1d(decoder_output,1) # [64, 768, 1]
        decoder_output = decoder_output.swapaxes(1,2).squeeze(1) # [64, 768]  

        ## intermediate layers
        out =self.intermediate_layer(decoder_output)
        out = self.LayerNorm(out)
        out = self.dropout(out)

        ## classifier
        out = self.classifier(out)
        # print(out.size())
        return out


class EFVLEGPT2SwinGRClassification(nn.Module):
    def __init__(self, num_class = 12, model_subver = 'v0'):
        super(EFVLEGPT2SwinGRClassification, self).__init__()
        '''
        v0: visual embedding, zero image position
        v1: No visual embedding, zero image position
        v2: visual embedding, actual image position
        v3: visual embedding, actual image position
        '''
        
        self.sub_ver = model_subver
        
        ## image processing
        self.img_feature_extractor = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        
        ## Visual_embedding
        # visual bert embedding
        VB_config = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
        VB_config.visual_embedding_dim = 768
        visualbert = VisualBertModel(config=VB_config)
        self.visual_embedder = visualbert.embeddings.visual_projection

        ## Question_embedding
        question_embedder = GPT2Model.from_pretrained('gpt2')
        self.question_embedder = question_embedder.wte

        ## GPT2 visual_cotext_aware_decoder
        self.VCAdecoder = GPT2Model.from_pretrained('gpt2')
        # self.VCAdecoder.wte = nn.Embedding(2, 768)  # later used for position embedding
 
        ## intermediate_layers
        self.intermediate_layer = nn.Linear(768, 512)  #(512+768)
        self.LayerNorm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.1)

        ## 1D global reasoning
        self.GloRe = GloRe_Unit1D(768, 128, stride=(1, 1), kernel=1)

        ## classifier
        self.classifier = nn.Linear(512, num_class)

    def forward(self, input, img):
        
        ## image encoder features
        img['pixel_values'] = img['pixel_values'].to(device)
        img_feature = self.img_feature_extractor(**img)
        
        ## visual Embedding : id type 1, pos: zero / incremental
        if self.sub_ver == 'v0' or self.sub_ver == 'v2':
            visual_embeds = self.visual_embedder(img_feature[0])
        else:
            visual_embeds = img_feature[0]
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
        visual_attention_mask = visual_attention_mask.to(device)

        visual_id_type = torch.ones(*visual_embeds.size()[:-1], dtype=torch.long, device=device)
        if self.sub_ver == 'v0' or self.sub_ver == 'v1':
            visual_position_id = torch.zeros(*visual_embeds.size()[:-1], dtype=torch.long, device=device)
        elif self.sub_ver == 'v2' or self.sub_ver == 'v3':
            visual_position_id = torch.arange(0,visual_embeds.size()[1])
            visual_position_id = torch.unsqueeze(visual_position_id,0)
            visual_position_id = visual_position_id.repeat(visual_embeds.size()[0], 1)
            visual_position_id = visual_position_id.to(device)
        
        
        ## question embedding: id type 0, pose incremental
        input['input_ids'] = input['input_ids'].to(device)
        input['attention_mask'] = input['attention_mask'].to(device)

        question_embeds = self.question_embedder(input['input_ids'])
        question_attention_mask = input['attention_mask']
        
        question_id_type = torch.zeros(*question_embeds.size()[:-1], dtype=torch.long, device=device)
        question_position_id = torch.arange(0,question_embeds.size()[1])
        question_position_id = torch.unsqueeze(question_position_id,0)
        question_position_id = question_position_id.repeat(question_embeds.size()[0], 1)
        question_position_id = question_position_id.to(device)
        

        ## combine visual and question embeds
        inputs_embeds = torch.cat((visual_embeds, question_embeds), dim=1)
        attention_mask = torch.cat((visual_attention_mask, question_attention_mask), dim=1)
        token_type_ids = torch.cat((visual_id_type, question_id_type), dim=1)
        position_ids = torch.cat((visual_position_id, question_position_id), dim=1)

        ## VCA_GPT2 decoder
        decoder_output = self.VCAdecoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids = position_ids, token_type_ids = token_type_ids)
        decoder_output = self.GloRe(decoder_output['last_hidden_state'])
        decoder_output = decoder_output.swapaxes(1,2)
        # decoder_output = decoder_output.last_hidden_state.swapaxes(1,2)
        decoder_output = F.adaptive_avg_pool1d(decoder_output,1)
        decoder_output = decoder_output.swapaxes(1,2).squeeze(1)       

        ## intermediate layers
        out =self.intermediate_layer(decoder_output)
        out = self.LayerNorm(out)
        out = self.dropout(out)

        ## classifier
        out = self.classifier(out)
        # print(out.size())
        return out
