import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import models.GCVIT
from timm.models import create_model
from transformers import  VisualBertConfig, GPT2Config
from transformers import VisualBertModel, GPT2Model, ViTModel, SwinModel, BioGptForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' Late Fusion GPT with CNN/Transformers'''

''' ResNet18 + GPT2 late fusion '''
class GPT2RS18Classification(nn.Module):
    def __init__(self, num_class = 12):
        super(GPT2RS18Classification, self).__init__()

        # text processing
        self.text_feature_extractor = GPT2Model.from_pretrained('gpt2')
 
        # image processing
        self.img_feature_extractor = models.resnet18(pretrained=True)
        new_fc = nn.Sequential(*list(self.img_feature_extractor.fc.children())[:-1])
        self.img_feature_extractor.fc = new_fc

        #intermediate_layers
        self.intermediate_layer = nn.Linear(1280, 512)  #(512+768)
        self.LayerNorm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.1)

        # classifier
        self.classifier = nn.Linear(512, num_class)

    def forward(self, input, img):
        
        # image encoder features
        img_feature = self.img_feature_extractor(img)
        
        # question tokenizer features
        input['input_ids'] = input['input_ids'].to(device)
        input['attention_mask'] = input['attention_mask'].to(device)

        # GPT text encoder
        text_feature = self.text_feature_extractor(**input)
        text_feature = text_feature.last_hidden_state.swapaxes(1,2)
        text_feature = F.adaptive_avg_pool1d(text_feature,1)
        text_feature = text_feature.swapaxes(1,2).squeeze(1)        
        
        # late visual-text fusion
        img_text_features = torch.cat((img_feature, text_feature), dim=1)

        # intermediate layers
        out =self.intermediate_layer(img_text_features)
        out = self.LayerNorm(out)
        out = self.dropout(out)

        # classifier
        out = self.classifier(out)
        # print(out.size())
        return out


''' ViT + GPT2 late fusion '''
class GPT2ViTClassification(nn.Module):
    def __init__(self, num_class = 12):
        super(GPT2ViTClassification, self).__init__()

        # text processing
        self.text_feature_extractor = GPT2Model.from_pretrained('gpt2')
 
        # image processing
        self.img_feature_extractor = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        #intermediate_layers
        self.intermediate_layer = nn.Linear(1536, 512)  #(512+768)
        self.LayerNorm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.1)

        # classifier
        self.classifier = nn.Linear(512, num_class)

    def forward(self, input, img):
        
        # image encoder features
        img['pixel_values'] = img['pixel_values'].to(device)
        img_feature = self.img_feature_extractor(**img)
        
        # question tokenizer features
        input['input_ids'] = input['input_ids'].to(device)
        input['attention_mask'] = input['attention_mask'].to(device)

        # GPT text encoder
        text_feature = self.text_feature_extractor(**input)
        text_feature = text_feature.last_hidden_state.swapaxes(1,2)
        text_feature = F.adaptive_avg_pool1d(text_feature,1)
        text_feature = text_feature.swapaxes(1,2).squeeze(1)        
        
        # late visual-text fusion
        img_text_features = torch.cat((img_feature[0][:, 0, :], text_feature), dim=1)
        # img_text_features = torch.cat((img_feature[1], text_feature), dim=1)

        # intermediate layers
        out =self.intermediate_layer(img_text_features)
        out = self.LayerNorm(out)
        out = self.dropout(out)

        # classifier
        out = self.classifier(out)
        # print(out.size())
        return out


''' Swin + GPT2 late fusion '''
class GPT2SwinClassification(nn.Module):
    def __init__(self, num_class = 12):
        super(GPT2SwinClassification, self).__init__()

        # text processing
        self.text_feature_extractor = GPT2Model.from_pretrained('gpt2')
 
        # image processing
        self.img_feature_extractor = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

        #intermediate_layers
        self.intermediate_layer = nn.Linear(1536, 512)  #(512+768)
        self.LayerNorm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.1)

        # classifier
        self.classifier = nn.Linear(512, num_class)

    def forward(self, input, img):

        # image encoder features
        img['pixel_values'] = img['pixel_values'].to(device)
        img_feature = self.img_feature_extractor(**img)
        
        # print(img_feature[0].size())
        # print(img_feature[1].size())
        
        # question tokenizer features
        input['input_ids'] = input['input_ids'].to(device)
        input['attention_mask'] = input['attention_mask'].to(device)

        # GPT text encoder
        text_feature = self.text_feature_extractor(**input)
        text_feature = text_feature.last_hidden_state.swapaxes(1,2)
        text_feature = F.adaptive_avg_pool1d(text_feature,1)
        text_feature = text_feature.swapaxes(1,2).squeeze(1)        
        
        # late visual-text fusion
        # img_text_features = torch.cat((img_feature[0][:, 0, :], text_feature), dim=1)
        img_text_features = torch.cat((img_feature[1], text_feature), dim=1)

        # intermediate layers
        out =self.intermediate_layer(img_text_features)
        out = self.LayerNorm(out)
        out = self.dropout(out)

        # classifier
        out = self.classifier(out)
        # print(out.size())
        return out

''' ResNet18 + BioGPT2 late fusion '''
class BioGPT2RS18Classification(nn.Module):
    def __init__(self, num_class = 12):
        super(BioGPT2RS18Classification, self).__init__()

        # text processing
        self.text_feature_extractor = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
 
        # image processing
        self.img_feature_extractor = models.resnet18(pretrained=True)
        new_fc = nn.Sequential(*list(self.img_feature_extractor.fc.children())[:-1])
        self.img_feature_extractor.fc = new_fc

        #intermediate_layers
        self.intermediate_layer = nn.Linear(42896, 512)  #(512+768)
        self.LayerNorm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.1)

        # classifier
        self.classifier = nn.Linear(512, num_class)

    def forward(self, input, img):
        
        # image encoder features
        img_feature = self.img_feature_extractor(img)
        
        # question tokenizer features
        input['input_ids'] = input['input_ids'].to(device)
        input['attention_mask'] = input['attention_mask'].to(device)

        # GPT text encoder
        text_feature = self.text_feature_extractor(**input)
        text_feature = text_feature[0].swapaxes(1,2)
        #m: [1, 12, 42384], text feature is too big compare to img. We may pool it to 512 the equal size of img
        #F.adaptive_avg_pool2d(output[0],[1, 512])
        text_feature = F.adaptive_avg_pool1d(text_feature,1) 
        text_feature = text_feature.swapaxes(1,2).squeeze()

        # late visual-text fusion
        #m: advanced level fusion can be used instead of naive concat (e.g., multihead attention fusion)
        img_text_features = torch.cat((img_feature, text_feature), dim=1)

        # intermediate layers
        out =self.intermediate_layer(img_text_features)
        #m: we may add one more intermidiate layer if the features size is bigger
        out = self.LayerNorm(out)
        out = self.dropout(out)

        # classifier
        out = self.classifier(out)
        # print(out.size())
        return out
