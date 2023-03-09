'''
class specific evaluation
'''
import os
import glob
import h5py
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import ViTFeatureExtractor, AutoFeatureExtractor


'''
EndoVis18 classification dataloader for GPT2 + ResNet18
'''
class EndoVis18VQACSGPTClassification(Dataset):
    '''
    	seq: train_seq  = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
    	     val_seq    = [1, 5, 16]
    '''
    def __init__(self, seq, model_ver = None, transform=None):
        
        if model_ver == "efvlegpt2ViT": 
            self.transform = None
            self.image_processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        elif model_ver == "efvlegpt2Swin": 
            self.transform = None
            self.image_processor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        elif transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                                    transforms.Resize((300,256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                    ])        

        self.vqas = []
        # files, question and answers #seq_1/frame110|What is the state of bipolar_forceps?|Idle
        for file in seq:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines: self.vqas.append([line])
        print('Total files: %d | Total question: %.d' %(len(seq), len(self.vqas)))
        
        # Labels
        self.labels = ['kidney', 'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation',
                        'Tool_Manipulation', 'Cutting', 'Cauterization', 'Suction', 
                        'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing',
                        'left-top', 'right-top', 'left-bottom', 'right-bottom']
        
    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):        
        loc = self.vqas[idx][0].split('|')[0]

        # img
        img_loc = os.path.join('dataset','EndoVis-18-VQA',loc.split('/')[0], 'left_frames',loc.split('/')[1]+'.png')
        if self.transform: 
            img = Image.open(img_loc)
            img = self.transform(img)
        else: 
            img = self.image_processor(Image.open(img_loc), return_tensors="pt")
            
        # question and answer
        question = self.vqas[idx][0].split('|')[1]
        label = self.labels.index(str(self.vqas[idx][0].split('|')[2]))

        return os.path.join('dataset','EndoVis-18-VQA',loc.split('/')[0], 'left_frames',loc.split('/')[1]+'.png'), img, question, label


'''
Cholec80 classification dataloader GPT
'''
class Cholec80VQACSGPTClassification(Dataset):
    '''
    	seq: val_seq    = ['5','11','12','17','19','26','27','31']
    '''
    def __init__(self, seq, model_ver = None, transform=None):

        if model_ver == "efvlegpt2ViT": 
            self.transform = None
            self.image_processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        elif model_ver == "efvlegpt2Swin":
            self.transform = None
            self.image_processor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        elif transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                                    transforms.Resize((300,256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                    ])
        
        # files, question and answers #5/14400|what is the phase of image?|calot triangle dissection	
        self.vqas = []
        for file in seq:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines: self.vqas.append([line])
        print('Total files: %d | Total question: %.d' %(len(seq), len(self.vqas)))
        
        # labels
        self.labels = ['no', 'calot triangle dissection', 'yes', '1', '2', 'gallbladder dissection', 
                        'clipping cutting', 'gallbladder retraction', '0', 'cleaning coagulation', 
                        'gallbladder packaging', 'preparation', '3']
        
    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        
        # img
        loc = self.vqas[idx][0].split('|')[0]
        
        # img
        img_loc = os.path.join('dataset','Cholec80-VQA', 'cropped_image',loc.split('/')[0],loc.split('/')[1]+'.png')
        if self.transform: 
            img = Image.open(img_loc)
            img = self.transform(img)
        else: 
            img = self.image_processor(Image.open(img_loc), return_tensors="pt")
            
        # question and answer
        question = self.vqas[idx][0].split('|')[1]
        label = self.labels.index(str(self.vqas[idx][0].split('|')[2]))

        return os.path.join('dataset','Cholec80-VQA', 'cropped_image',loc.split('/')[0],loc.split('/')[1]+'.png'), img, question, label



'''
PSIAVA classification dataloader GPT
'''
class PSIAVAVQACSGPTClassification(Dataset):
    '''
             val_seq    =   [
                            "dataset/PSI-AVA-VQA/Val/C1_location.txt",
                            "dataset/PSI-AVA-VQA/Val/C2_action.txt",
                            "dataset/PSI-AVA-VQA/Val/C3_phase.txt",
                            "dataset/PSI-AVA-VQA/Val/C4_step.txt"
                            ]
    '''
    def __init__(self, seq, model_ver = None, transform=None):

        if model_ver == "efvlegpt2ViT": #model_ver == "gpt2ViT":
            self.transform = None
            self.image_processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        elif model_ver == "efvlegpt2Swin":# or model_ver == "gpt2Swin" or model_ver == "efvlegpt2Swingr":
            self.transform = None
            self.image_processor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        elif transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                                    transforms.Resize((300,256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                    ])
        
        # files, question and answers #CASE021/00001.jpg|Where is the Bipolar Forceps located ?|top left
        
        self.vqas = []
        for file in seq:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines: self.vqas.append([line])
                    
        print('Total classes: %d | Total question: %.d' %(len(seq), len(self.vqas)))
        
        # labels
        self.labels =  ["top left", "top right", "bottom left", "bottom right", #location
                "Complejo_venoso_dorsal", "Control_Pediculos", "Espacio_de_Retzius", "Fascia_Denonvilliers","Id_Cuello_Vesical", 
                "LPAD", "LPAI", "Rec_Cuello_Vesical", "Separacion_Prostata_Uretra", "Tiempo_muerto", "Vesículas_Seminales", #phase
                "Anudar", "Clip_Pediculos", "Corte", "Corte_Prostata", "Corte_Vejiga", "Diseccion_Denon", "Diseccion_Ganglios_Iliacos",
                "Diseccion_Ganglios_Obturadores", "Diseccion_Prevesical", "Diseccion_Prostata", "Diseccion_Seminal", "Empacar_Ganglios",  
                "Empacar_Prostata", "Halar_sutura", "Id_Vena_Arteria_Iliaca", "Pasar_Aguja_Cuello", "Pasar_Aguja_Cvdp", "Pasar_Aguja_Uretra", 
                "Succion","Sujetar_Prostata" ]#, #"Tiempo_muerto", #step                
        
    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        
        # img
        vqa_data = self.vqas[idx][0].split('|')
        
        # img
        img_loc = os.path.join('dataset/PSI-AVA-VQA/keyframes',vqa_data[0])
        if self.transform: 
            img = Image.open(img_loc)
            img = self.transform(img)
        else: 
            img = self.image_processor(Image.open(img_loc), return_tensors="pt")
            
        # question and answer
        question = vqa_data[1]
        label = self.labels.index(str(vqa_data[2]))

        return vqa_data[0], img, question, label