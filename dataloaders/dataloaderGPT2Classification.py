import os
import glob
import h5py
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import ViTFeatureExtractor, AutoFeatureExtractor

'''
VQA-MED 19 Category 1 / Category 2 / Category 3 dataloader for GPT
'''
class MedVQAGPTClassification(Dataset):
    '''
        VQA-MED 19 dataloader for GPT
        datafolder = data folder location
        img_folder = image folder location
        cat        = category (1/2/3)
        validation = if validation, load val.txt, else load train.txt (True / False)
    '''
    def __init__(self, datafolder, imgfolder, cat, model_ver = None, transform=None, validation = False):
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

        self.data_folder_loc = datafolder
        self.img_folder_loc = imgfolder
        if cat == 'cat1':
            if validation:
                self.file_name = 'QAPairsByCategory/C1_Modality_val.txt'
            else:
                self.file_name = 'QAPairsByCategory/C1_Modality_train.txt'
            self.labels = ['cta - ct angiography', 'no', 'us - ultrasound', 'xr - plain film', 'noncontrast', 'yes', 't2', 'ct w/contrast (iv)', 'mr - flair', 'mammograph', 'ct with iv contrast', 
                           'gi and iv', 't1', 'mr - t2 weighted', 'mr - t1w w/gadolinium', 'contrast', 'iv', 'an - angiogram', 'mra - mr angiography/venography', 'nm - nuclear medicine', 'mr - dwi diffusion weighted', 
                           'ct - gi & iv contrast', 'ct noncontrast', 'mr - other pulse seq.', 'ct with gi and iv contrast', 'flair', 'mr - t1w w/gd (fat suppressed)', 'ugi - upper gi', 'mr - adc map (app diff coeff)', 
                           'bas - barium swallow', 'pet - positron emission', 'mr - pdw proton density', 'mr - t1w - noncontrast', 'be - barium enema', 'us-d - doppler ultrasound', 'mr - stir', 'mr - flair w/gd', 
                           'ct with gi contrast', 'venogram', 'mr t2* gradient,gre,mpgr,swan,swi', 'mr - fiesta', 'ct - myelogram', 'gi', 'sbft - small bowel', 'pet-ct fusion']
        elif cat == 'cat2':
            if validation:
                self.file_name = 'QAPairsByCategory/C2_Plane_val.txt'
            else:
                self.file_name = 'QAPairsByCategory/C2_Plane_train.txt'
            self.labels = ['axial', 'longitudinal', 'coronal', 'lateral', 'ap', 'sagittal', 'mammo - mlo', 'pa', 'mammo - cc', 'transverse', 'mammo - mag cc', 'frontal', 'oblique', '3d reconstruction', 'decubitus', 'mammo - xcc']
        elif cat == 'cat3':
            if validation:
                self.file_name = 'QAPairsByCategory/C3_Organ_val.txt'
            else:
                self.file_name = 'QAPairsByCategory/C3_Organ_train.txt'
            self.labels = ['lung, mediastinum, pleura', 'skull and contents', 'genitourinary', 'spine and contents', 'musculoskeletal', 'heart and great vessels', 'vascular and lymphatic', 'gastrointestinal', 'face, sinuses, and neck', 'breast']

        self.vqas = []
        file_data = open((self.data_folder_loc+self.file_name), "r")
        lines = [line.strip("\n") for line in file_data if line != "\n"]
        file_data.close()
        for line in lines: self.vqas.append([line])
        
        print('Total question: %.d' %len(lines))
              
    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        
        # img
        img_loc = os.path.join(self.data_folder_loc,self.img_folder_loc,self.vqas[idx][0].split('|')[0]+'.jpg')
        if self.transform: 
            img = Image.open(img_loc)
            img = self.transform(img)
        else: 
            img = self.image_processor(Image.open(img_loc), return_tensors="pt")

        # question and answer
        question = self.vqas[idx][0].split('|')[1]
        label = self.labels.index(str(self.vqas[idx][0].split('|')[2]))
        
        return self.vqas[idx][0].split('|')[0], img, question, label


'''
EndoVis18 classification dataloader for GPT2 + ResNet18
'''
class EndoVis18VQAGPTClassification(Dataset):
    '''
    	seq: train_seq  = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
    	     val_seq    = [1, 5, 16]
    	folder_head     = 'dataset/EndoVis-18-VQA/seq_'
    	folder_tail     = '/vqa/Classification/*.txt'
    '''
    def __init__(self, seq, folder_head, folder_tail, model_ver = None, transform=None):
        
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
        

        # files, question and answers
        filenames = []
        for curr_seq in seq: filenames = filenames + glob.glob(folder_head + str(curr_seq) + folder_tail)
        self.vqas = []
        for file in filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines: self.vqas.append([file, line])
        print('Total files: %d | Total question: %.d' %(len(filenames), len(self.vqas)))
        
        # Labels
        self.labels = ['kidney', 'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation',
                        'Tool_Manipulation', 'Cutting', 'Cauterization', 'Suction', 
                        'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing',
                        'left-top', 'right-top', 'left-bottom', 'right-bottom']
        
    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):        
        loc = self.vqas[idx][0].split('/')

        # img
        img_loc = os.path.join(loc[0],loc[1],loc[2], 'left_frames',loc[-1].split('_')[0]+'.png')
        if self.transform: 
            img = Image.open(img_loc)
            img = self.transform(img)
        else: 
            img = self.image_processor(Image.open(img_loc), return_tensors="pt")
            
        # question and answer
        question = self.vqas[idx][1].split('|')[0]
        label = self.labels.index(str(self.vqas[idx][1].split('|')[1]))

        return os.path.join(loc[0],loc[1],loc[2], 'left_frames',loc[-1].split('_')[0]+'.png'), img, question, label


'''
Cholec80 classification dataloader GPT
'''
class Cholec80VQAGPTClassification(Dataset):
    '''
    	seq: train_seq  = ['1','2','3','4','6','7','8','9','10','13','14','15','16','18','20',
                          '21','22','23','24','25','28','29','30','32','33','34','35','36','37','38','39','40']
             val_seq    = ['5','11','12','17','19','26','27','31']
    	folder_head     = 'dataset/Cholec80-VQA/Classification/'
	    folder_tail     = '/*.txt'
    '''
    def __init__(self, seq, folder_head, folder_tail, model_ver = None, transform=None):

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
        
        # files, question and answers
        filenames = []
        for curr_seq in seq: filenames = filenames + glob.glob(folder_head + str(curr_seq) + folder_tail)
        new_filenames = []
        for filename in filenames:
            frame_num = int(filename.split('/')[-1].split('.')[0].split('_')[0])
            if frame_num % 100 == 0: new_filenames.append(filename)
    		
        self.vqas = []
        for file in new_filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines: self.vqas.append([file, line])
        print('Total files: %d | Total question: %.d' %(len(filenames), len(self.vqas)))
        
        # labels
        self.labels = ['no', 'calot triangle dissection', 'yes', '1', '2', 'gallbladder dissection', 
                        'clipping cutting', 'gallbladder retraction', '0', 'cleaning coagulation', 
                        'gallbladder packaging', 'preparation', '3']
        
    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        
        # img
        loc = self.vqas[idx][0].split('/')
        
        # img
        img_loc = os.path.join(loc[0],loc[1], 'cropped_image',loc[3],loc[-1].split('_')[0]+'.png')
        if self.transform: 
            img = Image.open(img_loc)
            img = self.transform(img)
        else: 
            img = self.image_processor(Image.open(img_loc), return_tensors="pt")
            
        # question and answer
        question = self.vqas[idx][1].split('|')[0]
        label = self.labels.index(str(self.vqas[idx][1].split('|')[1]))

        return os.path.join(loc[0],loc[1], 'cropped_image',loc[3],loc[-1].split('_')[0]+'.png'), img, question, label



'''
PSIAVA classification dataloader GPT
'''
class PSIAVAVQAGPTClassification(Dataset):
    '''
    	seq: train_seq  =   [
                            "dataset/PSI-AVA-VQA/Train/C1_location.txt", 
                            "dataset/PSI-AVA-VQA/Train/C2_action.txt", 
                            "dataset/PSI-AVA-VQA/Train/C3_phase.txt", 
                            "dataset/PSI-AVA-VQA/Train/C4_step.txt"
                            ]
             val_seq    =   [
                            "dataset/PSI-AVA-VQA/Val/C1_location.txt",
                            "dataset/PSI-AVA-VQA/Val/C2_action.txt",
                            "dataset/PSI-AVA-VQA/Val/C3_phase.txt",
                            "dataset/PSI-AVA-VQA/Val/C4_step.txt"
                            ]
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
        
        # files, question and answers
        filenames = []
        for curr_seq in seq: 
            filenames = filenames + glob.glob((curr_seq))
        
        self.vqas = []
        for file in filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines: self.vqas.append([file, line])
                    
        print('Total classes: %d | Total question: %.d' %(len(filenames), len(self.vqas)))
        
        # labels
        self.labels =  ["top left", "top right", "bottom left", "bottom right", #location
                "Complejo_venoso_dorsal", "Control_Pediculos", "Espacio_de_Retzius", "Fascia_Denonvilliers","Id_Cuello_Vesical", 
                "LPAD", "LPAI", "Rec_Cuello_Vesical", "Separacion_Prostata_Uretra", "Tiempo_muerto", "Vesículas_Seminales", #phase
                "Anudar", "Clip_Pediculos", "Corte", "Corte_Prostata", "Corte_Vejiga", "Diseccion_Denon", "Diseccion_Ganglios_Iliacos",
                "Diseccion_Ganglios_Obturadores", "Diseccion_Prevesical", "Diseccion_Prostata", "Diseccion_Seminal", "Empacar_Ganglios",  
                "Empacar_Prostata", "Halar_sutura", "Id_Vena_Arteria_Iliaca", "Pasar_Aguja_Cuello", "Pasar_Aguja_Cvdp", "Pasar_Aguja_Uretra", 
                "Succion","Sujetar_Prostata" ]#
                
        
    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        
        # img
        vqa_data = self.vqas[idx][1].split('|')
        
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