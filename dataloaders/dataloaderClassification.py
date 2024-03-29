import os
import glob
import h5py
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import ViTFeatureExtractor, AutoFeatureExtractor


'''
VQA-MED 19 Category 1 / Category 2 / Category 3 dataloader for VB Transformer
'''
class MedVQAVBClassification(Dataset):
    '''
        VQA-MED 19 dataloader
        datafolder = data folder location
        img_folder = image folder location
        cat        = category (1/2/3)
        patch_size = patch size of image, which also determines token size (1/2/3/4/5)
        validation = if validation, load val.txt, else load train.txt (True / False)
    '''
    def __init__(self, datafolder, imgfolder, cat, patch_size = 4, validation = False):        
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
        self.patch_size = patch_size

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
        visual_feature_loc = self.data_folder_loc + 'vqa/img_features/'+(str(self.patch_size)+'x'+str(self.patch_size))+ '/'+ self.vqas[idx][0].split('|')[0]+'.hdf5'
        frame_data = h5py.File(visual_feature_loc, 'r')    
        visual_features = torch.from_numpy(frame_data['visual_features'][:])

        # question and answer
        question = self.vqas[idx][0].split('|')[1]
        answer = self.labels.index(str(self.vqas[idx][0].split('|')[2]))

        return self.vqas[idx][0].split('|')[0], visual_features, question, answer


'''
EndoVis18 classification dataloader for VB transfomers
'''
class EndoVis18VQAVBClassification(Dataset):
    '''
    	seq: train_seq  = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
    	     val_seq    = [1, 5, 16]
    	folder_head     = 'dataset/EndoVis-18-VQA/seq_'
    	folder_tail     = '/vqa/Classification/*.txt'
    	patch_size      = 1/2/3/4/5
    '''
    def __init__(self, seq, folder_head, folder_tail, patch_size=4):
             
        self.patch_size = patch_size
        
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
        
        # img
        loc = self.vqas[idx][0].split('/')
        visual_feature_loc = os.path.join(loc[0],loc[1],loc[2], 'vqa/img_features', (str(self.patch_size)+'x'+str(self.patch_size)),loc[-1].split('_')[0]+'.hdf5')
        frame_data = h5py.File(visual_feature_loc, 'r')    
        visual_features = torch.from_numpy(frame_data['visual_features'][:])
            
        # question and answer
        question = self.vqas[idx][1].split('|')[0]
        label = self.labels.index(str(self.vqas[idx][1].split('|')[1]))

        return loc[-1].split('_')[0], visual_features, question, label


'''
EndoVis18 Video classification dataloader
'''
class EndoVis18VidVQAVBClassification(Dataset):
    '''
    	seq: train_seq  = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
    	     val_seq    = [1, 5, 16]
    	folder_head     = 'dataset/EndoVis-18-VQA/seq_'
    	folder_tail     = '/vqa/Classification/*.txt'
    	patch_size      = 1/2/3/4/5
        temporal_size   = 2/3/4/5
    '''
    def __init__(self, seq, folder_head, folder_tail, patch_size=4, temporal_size = 3):
             
        self.patch_size = patch_size
        self.temporal_size = temporal_size
        
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
        
        # img
        loc = self.vqas[idx][0].split('/')
        visual_feature_loc = os.path.join(loc[0],loc[1],loc[2], ('vqa/vid_features'+str(self.temporal_size)), (str(self.patch_size)+'x'+str(self.patch_size)),loc[-1].split('_')[0]+'.hdf5')
        frame_data = h5py.File(visual_feature_loc, 'r')    
        visual_features = torch.from_numpy(frame_data['visual_features'][:])
            
        # question and answer
        question = self.vqas[idx][1].split('|')[0]
        label = self.labels.index(str(self.vqas[idx][1].split('|')[1]))

        return loc[-1].split('_')[0], visual_features, question, label


'''
Cholec80 classification dataloader
'''
class Cholec80VQAVBClassification(Dataset):
    '''
    	seq: train_seq  = ['1','2','3','4','6','7','8','9','10','13','14','15','16','18','20',
                          '21','22','23','24','25','28','29','30','32','33','34','35','36','37','38','39','40']
             val_seq    = ['5','11','12','17','19','26','27','31']
    	folder_head     = 'dataset/Cholec80-VQA/Classification/'
	    folder_tail     = '/*.txt'
	    patch_size      = 1/2/3/4/5
    '''
    def __init__(self, seq, folder_head, folder_tail, patch_size=4):

        self.patch_size = patch_size
        
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
        visual_feature_loc =  os.path.join(loc[0],loc[1], 'cropped_image',loc[3],'vqa/img_features',(str(self.patch_size)+'x'+str(self.patch_size)) ,loc[-1].split('_')[0]+'.hdf5')
        frame_data = h5py.File(visual_feature_loc, 'r')    
        visual_features = torch.from_numpy(frame_data['visual_features'][:])
        
        # question and answer
        question = self.vqas[idx][1].split('|')[0]
        label = self.labels.index(str(self.vqas[idx][1].split('|')[1]))

        return loc[-1].split('_')[0], visual_features, question, label
        

'''
Cholec80 Video classification dataloader
'''
class Cholec80VidVQAVBClassification(Dataset):
    '''
    	seq: train_seq  = ['1','2','3','4','6','7','8','9','10','13','14','15','16','18','20',
                          '21','22','23','24','25','28','29','30','32','33','34','35','36','37','38','39','40']
             val_seq    = ['5','11','12','17','19','26','27','31']
    	folder_head     = 'dataset/Cholec80-VQA/Classification/'
	    folder_tail     = '/*.txt'
	    patch_size      = 1/2/3/4/5
        temporal_size   = 1/2/3/4/5
    '''
    def __init__(self, seq, folder_head, folder_tail, patch_size=4, temporal_size = 3):

        self.patch_size = patch_size
        self.temporal_size = temporal_size

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
        visual_feature_loc =  os.path.join(loc[0],loc[1], 'cropped_image',loc[3],('vqa/vid2_features'+str(self.temporal_size)),(str(self.patch_size)+'x'+str(self.patch_size)) ,loc[-1].split('_')[0]+'.hdf5')
        frame_data = h5py.File(visual_feature_loc, 'r')    
        visual_features = torch.from_numpy(frame_data['visual_features'][:])
        
        # question and answer
        question = self.vqas[idx][1].split('|')[0]
        label = self.labels.index(str(self.vqas[idx][1].split('|')[1]))

        return loc[-1].split('_')[0], visual_features, question, label


'''
PSI-AVA classification dataloader for VB transfomers
'''
class PSIAVAVQAVBClassification(Dataset):
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

    	folder_head     = 'dataset/EndoVis-18-VQA/seq_'
    	folder_tail     = '/vqa/Classification/*.txt'
    	patch_size      = 1/2/3/4/5
    '''
    def __init__(self, seq, patch_size=5):
             
        self.patch_size = patch_size
        
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
                "Succion","Sujetar_Prostata" ]#, #"Tiempo_muerto", #step


    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        
        vqa_data = self.vqas[idx][1].split('|')
        
        # img
        img_loc = os.path.join('dataset/PSI-AVA-VQA/keyframes', (str(self.patch_size)+'x'+str(self.patch_size)), vqa_data[0].split('.')[0]+'.hdf5')
        frame_data = h5py.File(img_loc, 'r')    
        visual_features = torch.from_numpy(frame_data['visual_features'][:])
            
        # question and answer
        question = vqa_data[1]
        label = self.labels.index(str(vqa_data[2]))

        return vqa_data[0], visual_features, question, label