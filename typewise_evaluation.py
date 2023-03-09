import os
import argparse
import pandas as pd
from lib2to3.pytree import convert

from torch import nn
from torch import optim
import torch.utils.data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer, GPT2Tokenizer
from torch.utils.data  import DataLoader

from utils import *
from dataloaders.dataloaderClassification import *
from dataloaders.ClasswisedataloaderGPT2Classification import *
from models.VisualBertClassification import VisualBertClassification
from models.VisualBertResMLPClassification import VisualBertResMLPClassification
from models.EFGPT2Classification import EFVLEGPT2RS18Classification, EFVLEGPT2SwinClassification, EFVLEGPT2ViTClassification

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


'''
Seed randoms
'''
def seed_everything(seed=27):
    '''
    Set random seed for reproducible experiments
    Inputs: seed number 
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate(args, val_loader, model, criterion, epoch, tokenizer, device):
    
    model.eval()

    total_loss = 0.0    
    label_true = None
    label_pred = None
    label_score = None
    file_names = list()
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for i, (file_name, v_f, q, labels) in enumerate(val_loader,0):

            # prepare questions
            questions = []
            for question in q: questions.append(question)

            if args.model_ver == 'vb' or args.model_ver == 'vbrm':
                inputs = tokenizer(questions, return_tensors="pt", padding="max_length", max_length=args.question_len)
            
            elif args.model_ver == 'efvlegpt2rs18' or args.model_ver == "efvlegpt2Swin" or args.model_ver == 'efvlegpt2ViT':
                inputs = tokenizer(questions, padding="max_length",max_length=args.question_len, return_tensors="pt")

            # GPU / CPU
            # Visual features
            if args.model_ver == "efvlegpt2Swin" or args.model_ver == 'efvlegpt2ViT':         
                visual_features = v_f
                visual_features['pixel_values'] = torch.squeeze(visual_features['pixel_values'],1)
            else:
                visual_features = v_f.to(device)
            
            # label
            labels = labels.to(device)
            
            outputs = model(inputs, visual_features)

            loss = criterion(outputs,labels)

            total_loss += loss.item()
        
            scores, predicted = torch.max(F.softmax(outputs, dim=1).data, 1)    
            label_true = labels.data.cpu() if label_true == None else torch.cat((label_true, labels.data.cpu()), 0)
            label_pred = predicted.data.cpu() if label_pred == None else torch.cat((label_pred, predicted.data.cpu()), 0)
            label_score = scores.data.cpu() if label_score == None else torch.cat((label_score, scores.data.cpu()), 0)
            for f in file_name: file_names.append(f)
            
    acc = calc_acc(label_true, label_pred) 
    c_acc = 0.0
    # c_acc = calc_classwise_acc(label_true, label_pred)
    precision, recall, fscore = calc_precision_recall_fscore(label_true, label_pred)

    print('Test: epoch: %d loss: %.6f | Acc: %.6f | Precision: %.6f | Recall: %.6f | FScore: %.6f' %(epoch, total_loss, acc, precision, recall, fscore))

    if args.save_output:

        if args.dataset_type == 'c80':
            convert_arr = ['no', 'calot triangle dissection', 'yes', '1', '2', 'gallbladder dissection', 
                            'clipping cutting', 'gallbladder retraction', '0', 'cleaning coagulation', 
                            'gallbladder packaging', 'preparation', '3']
        elif args.dataset_type == 'm18':
            convert_arr = ['kidney', 'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation',
                            'Tool_Manipulation', 'Cutting', 'Cauterization', 'Suction', 
                            'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing',
                            'left-top', 'right-top', 'left-bottom', 'right-bottom']
        elif args.dataset_type == 'psi':
            convert_arr =  ["top left", "top right", "bottom left", "bottom right", #location
                            "Complejo_venoso_dorsal", "Control_Pediculos", "Espacio_de_Retzius", "Fascia_Denonvilliers","Id_Cuello_Vesical", 
                            "LPAD", "LPAI", "Rec_Cuello_Vesical", "Separacion_Prostata_Uretra", "Tiempo_muerto", "Vesículas_Seminales", #phase
                            "Anudar", "Clip_Pediculos", "Corte", "Corte_Prostata", "Corte_Vejiga", "Diseccion_Denon", "Diseccion_Ganglios_Iliacos",
                            "Diseccion_Ganglios_Obturadores", "Diseccion_Prevesical", "Diseccion_Prostata", "Diseccion_Seminal", "Empacar_Ganglios",  
                            "Empacar_Prostata", "Halar_sutura", "Id_Vena_Arteria_Iliaca", "Pasar_Aguja_Cuello", "Pasar_Aguja_Cvdp", "Pasar_Aguja_Uretra", 
                            "Succion","Sujetar_Prostata" ]

        df = pd.DataFrame(columns=["Img", "Ground Truth", "Prediction"])
        for i in range(len(label_true)):
            df = df.append({'Img': file_names[i], 'Ground Truth': convert_arr[label_true[i]], 'Prediction': convert_arr[label_pred[i]]}, ignore_index=True)
        
        df.to_csv(args.checkpoint.split('.')[0]+'_eval.csv')
    
    return (acc, c_acc, precision, recall, fscore)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VisualQuestionAnswerClassification')

    #EndoVis-18-VQA
    # CUDA_VISIBLE_DEVICES=0 python eval_classification.py --model_ver efvlegpt2rs18 --dataset_type m18  --checkpoint checkpoints/efvlegpt2rs18/m18_2/Best.pth.tar
    # CUDA_VISIBLE_DEVICES=0 python eval_classification.py --model_ver efvlegpt2Swin --dataset_type m18  --checkpoint checkpoints/efvlegpt2Swin/m18_2/Best.pth.tar
    # CUDA_VISIBLE_DEVICES=0 python eval_classification.py --model_ver efvlegpt2ViT --dataset_type m18  --checkpoint checkpoints/efvlegpt2ViT/m18_2/Best.pth.tar
    
    #Cholec80-VQA
    # CUDA_VISIBLE_DEVICES=0 python eval_classification.py --model_ver efvlegpt2rs18 --dataset_type c80  --checkpoint checkpoints/efvlegpt2rs18/c80/Best.pth.tar
    # CUDA_VISIBLE_DEVICES=0 python eval_classification.py --model_ver efvlegpt2Swin --dataset_type c80  --checkpoint checkpoints/efvlegpt2Swin/c80/Best.pth.tar
    # CUDA_VISIBLE_DEVICES=0 python eval_classification.py --model_ver efvlegpt2ViT --dataset_type c80  --checkpoint checkpoints/efvlegpt2ViT/c80/Best.pth.tar
    
    #PSI-AVA-VQA
    # CUDA_VISIBLE_DEVICES=0 python eval_classification.py --model_ver efvlegpt2rs18 --dataset_type psi  --checkpoint checkpoints/efvlegpt2rs18/psi/Best.pth.tar
    # CUDA_VISIBLE_DEVICES=0 python eval_classification.py --model_ver efvlegpt2Swin --dataset_type psi  --checkpoint checkpoints/efvlegpt2Swin/psi/Best.pth.tar
    # CUDA_VISIBLE_DEVICES=0 python eval_classification.py --model_ver efvlegpt2ViT --dataset_type psi  --checkpoint checkpoints/efvlegpt2ViT/psi/Best.pth.tar
    
    # Training parameters
    parser.add_argument('--batch_size',     type=int,   default=64,                                 help='batch_size')
    parser.add_argument('--workers',        type=int,   default=1,                                  help='for data-loading; right now, only 1 works with h5pys.')
    
    # existing checkpoint
    parser.add_argument('--checkpoint',     default=None,                                           help='path to checkpoint, None if none.')
    
    parser.add_argument('--dataset_type',   default= 'm18',                                         help='med_vqa/m18/c80/m18_vid/c80_vid')
    parser.add_argument('--class_file',     default= None,                                         help='')
    parser.add_argument('--dataset_cat',    default= '',                                            help='cat1/cat2/cat3')
    parser.add_argument('--model_ver',      default= None,                                          help='vb/vbrm/efvlegpt2rs18/efvlegpt2Swin/"')  #vrvb/gpt2rs18/gpt2ViT/gpt2Swin/biogpt2rs18/vilgpt2vqa/efgpt2rs18gr/efvlegpt2Swingr
    parser.add_argument('--tokenizer_ver',  default= 'gpt2v1',                                          help='btv2/btv3/gpt2v1')
    parser.add_argument('--patch_size',     default= 5,                                             help='1/2/3/4/5')
    parser.add_argument('--temporal_size',  default= 1,                                             help='1/2/3/4/5')
    parser.add_argument('--question_len',   default= 25,                                            help='25')
    parser.add_argument('--num_class',      default= 2,                                             help='25')
    
    parser.add_argument('--save_output',    default= False,                                         help='True/False')
    args = parser.parse_args()
    
    seed_everything()
    
    # GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
    print('device =', device)

    # dataset
    if args.dataset_type == 'm18':
        '''
        Train and test dataloader for EndoVis18
        '''
        # tokenizer
        tokenizer = None
        if args.tokenizer_ver == 'btv3': tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v3/bert-EndoVis-18-VQA/', do_lower_case=True)
        elif args.tokenizer_ver == 'gpt2v1':
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        
        # data location
        val_seq = [args.class_file]     #["dataset/EndoVis-18-VQA/Val/endovis_C1.txt", "dataset/EndoVis-18-VQA/Val/endovis_C2.txt"]
        # dataloader
        if args.model_ver == 'efvlegpt2rs18' or args.model_ver == "efvlegpt2Swin" or args.model_ver == 'efvlegpt2ViT':
            val_dataset = EndoVis18VQACSGPTClassification(val_seq, model_ver=args.model_ver)
            val_dataloader = DataLoader(dataset=val_dataset, batch_size= args.batch_size, shuffle=False, num_workers=8)

        # num_classes
        args.num_class = 18

    elif args.dataset_type == 'c80':
        '''
        Train and test for cholec dataset
        '''
        # tokenizer
        if args.tokenizer_ver == 'btv3': tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v3/bert-Cholec80-VQA/', do_lower_case=True)
        elif args.tokenizer_ver == 'gpt2v1':
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        
        # dataloader
        val_seq = [args.class_file]   #["dataset/Cholec80-VQA/Val/C1_Phase.txt"] # "dataset/Cholec80-VQA/Val/C2_Tool.txt" "dataset/Cholec80-VQA/Val/C3_Count.txt"

        # dataloader
        if args.model_ver == 'efvlegpt2rs18' or args.model_ver == "efvlegpt2Swin" or args.model_ver == 'efvlegpt2ViT':            
            val_dataset = Cholec80VQACSGPTClassification(val_seq, model_ver=args.model_ver)
            val_dataloader = DataLoader(dataset=val_dataset, batch_size= args.batch_size, shuffle=False, num_workers=8)

        # num_classes
        args.num_class = 13

    elif args.dataset_type == 'psi':
        '''
        Train and test for psi-ava-vqa dataset
        '''
        # tokenizer
        if args.tokenizer_ver == 'btv3': tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif args.tokenizer_ver == 'gpt2v1':
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        
        # dataloader
        val_seq    = [args.class_file] #["dataset/PSI-AVA-VQA/Val/C1_location.txt", "dataset/PSI-AVA-VQA/Val/C3_phase.txt", "dataset/PSI-AVA-VQA/Val/C4_step.txt"]

        # dataloader
        if args.model_ver == 'efvlegpt2rs18' or args.model_ver == "efvlegpt2Swin" or args.model_ver == 'efvlegpt2ViT':            
            val_dataset = PSIAVAVQACSGPTClassification(val_seq, model_ver=args.model_ver)
            val_dataloader = DataLoader(dataset=val_dataset, batch_size= args.batch_size, shuffle=False, num_workers=8)

        # num_classes
        args.num_class = 35 #155 #35
    
    # pre-trained model
    checkpoint = torch.load(args.checkpoint, map_location=str(device))
    model = checkpoint['model']
    epoch = checkpoint['epoch']
    # best_Acc = checkpoint['Acc']
    # print(model)

    # Move to GPU, if available
    model = model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('model params: ', pytorch_total_params)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)
 
    # validation
    test_acc, test_c_acc, test_precision, test_recall, test_fscore = validate(args, val_loader=val_dataloader, model = model, \
                                                                                criterion=criterion, epoch=epoch, tokenizer = tokenizer, device = device)