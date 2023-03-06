import os
import argparse
import pandas as pd
from lib2to3.pytree import convert

from torch import nn
from torch import optim
import torch.utils.data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer
from torch.utils.data  import DataLoader

from utils import *
from dataloaders.dataloaderClassification import *
from models.VisualBertClassification import VisualBertClassification
from models.VisualBertResMLPClassification import VisualBertResMLPClassification

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


def validate(args, val_loader, model, criterion, epoch, tokenizer, device, save_output = False):
    
    model.eval()

    total_loss = 0.0    
    label_true = None
    label_pred = None
    label_score = None
    file_names = list()
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for i, (file_name, visual_features, q, labels) in enumerate(val_loader,0):
            # prepare questions
            questions = []
            for question in q: questions.append(question)
            inputs = tokenizer(questions, return_tensors="pt", padding="max_length", max_length=args.question_len)

            # GPU / CPU
            visual_features = visual_features.to(device)
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

    if save_output:
        '''
            Saving predictions
        '''
        if os.path.exists(args.checkpoint_dir + 'text_files') == False:
            os.mkdir(args.checkpoint_dir + 'text_files' ) 
        file1 = open(args.checkpoint_dir + 'text_files/labels.txt', 'w')
        file1.write(str(label_true))
        file1.close()

        file1 = open(args.checkpoint_dir + 'text_files/predictions.txt', 'w')
        file1.write(str(label_pred))
        file1.close()

        if args.dataset_type == 'med_vqa':
            if args.dataset_cat == 'cat1': 
                convert_arr = ['cta - ct angiography', 'no', 'us - ultrasound', 'xr - plain film', 'noncontrast', 'yes', 't2', 'ct w/contrast (iv)', 'mr - flair', 'mammograph', 'ct with iv contrast', 
                            'gi and iv', 't1', 'mr - t2 weighted', 'mr - t1w w/gadolinium', 'contrast', 'iv', 'an - angiogram', 'mra - mr angiography/venography', 'nm - nuclear medicine', 'mr - dwi diffusion weighted', 
                            'ct - gi & iv contrast', 'ct noncontrast', 'mr - other pulse seq.', 'ct with gi and iv contrast', 'flair', 'mr - t1w w/gd (fat suppressed)', 'ugi - upper gi', 'mr - adc map (app diff coeff)', 
                            'bas - barium swallow', 'pet - positron emission', 'mr - pdw proton density', 'mr - t1w - noncontrast', 'be - barium enema', 'us-d - doppler ultrasound', 'mr - stir', 'mr - flair w/gd', 
                            'ct with gi contrast', 'venogram', 'mr t2* gradient,gre,mpgr,swan,swi', 'mr - fiesta', 'ct - myelogram', 'gi', 'sbft - small bowel', 'pet-ct fusion']
            elif args.dataset_cat == 'cat2':
                convert_arr = ['axial', 'longitudinal', 'coronal', 'lateral', 'ap', 'sagittal', 'mammo - mlo', 'pa', 'mammo - cc', 'transverse', 'mammo - mag cc', 'frontal', 'oblique', '3d reconstruction', 'decubitus', 'mammo - xcc']
            else:
                convert_arr = ['lung, mediastinum, pleura', 'skull and contents', 'genitourinary', 'spine and contents', 'musculoskeletal', 'heart and great vessels', 'vascular and lymphatic', 'gastrointestinal', 'face, sinuses, and neck', 'breast']
        elif args.dataset_type == 'c80':
            convert_arr = ['no', 'calot triangle dissection', 'yes', '1', '2', 'gallbladder dissection', 
                            'clipping cutting', 'gallbladder retraction', '0', 'cleaning coagulation', 
                            'gallbladder packaging', 'preparation', '3']
        elif args.dataset_type == 'm18':
            convert_arr = ['kidney', 'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation',
                            'Tool_Manipulation', 'Cutting', 'Cauterization', 'Suction', 
                            'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing',
                            'left-top', 'right-top', 'left-bottom', 'right-bottom']

        df = pd.DataFrame(columns=["Img", "Ground Truth", "Prediction"])
        for i in range(len(label_true)):
            df = df.append({'Img': file_names[i], 'Ground Truth': convert_arr[label_true[i]], 'Prediction': convert_arr[label_pred[i]]}, ignore_index=True)
        
        df.to_csv(args.checkpoint_dir + args.checkpoint_dir.split('/')[1] + '_' + args.checkpoint_dir.split('/')[2] + '_eval.csv')
    
    return (acc, c_acc, precision, recall, fscore)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VisualQuestionAnswerClassification')
    
    # CUDA_VISIBLE_DEVICES=0 python eval_classification.py --transformer_ver vb --dataset_type med_vqa --dataset_cat cat1  --checkpoint_dir checkpoints/clf_vb_5/med_vqa_c1/Best.pth.tar
    # CUDA_VISIBLE_DEVICES=0 python eval_classification.py --transformer_ver vb --dataset_type med_vqa --dataset_cat cat2  --checkpoint_dir checkpoints/clf_vb_5/med_vqa_c2/Best.pth.tar
    # CUDA_VISIBLE_DEVICES=0 python eval_classification.py --transformer_ver vb --dataset_type med_vqa --dataset_cat cat3  --checkpoint_dir checkpoints/clf_vb_5/med_vqa_c3/Best.pth.tar
    # CUDA_VISIBLE_DEVICES=0 python eval_classification.py --transformer_ver vb --dataset_type m18  --checkpoint_dir checkpoints/clf_vb_5/m18/Best.pth.tar
    # CUDA_VISIBLE_DEVICES=0 python eval_classification.py --transformer_ver vb --dataset_type c80  --checkpoint_dir checkpoints/clf_vb_5/c80/Best.pth.tar

    # CUDA_VISIBLE_DEVICES=0 python eval_classification.py --transformer_ver vbrm --dataset_type med_vqa --dataset_cat cat1  --checkpoint_dir checkpoints/clf_vbrm_5/med_vqa_c1/Best.pth.tar
    # CUDA_VISIBLE_DEVICES=0 python eval_classification.py --transformer_ver vbrm --dataset_type med_vqa --dataset_cat cat2  --checkpoint_dir checkpoints/clf_vbrm_5/med_vqa_c2/Best.pth.tar
    # CUDA_VISIBLE_DEVICES=0 python eval_classification.py --transformer_ver vbrm --dataset_type med_vqa --dataset_cat cat3  --checkpoint_dir checkpoints/clf_vbrm_5/med_vqa_c3/Best.pth.tar
    # CUDA_VISIBLE_DEVICES=0 python eval_classification.py --transformer_ver vbrm --dataset_type m18  --checkpoint_dir checkpoints/clf_vbrm_5/M18/Best.pth.tar
    # CUDA_VISIBLE_DEVICES=0 python eval_classification.py --transformer_ver vbrm --dataset_type c80  --checkpoint_dir checkpoints/clf_vbrm_5/c80/Best.pth.tar
    # Training parameters
    parser.add_argument('--batch_size',     type=int,   default=64,                                 help='batch_size')
    parser.add_argument('--workers',        type=int,   default=1,                                  help='for data-loading; right now, only 1 works with h5pys.')
    
    # existing checkpoint
    parser.add_argument('--checkpoint',     default=None,                                           help='path to checkpoint, None if none.')
    
    parser.add_argument('--dataset_type',   default= 'm18',                                         help='med_vqa/m18/c80/m18_vid/c80_vid')
    parser.add_argument('--dataset_cat',    default= '',                                            help='cat1/cat2/cat3')
    parser.add_argument('--transformer_ver',default= 'vb',                                          help='vb/vbrm')
    parser.add_argument('--tokenizer_ver',  default= 'v2',                                          help='v2/v3')
    parser.add_argument('--patch_size',     default= 5,                                             help='1/2/3/4/5')
    parser.add_argument('--temporal_size',  default= 1,                                             help='1/2/3/4/5')
    parser.add_argument('--question_len',   default= 25,                                            help='25')
    parser.add_argument('--num_class',      default= 2,                                             help='25')
    args = parser.parse_args()

    # load checkpoint, these parameters can't be modified
    # final_args = {"emb_dim": args.emb_dim, "n_heads": args.n_heads, "dropout": args.dropout, "encoder_layers": args.encoder_layers}
    
    seed_everything()
    
    # GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
    print('device =', device)

    # dataset
    if args.dataset_type == 'med_vqa':
        '''
        Test dataloader for MED_VQA
        '''
        # tokenizer
        tokenizer = None
        if args.tokenizer_ver == 'v2': tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v2/bert-medvqa/')
        elif args.tokenizer_ver == 'v3': tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v3/bert-medvqa/', do_lower_case=True)
        
        # data location
        val_folder = 'dataset/VQA-Med/ImageClef-2019-VQA-Med-Validation/'
        val_img_folder = 'Val_images/'

        # dataloader
        val_dataset = MedVQAClassification(val_folder, val_img_folder, args.dataset_cat, patch_size = args.patch_size, validation=True)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size= args.batch_size, shuffle=False)

        # num_classes
        if args.dataset_cat == 'cat1': args.num_class = 45
        elif args.dataset_cat == 'cat2': args.num_class = 16
        elif args.dataset_cat == 'cat3': args.num_class = 10

    elif args.dataset_type == 'm18':
        '''
        Test dataloader for EndoVis18
        '''
        # tokenizer
        tokenizer = None
        if args.tokenizer_ver == 'v2': tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v2/bert-EndoVis-18-VQA/')
        elif args.tokenizer_ver == 'v3': tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v3/bert-EndoVis-18-VQA/', do_lower_case=True)
        
        # data location
        val_seq = [1, 5, 16]
        folder_head = 'dataset/EndoVis-18-VQA/seq_'
        folder_tail = '/vqa/Classification/*.txt'
        
        # dataloader
        val_dataset = EndoVis18VQAClassification(val_seq, folder_head, folder_tail, patch_size = args.patch_size)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size= args.batch_size, shuffle=False)

        # num_classes
        args.num_class = 18

    elif args.dataset_type == 'm18_vid':
        '''
        Test dataloader for EndoVis18 temporal
        '''
        # tokenizer
        tokenizer = None
        if args.tokenizer_ver == 'v2': tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v2/bert-EndoVis-18-VQA/')
        elif args.tokenizer_ver == 'v3': tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v3/bert-EndoVis-18-VQA/', do_lower_case=True)
        
        # data location
        val_seq = [1, 5, 16]
        folder_head = 'dataset/EndoVis-18-VQA/seq_'
        folder_tail = '/vqa/Classification/*.txt'
        
        # dataloader
        val_dataset = EndoVis18VidVQAClassification(val_seq, folder_head, folder_tail, patch_size = args.patch_size, temporal_size=args.temporal_size)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size= args.batch_size, shuffle=False)

        # num_classes
        args.num_class = 18

    elif args.dataset_type == 'c80':
        '''
        Test for cholec dataset
        '''
        # tokenizer
        if args.tokenizer_ver == 'v2': tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v2/bert-Cholec80-VQA/')
        elif args.tokenizer_ver == 'v3': tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v3/bert-Cholec80-VQA/', do_lower_case=True)
        
        # dataloader
        val_seq = [5, 11, 12, 17, 19, 26, 27, 31]
        folder_head = 'dataset/Cholec80-VQA/Classification/'
        folder_tail = '/*.txt'

        # dataloader
        val_dataset = Cholec80VQAClassification(val_seq, folder_head, folder_tail, patch_size = args.patch_size)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size= args.batch_size, shuffle=False)

        # num_classes
        args.num_class = 13

    elif args.dataset_type == 'c80_vid':
        '''
        Test dataloader for c80 temporal
        '''
        # tokenizer
        tokenizer = None
        if args.tokenizer_ver == 'v2': tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v2/bert-Cholec80-VQA/')
        elif args.tokenizer_ver == 'v3': tokenizer = BertTokenizer.from_pretrained('./dataset/bertvocab/v3/bert-Cholec80-VQA/', do_lower_case=True)
        
        # data location
        val_seq = [5, 11, 12, 17, 19, 26, 27, 31]
        folder_head = 'dataset/Cholec80-VQA/Classification/'
        folder_tail = '/*.txt'
        
        # dataloader
        val_dataset = Cholec80VQAVidClassification(val_seq, folder_head, folder_tail, patch_size = args.patch_size, temporal_size=args.temporal_size)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size= args.batch_size, shuffle=False)

        # num_classes
        args.num_class = 13
    
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
    test_acc, test_c_acc, test_precision, test_recall, test_fscore = validate(args, val_loader=val_dataloader, model = model, criterion=criterion, epoch=epoch, tokenizer = tokenizer, device = device)