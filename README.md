<div align="center">

<samp>

<h2> SurgicalGPT: End-to-End Language-Vision GPT for Visual Question Answering in Surgery </h1>

<h4> Lalithkumar Seenivasan*, Mobarakol Islam*, Gokul Kannan and Hongliang Ren </h3>

</samp>   

---
| **[[```arXiv```](<https://arxiv.org/abs/2304.09974>)]** | **[[```Paper```](<https://link.springer.com/chapter/10.1007/978-3-031-43996-4_27>)]** | 
|:-------------------:|:-------------------:|:-------------------:|
    
The International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2023
---


</div>     
---
## Dataset

1. EndoVis-18-VQA  **[[`EndoVis-18-VQA Q&A pair annotation`](https://drive.google.com/drive/folders/1hu_yK27Xz2_lvjjZ97-WF2MK_JO14MWI?usp=sharing)]**
2. Cholec80-VQA **[[`Cholec80-VQA Q&A pair annotation`](https://drive.google.com/drive/folders/1yrg0cR2haNTRBHg-Fho0o7TFVSjN64ym?usp=sharing)]**
3. PSI-AVA-VQA **[[`PSI-VQA Q&A pair annotation`](https://drive.google.com/drive/folders/17OkeLxOCep3f99nDkdU_Y9EkPwHgT814?usp=sharing)]**

---
## Training Example
1. LV-GPT (Swin) on EndoVis18-VQA with early word, no visualbert vision embedding and zero pose embedding
    - model_subver:
        - 'v0' : Vision tokens are further embedded using VisualBert vision embedding
        - "v1' :Vision tokens are directly used as vision embedding
    - dataset_type: 
        - 'm18' : EndoVis18-VQA
        - 'c80' : Cholec80-VQA
        - 'psi' : PSI-AVA-VQA
    - vis_pos_emb:
        - None
        - 'pos' : vision tokens pos = 0, 1, 2, 3, ...., n.
        ='zeroes' = vision tokens pos = 0
```bash
python train.py --lr=0.00001 --checkpoint_dir='checkpoints/efvlegpt2Swin/m18_v1_z_qf_' --dataset_type='m18' --tokenizer_ver='gpt2v1' --model_ver='efvlegpt2Swin' --model_subver='v1' --vis_pos_emb='zeroes'
```
---
## Evaluation
Sample command
```
python Evaluation.py --model_ver efvlegpt2Swin --dataset_type m18  --checkpoint checkpoints/efvlegpt2Swin/m18_v1_z_qf_Best.pth.tar
```

## Sub-Type Evaluation
Sample command
```
python typewise_evaluation.py --model_ver efvlegpt2Swin --dataset_type m18  --checkpoint checkpoints/efvlegpt2Swin/m18_2/m18_v1_z_qf_Best.pth.tar --class_file "dataset/EndoVis-18-VQA/Val/endovis_C1.txt"
```
