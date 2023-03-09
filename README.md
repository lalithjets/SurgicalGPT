<div align="center">

<samp>

<h2> SurgicalGPT: End-to-End Language-Vision GPT for Visual Question Answering in Surgery </h1>

<h4> anonymous </h3>


</div>     

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
