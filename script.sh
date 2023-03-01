#--------------------------------------------------------------------------------------------------------------
# echo "GPT2ViT MedVQA C1"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.000005 \
#                                         --checkpoint_dir='checkpoints/clf_gpt2vit/med_vqa_c1/' \
#                                         --dataset_type='med_vqa' --dataset_cat='cat1' \
#                                         --model_ver='gpt2ViT'

# echo "GPT2ViT MedVQA C2"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.000005 \
#                                         --checkpoint_dir='checkpoints/clf_gpt2vit/med_vqa_c2/' \
#                                         --dataset_type='med_vqa' --dataset_cat='cat2' \
#                                         --model_ver='gpt2ViT'

# echo "GPT2ViT MedVQA C3"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.000005 \
#                                         --checkpoint_dir='checkpoints/clf_gpt2vit/med_vqa_c3/' \
#                                         --dataset_type='med_vqa' --dataset_cat='cat3' \
#                                         --model_ver='gpt2ViT'

# echo "GPT2SwinD MedVQA C1"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.000005 \
#                                         --checkpoint_dir='checkpoints/clf_gpt2swin/med_vqa_c1/m' \
#                                         --dataset_type='med_vqa' --dataset_cat='cat1' \
#                                         --model_ver='gpt2Swin'

# echo "GPT2SwinD MedVQA C2"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.000005 \
#                                         --checkpoint_dir='checkpoints/clf_gpt2swin/med_vqa_c2/m' \
#                                         --dataset_type='med_vqa' --dataset_cat='cat2' \
#                                         --model_ver='gpt2Swin'

# echo "GPT2SwinD MedVQA C3"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.000005 \
#                                         --checkpoint_dir='checkpoints/clf_gpt2swin/med_vqa_c3/m' \
#                                         --dataset_type='med_vqa' --dataset_cat='cat3' \
#                                         --model_ver='gpt2Swin'


# echo "GPT2ViTM MedVQA C1"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.000005 \
#                                         --checkpoint_dir='checkpoints/clf_gpt2vit/med_vqa_c1/m' \
#                                         --dataset_type='med_vqa' --dataset_cat='cat1' \
#                                         --model_ver='gpt2ViT'

# echo "GPT2ViTM MedVQA C2"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.000005 \
#                                         --checkpoint_dir='checkpoints/clf_gpt2vit/med_vqa_c2/m' \
#                                         --dataset_type='med_vqa' --dataset_cat='cat2' \
#                                         --model_ver='gpt2ViT'

# echo "GPT2ViTM MedVQA C3"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.000005 \
#                                         --checkpoint_dir='checkpoints/clf_gpt2vit/med_vqa_c3/m' \
#                                         --dataset_type='med_vqa' --dataset_cat='cat3' \
#                                         --model_ver='gpt2ViT'


# echo "EFGPT2RS18 EndoVis18"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.00001 \
#                                         --checkpoint_dir='checkpoints/clf_efgpt2rs18/m18/vpe_vp_' \
#                                         --dataset_type='m18' --dataset_cat='cat1' \
#                                         --model_ver='efgpt2rs18'

# echo "GPT2ViTv2 Cholec80"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.000001 \
#                                         --checkpoint_dir='checkpoints/clf_gpt2swin/c80/1' \
#                                         --dataset_type='c80' --dataset_cat='cat1' \
#                                         --model_ver='gpt2Swin'

# echo "GPT2ViT Cholec80"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.0000005 \
#                                         --checkpoint_dir='checkpoints/clf_gpt2vit/c80/' \
#                                         --dataset_type='c80' --dataset_cat='cat1' \
#                                         --model_ver='gpt2ViT'


echo "efgpt2gcvit EndoVis18"
CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.00005    \
                                        --checkpoint_dir='checkpoints/clf_efgpt2gcvit/m18/v0_' \
                                        --dataset_type='m18' --dataset_cat='cat1' \
                                        --model_ver='efgpt2gcViT' --model_subver='v0'

# echo "efvlegpt2swingr EndoVis18"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.00001    \
#                                         --checkpoint_dir='checkpoints/clf_efvlegpt2swingr/m18/v1_' \
#                                         --dataset_type='m18' --dataset_cat='cat1' \
#                                         --model_ver='efvlegpt2Swingr' --model_subver='v1'

# echo "efvlegpt2swingr EndoVis18"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.00001    \
#                                         --checkpoint_dir='checkpoints/clf_efvlegpt2swin/m18/v2_' \
#                                         --dataset_type='m18' --dataset_cat='cat1' \
#                                         --model_ver='efvlegpt2Swingr' --model_subver='v2'

# echo "efvlegpt2swingr EndoVis18"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.00001    \
#                                         --checkpoint_dir='checkpoints/clf_efvlegpt2swin/m18/v2_' \
#                                         --dataset_type='m18' --dataset_cat='cat1' \
#                                         --model_ver='efvlegpt2Swingr' --model_subver='v3'
                                        # --tokenizer_ver='biogpt2v1'

# echo "EFGPT2RS18 MedVQA C1"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.000005 \
#                                         --checkpoint_dir='checkpoints/clf_efgpt2rs18/med_vqa_c1/vpe_' \
#                                         --dataset_type='med_vqa' --dataset_cat='cat1' \
#                                         --model_ver='efgpt2rs18' --model_subver='v0'

# echo "EFGPT2RS18 MedVQA C2"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.000005 \
#                                         --checkpoint_dir='checkpoints/clf_efgpt2rs18/med_vqa_c2/vpe_' \
#                                         --dataset_type='med_vqa' --dataset_cat='cat2' \
#                                         --model_ver='efgpt2rs18' --model_subver='v0'

# echo "EFGPT2RS18 MedVQA C3"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.000005 \
#                                         --checkpoint_dir='checkpoints/clf_efgpt2rs18/med_vqa_c3/vpe_' \
#                                         --dataset_type='med_vqa' --dataset_cat='cat3' \
#                                         --model_ver='efgpt2rs18' --model_subver='v0'   

# echo "EFGPT2RS18 MedVQA Cholec80"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.0000005 \
#                                         --checkpoint_dir='checkpoints/clf_efgpt2rs18/c80/vpe_' \
#                                         --dataset_type='c80' --dataset_cat='cat1' \
#                                         --model_ver='efgpt2rs18' --model_subver='v0'                                  

# lr for GPT
# MedVQA: 0.000005
# M18: 0.00001
# C80: 0.0000001