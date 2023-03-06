# EFVLEGPT2RS18Classification: efvlegpt2rs18/
#     v0: visual embedding : Default patch1 + embedding form VB + GPT2 decoder
#     v1: visual embedding : Default patch1 + from nn.linear    + GPT2 decoder
#     v2: visual embedding : visual patches + embedding form VB + GPT2 decoder
#     v3: visual embedding : visual patches + from nn.linear    + GPT2 decoder
# EFVLEGPT2SwinClassification: efvlegpt2Swin/
#     v0: visual embedding : Default patch1 + embedding form VB + GPT2 decoder
#     v1: visual embedding : Default patch1 + GPT2 decoder


# echo "efvlegpt2rs18 EndoVis18"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.00001 --epochs=70 \
#                                         --checkpoint_dir='checkpoints/efvlegpt2rs18/m18/v3_' \
#                                         --dataset_type='m18' --dataset_cat='cat1' --tokenizer_ver='gpt2v1' \
#                                         --model_ver='efvlegpt2rs18' --model_subver='v3' #--vis_pos_emb= zeroes


# echo "efvlegpt2Swin EndoVis18"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.00001 --epochs=60 \
#                                         --checkpoint_dir='checkpoints/efvlegpt2Swin/m18/v0_qf_' \
#                                         --dataset_type='m18' --dataset_cat='cat1' --tokenizer_ver='gpt2v1' \
#                                         --model_ver='efvlegpt2Swin' --model_subver='v0' # --vis_pos_emb='pos'

# echo "efvlegpt2Swin EndoVis18"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.00001 --epochs=60 \
#                                         --checkpoint_dir='checkpoints/efvlegpt2Swin/m18/v1_qf_' \
#                                         --dataset_type='m18' --dataset_cat='cat1' --tokenizer_ver='gpt2v1' \
#                                         --model_ver='efvlegpt2Swin' --model_subver='v1' # --vis_pos_emb='pos'

# echo "efvlegpt2Swin EndoVis18"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.00001 --epochs=60 \
#                                         --checkpoint_dir='checkpoints/efvlegpt2Swin/m18/v0_z_qf_' \
#                                         --dataset_type='m18' --dataset_cat='cat1' --tokenizer_ver='gpt2v1' \
#                                         --model_ver='efvlegpt2Swin' --model_subver='v0' --vis_pos_emb='zeroes'

# echo "efvlegpt2Swin EndoVis18"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.00001 --epochs=60 \
#                                         --checkpoint_dir='checkpoints/efvlegpt2Swin/m18/v1_z_qf_vs_' \
#                                         --dataset_type='m18' --dataset_cat='cat1' --tokenizer_ver='gpt2v1' \
#                                         --model_ver='efvlegpt2Swin' --model_subver='v1' --vis_pos_emb='zeroes'

# echo "efvlegpt2Swin Cholec80"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.000005 \
#                                         --checkpoint_dir='checkpoints/efvlegpt2Swin/c80/v0_qf_' \
#                                         --dataset_type='c80' --dataset_cat='cat1' --tokenizer_ver='gpt2v1' \
#                                         --model_ver='efvlegpt2Swin' --model_subver='v0' #--vis_pos_emb='zeroes'

echo "efvlegpt2Swin Cholec80"
CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.000005 \
                                        --checkpoint_dir='checkpoints/efvlegpt2Swin/c80/v1_qf_' \
                                        --dataset_type='c80' --dataset_cat='cat1' --tokenizer_ver='gpt2v1' \
                                        --model_ver='efvlegpt2Swin' --model_subver='v1' #--vis_pos_emb='zeroes'

# echo "efvlegpt2rs18 EndoVis18"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.00001 --epochs=60 \
#                                         --checkpoint_dir='checkpoints/efvlegpt2rs18/m18/v0_z_qf_' \
#                                         --dataset_type='m18' --dataset_cat='cat1' --tokenizer_ver='gpt2v1' \
#                                         --model_ver='efvlegpt2rs18' --model_subver='v0' --vis_pos_emb='zeroes'

# echo "efvlegpt2rs18 EndoVis18"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.00001 --epochs=60 \
#                                         --checkpoint_dir='checkpoints/efvlegpt2rs18/m18/v1_z_qf_' \
#                                         --dataset_type='m18' --dataset_cat='cat1' --tokenizer_ver='gpt2v1' \
#                                         --model_ver='efvlegpt2rs18' --model_subver='v1' --vis_pos_emb='zeroes'

# echo "efvlegpt2rs18 EndoVis18"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.00001 --epochs=60 \
#                                         --checkpoint_dir='checkpoints/efvlegpt2rs18/m18/v0_p_qf_' \
#                                         --dataset_type='m18' --dataset_cat='cat1' --tokenizer_ver='gpt2v1' \
#                                         --model_ver='efvlegpt2rs18' --model_subver='v0' --vis_pos_emb='pos'

# echo "efvlegpt2rs18 EndoVis18"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.00001 --epochs=60 \
#                                         --checkpoint_dir='checkpoints/efvlegpt2rs18/m18/v1_p_qf_' \
#                                         --dataset_type='m18' --dataset_cat='cat1' --tokenizer_ver='gpt2v1' \
#                                         --model_ver='efvlegpt2rs18' --model_subver='v1' --vis_pos_emb='pos'

# echo "efvlegpt2rs18 EndoVis18"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.00001 --epochs=60 \
#                                         --checkpoint_dir='checkpoints/efvlegpt2rs18/m18/v2_p_qf_' \
#                                         --dataset_type='m18' --dataset_cat='cat1' --tokenizer_ver='gpt2v1' \
#                                         --model_ver='efvlegpt2rs18' --model_subver='v2' --vis_pos_emb='pos'

# echo "efvlegpt2rs18 EndoVis18"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.00001 --epochs=60 \
#                                         --checkpoint_dir='checkpoints/efvlegpt2rs18/m18/v3_p_qf_' \
#                                         --dataset_type='m18' --dataset_cat='cat1' --tokenizer_ver='gpt2v1' \
#                                         --model_ver='efvlegpt2rs18' --model_subver='v3' --vis_pos_emb='pos'

# echo "efvlegpt2ViT EndoVis18"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.00001 --epochs=60  --batch_size=32  \
#                                         --checkpoint_dir='checkpoints/efvlegpt2ViT/m18/v0_qf_' \
#                                         --dataset_type='m18' --dataset_cat='cat1' --tokenizer_ver='gpt2v1' \
#                                         --model_ver='efvlegpt2ViT' --model_subver='v0' #--vis_pos_emb='zeroes'

# echo "efvlegpt2ViT EndoVis18"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.00001 --epochs=60  --batch_size=32  \
#                                         --checkpoint_dir='checkpoints/efvlegpt2ViT/m18/v1_qf_' \
#                                         --dataset_type='m18' --dataset_cat='cat1' --tokenizer_ver='gpt2v1' \
#                                         --model_ver='efvlegpt2ViT' --model_subver='v1' #--vis_pos_emb='zeroes'

# echo "efvlegpt2ViT EndoVis18"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.00001 --epochs=60  --batch_size=32  \
#                                         --checkpoint_dir='checkpoints/efvlegpt2ViT/m18/v0_z_qf_' \
#                                         --dataset_type='m18' --dataset_cat='cat1' --tokenizer_ver='gpt2v1' \
#                                         --model_ver='efvlegpt2ViT' --model_subver='v0' --vis_pos_emb='zeroes'

# echo "efvlegpt2ViT EndoVis18"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.00001 --epochs=60  --batch_size=32  \
#                                         --checkpoint_dir='checkpoints/efvlegpt2ViT/m18/v0_p_qf_' \
#                                         --dataset_type='m18' --dataset_cat='cat1' --tokenizer_ver='gpt2v1' \
#                                         --model_ver='efvlegpt2ViT' --model_subver='v0' --vis_pos_emb='pos'

# echo "efvlegpt2rs18 EndoVis18"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.00001 --epochs=60 \
#                                         --checkpoint_dir='checkpoints/efvlegpt2rs18/m18/v0_qf_' \
#                                         --dataset_type='m18' --dataset_cat='cat1' --tokenizer_ver='gpt2v1' \
#                                         --model_ver='efvlegpt2rs18' --model_subver='v0' #--vis_pos_emb='pos'

# echo "efvlegpt2rs18 EndoVis18"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.00001 --epochs=60 \
#                                         --checkpoint_dir='checkpoints/efvlegpt2rs18/m18/v1_qf_' \
#                                         --dataset_type='m18' --dataset_cat='cat1' --tokenizer_ver='gpt2v1' \
#                                         --model_ver='efvlegpt2rs18' --model_subver='v1' #--vis_pos_emb='pos'

# echo "efvlegpt2rs18 EndoVis18"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.00001 --epochs=60 \
#                                         --checkpoint_dir='checkpoints/efvlegpt2rs18/m18/v2_qf_' \
#                                         --dataset_type='m18' --dataset_cat='cat1' --tokenizer_ver='gpt2v1' \
#                                         --model_ver='efvlegpt2rs18' --model_subver='v2' #--vis_pos_emb='pos'

# echo "efvlegpt2rs18 EndoVis18"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.00001 --epochs=60 \
#                                         --checkpoint_dir='checkpoints/efvlegpt2rs18/m18/v3_qf_' \
#                                         --dataset_type='m18' --dataset_cat='cat1' --tokenizer_ver='gpt2v1' \
#                                         --model_ver='efvlegpt2rs18' --model_subver='v3' #--vis_pos_emb='pos'

# echo "efvlegpt2rs18 Cholec80"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.000005 \
#                                         --checkpoint_dir='checkpoints/efvlegpt2rs18/c80/v3_z_qf_' \
#                                         --dataset_type='c80' --dataset_cat='cat1' --tokenizer_ver='gpt2v1' \
#                                         --model_ver='efvlegpt2rs18' --model_subver='v3' --vis_pos_emb='zeroes'

# echo "efvlegpt2Swin Cholec80"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.000005 \
#                                         --checkpoint_dir='checkpoints/efvlegpt2Swin/c80/v1_z_qf_' \
#                                         --dataset_type='c80' --dataset_cat='cat1' --tokenizer_ver='gpt2v1' \
#                                         --model_ver='efvlegpt2Swin' --model_subver='v1' --vis_pos_emb='zeroes'
                                        
# echo "efvlegpt2ViT Cholec80"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.000005  --batch_size=32  \
#                                         --checkpoint_dir='checkpoints/efvlegpt2ViT/c80/v1_z_qf_' \
#                                         --dataset_type='c80' --dataset_cat='cat1' --tokenizer_ver='gpt2v1' \
#                                         --model_ver='efvlegpt2ViT' --model_subver='v1' --vis_pos_emb='zeroes'


# echo "efvlegpt2Swin Cholec80"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.000005 \
#                                         --checkpoint_dir='checkpoints/efvlegpt2Swin/c80/v1_z_' \
#                                         --dataset_type='c80' --dataset_cat='cat1' --tokenizer_ver='gpt2v1' \
#                                         --model_ver='efvlegpt2Swin' --model_subver='v1' --vis_pos_emb='zeroes'

#--------------------------------------------------------------------
# echo "efvlegpt2rs18 Cholec80"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.000005 \
#                                         --checkpoint_dir='checkpoints/efvlegpt2rs18/c80/v3_p_qf_' \
#                                         --dataset_type='c80' --dataset_cat='cat1' --tokenizer_ver='gpt2v1' \
#                                         --model_ver='efvlegpt2rs18' --model_subver='v3' --vis_pos_emb='pos'


# echo "efvlegpt2ViT Cholec80"
# CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.000005  --batch_size=32  \
#                                         --checkpoint_dir='checkpoints/efvlegpt2ViT/c80/v1_p_qf_' \
#                                         --dataset_type='c80' --dataset_cat='cat1' --tokenizer_ver='gpt2v1' \
#                                         --model_ver='efvlegpt2ViT' --model_subver='v1' --vis_pos_emb='pos'


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


# lr for GPT
# MedVQA: 0.000005
# M18: 0.00001
# C80: 0.0000001