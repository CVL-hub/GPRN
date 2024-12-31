
# train
# CUDA_VISIBLE_DEVICES=1 python -W ignore train.py \
#   --dataset isic --data-root ./dataset \
#   --backbone resnet50 --batch-size 6 --shot 1 --refine --lr 0.001 --positive_point 20 --negative_point 20 --alpha 0.5 --fuse_method coff \
#   --weight 0.1 --post_refine



  
#fine-tuning
#for deepglobe lr = 0.001 bs = 6, for fss lr = 0.0003 bs = 6, for isic lr = 0.001 bs = 6, for chestx-ray lr = 0.0003 bs = 6
# CUDA_VISIBLE_DEVICES=0 python -W ignore finetune.py \
#   --dataset fss --data-root ./dataset \
#   --backbone resnet50 --batch-size 6 --shot 1 --refine --lr 0.0003 --positive_point 20 --negative_point 20 --alpha 0.5 --fuse_method coff \
#   --train_seed 802 --weight 0.1 --post_refine --train
   

# # test
CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
  --dataset fss --data-root ./dataset \
  --backbone resnet50 --batch-size 6 --shot 5 --refine --positive_point 20 --negative_point 20 --alpha 0.5 --fuse_method coff \
  --post_refine


