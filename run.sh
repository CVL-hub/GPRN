




  
# fine-tuning
CUDA_VISIBLE_DEVICES=0 python -W ignore ifa-no-train.py \
  --dataset fss --data-root ./dataset \
  --backbone resnet50 --batch-size 6 --shot 1 --refine --lr 0.0005 --positive_point 20 --negative_point 20 --alpha 0.5 --fuse_method entropy \
  --train_seed 802 --weight 0.1 --post_refine
   

# test
CUDA_VISIBLE_DEVICES=0 python -W ignore test-no-train.py \
  --dataset fss --data-root ./dataset \
  --backbone resnet50 --batch-size 6 --shot 1 --refine --positive_point 20 --negative_point 20 --alpha 0.5 --fuse_method entropy \
  --post_refine


