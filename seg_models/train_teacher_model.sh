#!/usr/bin/env bash
cd ..
nohup python seg_models/train_models.py --dataset_name=V2 --save_dir=/media/give/HDD3/ld/Documents/datasets/LiTS/ck/ --backbone_name=resnet --target_name=liver --dataset_name=LiTS --target_label=1 --num_epoches=100 --cross_entropy=0.0 --dice=0.0 --focal_loss=1.0 --triplet_loss=0.0 > nohup_train_student_model.log &