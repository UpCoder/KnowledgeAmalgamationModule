#!/usr/bin/env bash
cd ..
nohup python seg_models/train_models.py --dataset_name=V1 --backbone_name=resnet --target_name=spleen --target_label=1 --num_epoches=100 --cross_entropy=1.0 --dice=0.0 --focal_loss=0.0 > nohup_train_student_model.log &