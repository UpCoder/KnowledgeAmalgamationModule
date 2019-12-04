#!/usr/bin/env bash
cd ..
nohup python seg_models/train_models.py --target_name=liver --target_label=2 --num_epoches=100 --cross_entropy=1.0 --dice=0.0 --focal_loss=1.0 > nohup_train_student_model.log &