#!/usr/bin/env bash
DATASET_DIR="/media/give/HDD3/ld/Documents/datasets/Abdomen/RawData/Training/img"
echo $DATASET_DIR
for file in `ls $DATASET_DIR`
do
    echo $file
    if [[ ${file} =~ gz$ ]]; then
        ABS_PATH=$DATASET_DIR/$file
        echo gunzip $ABS_PATH
        `gunzip $ABS_PATH`
    fi
done

DATASET_DIR="/media/give/HDD3/ld/Documents/datasets/Abdomen/RawData/Training/label"
echo $DATASET_DIR
for file in `ls $DATASET_DIR`
do
    echo $file
    if [[ ${file} =~ gz$ ]]; then
        ABS_PATH=$DATASET_DIR/$file
        echo gunzip $ABS_PATH
        `gunzip $ABS_PATH`
    fi
done