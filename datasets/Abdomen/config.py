DATA_DIR = '/media/give/HDD3/ld/Documents/datasets/Abdomen/'
RAW_DATA_DIR = '/media/give/HDD3/ld/Documents/datasets/Abdomen/RawData/'
RAW_DATA_TRAINING_DIR = '/media/give/HDD3/ld/Documents/datasets/Abdomen/RawData/Training'
RAW_DATA_EVALUATING_DIR = '/media/give/HDD3/ld/Documents/datasets/Abdomen/RawData/Training'
RAW_DATA_TF_DIR = '/media/give/HDD3/ld/Documents/datasets/Abdomen/RawData/Training/tfrecords'
# (1) spleen
# (2) right kidney
# (3) left kidney
# (4) gallbladder
# (5) esophagus
# (6) liver
# (7) stomach
# (8) aorta
# (9) inferior vena cava
# (10) portal vein and splenic vein
# (11) pancreas
# (12) right adrenal gland
# (13) left adrenal gland


# 1 spleen
# 2 liver
name2global_label = {
    'spleen': 1,
    'right kidney': 2,
    'left kidney': 3,
    'gallbladder': 4,
    'esophagus': 5,
    'liver': 6,
    'stomach': 7,
    'aorta': 8,
    'inferior vena cava': 9,
    'portal vein and splenic vein': 10,
    'pancreas': 11,
    'right adrenal gland': 12,
    'left adrenal gland': 13
}

DATASET_V1 = {
    'name': 'V1',
    'size': 1500,
    'window_center': 40,
    'window_width': 250
}
EVALUATING_RANGE = range(1, 11)
name2dataset = {
    'V1': DATASET_V1
}


def get_dataset_config(name):
    return name2dataset[name]
