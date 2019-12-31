DATA_DIR = '/media/give/HDD3/ld/Documents/datasets/Abdomen/'
RAW_DATA_DIR = '/media/give/HDD3/ld/Documents/datasets/Abdomen/RawData/'


class AbdomenDatasetConfig:
    def __init__(self):
        self.RAW_DATA_TRAINING_DIR = '/media/give/HDD3/ld/Documents/datasets/Abdomen/RawData/Training'
        self.RAW_DATA_EVALUATING_DIR = '/media/give/HDD3/ld/Documents/datasets/Abdomen/RawData/Training'
        self.RAW_DATA_TF_DIR = '/media/give/HDD3/ld/Documents/datasets/Abdomen/RawData/Training/tfrecords'
        self.img_prefix = 'img'
        self.label_prefix = 'label'
        self.labels_mapping = {1: 1, 6: 2}


class ChenSpleenDatasetConfig:
    def __init__(self):
        self.RAW_DATA_TRAINING_DIR = '/media/give/HDD3/ld/Documents/datasets/chen_spleen/origin_nii'
        self.RAW_DATA_EVALUATING_DIR = '/media/give/HDD3/ld/Documents/datasets/chen_spleen/origin_nii'
        self.RAW_DATA_TF_DIR = '/media/give/HDD3/ld/Documents/datasets/chen_spleen/tfrecords'
        self.img_prefix = 'volume-ID_'
        self.label_prefix = 'segmentation-ID_'
        self.labels_mapping = {1: 1}


class LiTSDatasetConfig:
    def __init__(self):
        self.RAW_DATA_TRAINING_DIR = '/media/give/HDD3/ld/Documents/datasets/LiTS/Training_Batch_2_origin'
        self.RAW_DATA_EVALUATING_DIR = '/media/give/HDD3/ld/Documents/datasets/LiTS/Training_Batch_1_origin'
        self.RAW_DATA_TF_DIR = '/media/give/HDD3/ld/Documents/datasets/LiTS/Training_Batch_2_tfrecords'
        self.img_prefix = 'volume-'
        self.label_prefix = 'segmentation-'
        self.labels_mapping = {1: 1}


def getDatasetConfigFactory(dataset_name):
    if dataset_name == 'Abdomen':
        return AbdomenDatasetConfig()
    elif dataset_name == 'chen_spleen':
        return ChenSpleenDatasetConfig()
    elif dataset_name == 'LiTS':
        return LiTSDatasetConfig()

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
    'window_width': 250,
    'prob': None,
}
# Abdomen
# DATASET_V2 = {
#     'name': 'V2',
#     'size': 2623,
#     'window_center': 100,
#     'window_width': 500,
#     'prob': None
# }
# spleen
# DATASET_V2 = {
#     'name': 'V2',
#     'size': 8200,
#     'window_center': 100,
#     'window_width': 500,
#     'prob': None
# }

# LiTS
DATASET_V2 = {
    'name': 'V2',
    'size': 8200,
    'window_center': 100,
    'window_width': 500,
    'prob': None
}


DATASET_V3 = {
    'name': 'V3',
    'size': 1700,
    'window_center': None,
    'window_width': None,
    'prob': 0.5
}

EVALUATING_RANGE = range(1, 11)
name2dataset = {
    'V1': DATASET_V1,
    'V2': DATASET_V2,
    'V3': DATASET_V3
}


def get_dataset_config(name):
    return name2dataset[name]
