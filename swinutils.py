#  this code is heavily borrow from the nnUNet
#  For more details, please refer to https://github.com/MIC-DKFZ/nnUNet


import torch
import torch.nn as nn
from paths import nnUNet_raw_data, preprocessing_output_dir, nnUNet_cropped_data, network_training_output_dir, default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent

def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d


def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(gpu_id, non_blocking=non_blocking)
    return data


def convert_id_to_task_name(task_id: int):
    startswith = "Task%03.0d" % task_id
    if preprocessing_output_dir is not None:
        candidates_preprocessed = subdirs(preprocessing_output_dir, prefix=startswith, join=False)
    else:
        candidates_preprocessed = []

    if nnUNet_raw_data is not None:
        candidates_raw = subdirs(nnUNet_raw_data, prefix=startswith, join=False)
    else:
        candidates_raw = []

    if nnUNet_cropped_data is not None:
        candidates_cropped = subdirs(nnUNet_cropped_data, prefix=startswith, join=False)
    else:
        candidates_cropped = []

    candidates_trained_models = []
    if network_training_output_dir is not None:
        for m in ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres']:
            if isdir(join(network_training_output_dir, m)):
                candidates_trained_models += subdirs(join(network_training_output_dir, m), prefix=startswith, join=False)

    all_candidates = candidates_cropped + candidates_preprocessed + candidates_raw + candidates_trained_models
    unique_candidates = np.unique(all_candidates)
    if len(unique_candidates) > 1:
        raise RuntimeError("More than one task name found for task id %d. Please correct that. (I looked in the "
                        "following folders:\n%s\n%s\n%s" % (task_id, nnUNet_raw_data, preprocessing_output_dir,
                                                            nnUNet_cropped_data))
    if len(unique_candidates) == 0:
        raise RuntimeError("Could not find a task with the ID %d. Make sure the requested task ID exists and that "
                        "nnU-Net knows where raw and preprocessed data are located (see Documentation - "
                        "Installation). Here are your currently defined folders:\nnnUNet_preprocessed=%s\nRESULTS_"
                        "FOLDER=%s\nnnUNet_raw_data_base=%s\nIf something is not right, adapt your environemnt "
                        "variables." %
                        (task_id,
                            os.environ.get('nnUNet_preprocessed') if os.environ.get('nnUNet_preprocessed') is not None else 'None',
                            os.environ.get('RESULTS_FOLDER') if os.environ.get('RESULTS_FOLDER') is not None else 'None',
                            os.environ.get('nnUNet_raw_data_base') if os.environ.get('nnUNet_raw_data_base') is not None else 'None',
                            ))
    return unique_candidates[0]


def convert_task_name_to_id(task_name: str):
    assert task_name.startswith("Task")
    task_id = int(task_name[4:7])
    return task_id

def get_default_configuration(network, task, network_trainer, plans_identifier=default_plans_identifier,):
    assert network in ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres'], \
        "network can only be one of the following: \'3d\', \'3d_lowres\', \'3d_fullres\', \'3d_cascade_fullres\'"

    dataset_directory = join(preprocessing_output_dir, task)

    if network == '2d':
        plans_file = join(preprocessing_output_dir, task, plans_identifier + "_plans_2D.pkl")
    else:
        plans_file = join(preprocessing_output_dir, task, plans_identifier + "_plans_3D.pkl")

    plans = load_pickle(plans_file)
    possible_stages = list(plans['plans_per_stage'].keys())

    if (network == '3d_cascade_fullres' or network == "3d_lowres") and len(possible_stages) == 1:
        raise RuntimeError("3d_lowres/3d_cascade_fullres only applies if there is more than one stage. This task does "
                           "not require the cascade. Run 3d_fullres instead")

    if network == '2d' or network == "3d_lowres":
        stage = 0
    else:
        stage = possible_stages[-1]

    output_folder_name = join(network_training_output_dir, network, task, network_trainer + "__" + plans_identifier)

    print("###############################################")
    print("I am running the following nnUNet: %s" % network)
    print("For that I will be using the following configuration:")
    summarize_plans(plans_file)
    print("I am using stage %d from these plans" % stage)

    if (network == '2d' or len(possible_stages) > 1) and not network == '3d_lowres':
        batch_dice = True
        print("I am using batch dice + CE loss")
    else:
        batch_dice = False
        print("I am using sample dice + CE loss")

    print("\nI am using data from this folder: ", join(dataset_directory, plans['data_identifier']))
    print("###############################################")
    return plans_file, output_folder_name, dataset_directory, batch_dice, stage

# This file is intended to double check nnUNets design choices. It is intended to be used for developent purposes only
def summarize_plans(file):
    plans = load_pickle(file)
    print("num_classes: ", plans['num_classes'])
    print("modalities: ", plans['modalities'])
    print("use_mask_for_norm", plans['use_mask_for_norm'])
    print("keep_only_largest_region", plans['keep_only_largest_region'])
    print("min_region_size_per_class", plans['min_region_size_per_class'])
    print("min_size_per_class", plans['min_size_per_class'])
    print("normalization_schemes", plans['normalization_schemes'])
    print("stages...\n")

    for i in range(len(plans['plans_per_stage'])):
        print("stage: ", i)
        print(plans['plans_per_stage'][i])
        print("")
