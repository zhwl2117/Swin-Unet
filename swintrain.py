# author: Wenliang Zhong wxz9204@mavs.uta.edu


import argparse

from swinutils import convert_id_to_task_name, get_default_configuration, default_plans_identifier
from swintrainer import SwinTrainer
from config import get_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-network", default='3d_fullres')
    parser.add_argument("-task", help="can be task name or task id", default='5')
    parser.add_argument("-fold", help='0, 1, ..., 5 or \'all\'', default=0)
    parser.add_argument("-val", "--validation_only", help="use this if you want to only run the validation",
                        action="store_true")
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true")
    parser.add_argument("--use_compressed_data", default=False, action="store_true",
                        help="If you set use_compressed_data, the training cases will not be decompressed. Reading compressed data "
                            "is much more CPU and RAM intensive and should only be used if you know what you are "
                            "doing", required=False)
    parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                        default=default_plans_identifier, required=False)
    parser.add_argument("--disable_saving", required=False, action='store_true',
                        help="If set nnU-Net will not save any parameter files (except a temporary checkpoint that "
                            "will be removed at the end of the training). Useful for development when you are "
                            "only interested in the results and want to save some disk space")
    parser.add_argument("--deterministic",
                        help="Makes training deterministic, but reduces training speed substantially. I (Fabian) think "
                            "this is not necessary. Deterministic training will make you overfit to some random seed. "
                            "Don't use that.",
                        required=False, default=False, action="store_true")
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )

    args = parser.parse_args()
    task = args.task
    fold = args.fold
    network = args.network
    plans_identifier = args.p
    validation_only = args.validation_only
    deterministic = args.deterministic
    configs = get_config(args)

    if not task.startwith('Task'):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    if fold != 'all':
        fold = int(fold)

    plans_file, output_folder_name, dataset_directory, batch_dice, stage= get_default_configuration(network, task, plans_identifier)

    trainer = SwinTrainer(configs, plans_file, fold, output_folder=output_folder_name, dataset_directory=dataset_directory,
                        batch_dice=batch_dice, stage=stage, deterministic=deterministic)

    if args.disable_saving:
        trainer.save_final_checkpoint = False # whether or not to save the final checkpoint
        trainer.save_best_checkpoint = False  # whether or not to save the best checkpoint according to
        # self.best_val_eval_criterion_MA
        trainer.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest. We need that in case
        # the training chashes
        trainer.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each

    trainer.initialize(not validation_only)
    if not validation_only:
        if args.continue_training:
                # -c was set, continue a previous training and ignore pretrained weights
                trainer.load_latest_checkpoint()
        else:
            # new training without pretraine weights, do nothing
            pass

        trainer.run_training()
    else:
        pass




if __name__ == '__main__':
    main()
