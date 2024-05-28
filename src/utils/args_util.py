import os
import random
import time
import yaml
import numpy as np
import torch


def get_config(config_file):
    """
    :param config_file: str, 超参数所在的文件位置
    :return: dict, 解析后的超参数字典
    """
    with open(config_file, 'r', encoding="UTF-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # set global seed of random, numpy and torch
    if 'seed' in config and config['seed'] is not None:
        np.random.seed(config['seed'])
        random.seed(config['seed'])
        torch.manual_seed(config['seed'])

    # create name for this experiment
    run_id = str(os.getpid())
    exp_name = '_'.join([
        config['exp_name'],
        time.strftime('%Y-%b-%d-%H-%M-%S'), run_id
    ])

    # save paths
    save_dir = os.path.join(config['result_dir'], exp_name)
    args_save_name = os.path.join(save_dir, 'args.yaml')
    config['save_dir'] = save_dir

    # snapshot hyperparameters
    mkdir(config['result_dir'])
    mkdir(save_dir)
    mkdir(os.path.join(save_dir, "actor"))
    mkdir(os.path.join(save_dir, "critic"))
    mkdir(os.path.join(save_dir, "pmi"))
    mkdir(os.path.join(save_dir, "animated"))
    mkdir(os.path.join(save_dir, "t_xy"))
    mkdir(os.path.join(save_dir, "u_xy"))
    mkdir(os.path.join(save_dir, "covered_target_num"))

    # create cuda devices
    set_device(config)

    with open(args_save_name, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return config


def mkdir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


def set_device(config):
    """
    :param config: dict
    :return: None
    """
    if config['gpus'] == -1 or not torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print('use cpu')
        config['devices'] = [torch.device('cpu')]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in range(config['first_device'],
                                                                            config['first_device'] + config['gpus']))
        print('use gpus: {}'.format(config['gpus']))
        config['devices'] = [torch.device('cuda', i) for i in range(config['first_device'],
                                                                    config['first_device'] + config['gpus'])]


if __name__ == "__main__":
    example = get_config("../configs/MAAC.yaml")
    print(type(example))
    print(example)
