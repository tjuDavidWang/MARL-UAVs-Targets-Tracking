import csv
import os.path


def save_csv(config, return_list):
    """
    :param config:
    :param return_list:
        return_list = {
        'return_list': self.return_list,
        'target_tracking_return_list' :target_tracking_return_list,
        'boundary_punishment_return_list':boundary_punishment_return_list,
        'duplicate_tracking_punishment_return_list':duplicate_tracking_punishment_return_list
    }
    :return:
    """
    with open(os.path.join(config["save_dir"], 'return_list.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Reward'])  # 写入表头
        for reward in return_list['return_list']:
            writer.writerow([reward])

    with open(os.path.join(config["save_dir"], 'target_tracking_return_list.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['target_tracking'])  # 写入表头
        for reward in return_list['target_tracking_return_list']:
            writer.writerow([reward])

    with open(os.path.join(config["save_dir"], 'boundary_punishment_return_list.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['boundary_punishment'])  # 写入表头
        for reward in return_list['boundary_punishment_return_list']:
            writer.writerow([reward])

    with open(os.path.join(config["save_dir"], 'duplicate_tracking_punishment_return_list.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['duplicate_tracking_punishment'])  # 写入表头
        for reward in return_list['duplicate_tracking_punishment_return_list']:
            writer.writerow([reward])
