import os.path

from tqdm import tqdm
import numpy as np
import torch
from utils.draw_util import draw_animation
from math import pi
from torch.utils.tensorboard import SummaryWriter


def train(config, env, agent, pmi, num_episodes, num_steps, frequency):
    """
    :param config:
    :param pmi: pmi network
    :param frequency: 打印消息的频率
    :param num_steps: 每局进行的步数
    :param env:
    :param agent: # 因为所有的无人机共享权重训练, 所以共用一个agent
    :param num_episodes: 局数
    :return:
    """
    # initialize saving list
    writer = SummaryWriter(log_dir=os.path.join(config["save_dir"], 'logs'))  # 可以指定log存储的目录
    return_list = []
    target_tracking_return_list = []
    boundary_punishment_return_list = []
    duplicate_tracking_punishment_return_list = []

    with tqdm(total=num_episodes, desc='Episodes') as pbar:
        for i in range(num_episodes):
            # reset environment
            env.reset(t_v_max=config["target"]["v_max"],
                      t_h_max=pi / float(config["target"]["h_max"]),
                      u_v_max=config["uav"]["v_max"],
                      u_h_max=pi / float(config["uav"]["h_max"]),
                      na=config["environment"]["na"],
                      dc=config["uav"]["dc"],
                      dp=config["uav"]["dp"],
                      dt=config["uav"]["dt"],)

            # reset return
            episode_return = 0
            episode_target_tracking_return = 0
            episode_boundary_punishment_return = 0
            episode_duplicate_tracking_punishment_return = 0

            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': []}

            # episode start
            for _ in range(num_steps):
                action_list = []

                # each uav makes choices first
                for uav in env.uav_list:
                    state = uav.get_local_state()
                    action = agent.take_action(state).item()  # size: 1
                    transition_dict['states'].append(state)
                    action_list.append(action)

                # use action_list to update the environment
                next_state_list, reward_list = env.step(pmi, action_list)  # action: List[int]
                transition_dict['actions'].extend(action_list)
                transition_dict['next_states'].extend(next_state_list)
                transition_dict['rewards'].extend(reward_list['rewards'])

                # update return
                episode_return += sum(reward_list['rewards'])
                episode_target_tracking_return += sum(reward_list['target_tracking_reward'])
                episode_boundary_punishment_return += sum(reward_list['boundary_punishment'])
                episode_duplicate_tracking_punishment_return += sum(reward_list['duplicate_tracking_punishment'])

            # saving return lists
            return_list.append(episode_return)
            target_tracking_return_list.append(episode_target_tracking_return)
            boundary_punishment_return_list.append(episode_boundary_punishment_return)
            duplicate_tracking_punishment_return_list.append(episode_duplicate_tracking_punishment_return)

            # update actor-critic network
            actor_loss, critic_loss = agent.update(transition_dict)
            writer.add_scalar('actor_loss', actor_loss, i)
            writer.add_scalar('critic_loss', critic_loss, i)

            # update pmi network
            if pmi:
                avg_pmi_loss = pmi.train_pmi(config, torch.tensor(np.array(transition_dict["states"])), env.n_uav)
                writer.add_scalar('avg_pmi_loss', avg_pmi_loss, i)

            if (i + 1) % frequency == 0:
                # print some information
                if pmi:
                    pbar.set_postfix({'episode': '%d' % (i + 1),
                                      'return': '%.3f' % np.mean(return_list[-frequency:]),
                                      'actor loss': '%f' % actor_loss,
                                      'critic loss': '%f' % critic_loss,
                                      'avg pmi loss': '%f' % avg_pmi_loss})
                else:
                    pbar.set_postfix({'episode': '%d' % (i + 1),
                                      'return': '%.3f' % np.mean(return_list[-frequency:]),
                                      'actor loss': '%f' % actor_loss,
                                      'critic loss': '%f' % critic_loss})

                # save results and weights
                draw_animation(config=config, env=env, num_steps=num_steps, ep_num=i)
                agent.save(config["save_dir"], i + 1)
                if pmi:
                    pmi.save(config["save_dir"], i + 1)
                env.save_position(config["save_dir"], i + 1)

            pbar.update(1)

    other_return_list = {
        'target_tracking_return_list': target_tracking_return_list,
        'boundary_punishment_return_list': boundary_punishment_return_list,
        'duplicate_tracking_punishment_return_list': duplicate_tracking_punishment_return_list
    }

    writer.close()

    return return_list, other_return_list
