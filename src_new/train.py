from tqdm import tqdm
import numpy as np
import torch
from utils.draw_util import draw_animation
from math import pi


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
    return_list = []
    target_tracking_return_list = []
    boundary_punishment_return_list = []
    duplicate_tracking_punishment_return_list = []
    with tqdm(total=num_episodes, desc='Episodes') as pbar:
        for i in range(num_episodes):
            # initial environment
            env.reset(t_v_max=pi / config["target"]["v_max"],
                      t_h_max=pi / config["target"]["h_max"],
                      u_v_max=pi / config["uav"]["v_max"],
                      u_h_max=pi / config["uav"]["h_max"],
                      na=config["environment"]["na"],
                      dc=config["uav"]["dc"],
                      dp=config["uav"]["dp"],
                      dt=config["uav"]["dt"],)
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

            return_list.append(episode_return)
            target_tracking_return_list.append(episode_target_tracking_return)
            boundary_punishment_return_list.append(episode_boundary_punishment_return)
            duplicate_tracking_punishment_return_list.append(episode_duplicate_tracking_punishment_return)

            agent.update(transition_dict)

            if pmi:
                pmi.train_pmi(torch.tensor(np.array(transition_dict["states"])), env.n_uav)
            if (i + 1) % frequency == 0:
                pbar.set_postfix({'episode': '%d' % (i + 1),
                                  'return': '%.3f' % np.mean(return_list[-frequency:])})
                draw_animation(config=config, env=env, num_steps=num_steps, ep_num=i)

            pbar.update(1)

            other_return_list = {
                'target_tracking_return_list': target_tracking_return_list,
                'boundary_punishment_return_list': boundary_punishment_return_list,
                'duplicate_tracking_punishment_return_list': duplicate_tracking_punishment_return_list
            }

    return return_list, other_return_list
