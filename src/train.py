import os.path
import csv
from tqdm import tqdm
import numpy as np
import torch
from utils.draw_util import draw_animation
from torch.utils.tensorboard import SummaryWriter
import random
import collections


class ReturnValueOfTrain:
    def __init__(self):
        self.return_list = []
        self.target_tracking_return_list = []
        self.boundary_punishment_return_list = []
        self.duplicate_tracking_punishment_return_list = []
        self.average_covered_targets_list = []
        self.max_covered_targets_list = []

    def item(self):
        value_dict = {
            'return_list': self.return_list,
            'target_tracking_return_list': self.target_tracking_return_list,
            'boundary_punishment_return_list': self.boundary_punishment_return_list,
            'duplicate_tracking_punishment_return_list': self.duplicate_tracking_punishment_return_list,
            'average_covered_targets_list': self.average_covered_targets_list,
            'max_covered_targets_list': self.max_covered_targets_list
        }
        return value_dict

    def save_epoch(self, reward, tt_return, bp_return, dtp_return, average_targets, max_targets):
        self.return_list.append(reward)
        self.target_tracking_return_list.append(tt_return)
        self.boundary_punishment_return_list.append(bp_return)
        self.duplicate_tracking_punishment_return_list.append(dtp_return)
        self.average_covered_targets_list.append(average_targets)
        self.max_covered_targets_list.append(max_targets)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, transition_dict):
        # 从transition_dict中提取各个列表
        states = transition_dict['states']
        actions = transition_dict['actions']
        rewards = transition_dict['rewards']
        next_states = transition_dict['next_states']

        # 将各个元素合并成元组，并添加到缓冲区中
        experiences = zip(states, actions, rewards, next_states)
        self.buffer.extend(experiences)

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, min(batch_size, self.size()))
        states, actions, rewards, next_states = zip(*transitions)

        # 构造返回的字典
        sample_dict = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states
        }
        return sample_dict

    def size(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = collections.deque(maxlen=capacity)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition_dict):
        states = transition_dict['states']
        actions = transition_dict['actions']
        rewards = transition_dict['rewards']
        next_states = transition_dict['next_states']

        experiences = zip(states, actions, rewards, next_states)

        for experience in experiences:
            max_priority = self.priorities.max() if self.buffer else 1.0

            if len(self.buffer) < self.capacity:
                self.buffer.append(experience)
            else:
                self.buffer[self.pos] = experience

            self.priorities[self.pos] = max_priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return dict(states=[], actions=[], rewards=[], next_states=[]), None, None

        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]

        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states = np.array(batch[0])
        actions = np.array(batch[1])
        rewards = np.array(batch[2])
        next_states = np.array(batch[3])

        sample_dict = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
        }
        return sample_dict, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

    def size(self):
        return len(self.buffer)


def operate_epoch(config, env, agent, pmi, num_steps, cwriter_state=None, cwriter_prob=None):
    """
    :param config:
    :param env:
    :param agent: 
    :param pmi: 
    :param num_steps: 
    :param cwriter_state: 用于记录一个epoch内的state信息, 调试bug时使用
    :param cwriter_prob:  用于记录一个epoch内的prob信息, 调试bug时使用
    :return: 
    """
    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': []}
    episode_return = 0
    episode_target_tracking_return = 0
    episode_boundary_punishment_return = 0
    episode_duplicate_tracking_punishment_return = 0
    covered_targets_list = []

    for i in range(num_steps):
        config['step'] = i + 1
        action_list = []

        # each uav makes choices first
        for uav in env.uav_list:
            state = uav.get_local_state()
            if cwriter_state:
                cwriter_state.writerow(state.tolist())
            action, probs = agent.take_action(state)
            if cwriter_prob:
                cwriter_prob.writerow(probs.tolist())
            transition_dict['states'].append(state)
            action_list.append(action.item())

        # use action_list to update the environment
        next_state_list, reward_list, covered_targets = env.step(config, pmi, action_list)  # action: List[int]
        transition_dict['actions'].extend(action_list)
        transition_dict['next_states'].extend(next_state_list)
        transition_dict['rewards'].extend(reward_list['rewards'])

        episode_return += sum(reward_list['rewards'])
        episode_target_tracking_return += sum(reward_list['target_tracking_reward'])
        episode_boundary_punishment_return += sum(reward_list['boundary_punishment'])
        episode_duplicate_tracking_punishment_return += sum(reward_list['duplicate_tracking_punishment'])
        covered_targets_list.append(covered_targets)

    episode_return /= num_steps * env.n_uav
    episode_target_tracking_return /= num_steps * env.n_uav
    episode_boundary_punishment_return /= num_steps * env.n_uav
    episode_duplicate_tracking_punishment_return /= num_steps * env.n_uav
    average_covered_targets = np.mean(covered_targets_list)
    max_covered_targets = np.max(covered_targets_list)

    return (transition_dict, episode_return, episode_target_tracking_return,
            episode_boundary_punishment_return, episode_duplicate_tracking_punishment_return,
            average_covered_targets, max_covered_targets)


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
    save_dir = os.path.join(config["save_dir"], "logs")
    writer = SummaryWriter(log_dir=save_dir)  # 可以指定log存储的目录
    return_value = ReturnValueOfTrain()
    # buffer = ReplayBuffer(config["actor_critic"]["buffer_size"])
    buffer = PrioritizedReplayBuffer(config["actor_critic"]["buffer_size"])
    if config["actor_critic"]["sample_size"] > 0:
        sample_size = config["actor_critic"]["sample_size"]
    else:
        sample_size = config["environment"]["n_uav"] * num_steps

    with open(os.path.join(save_dir, 'state.csv'), mode='w', newline='') as state_file, \
            open(os.path.join(save_dir, 'prob.csv'), mode='w', newline='') as prob_file:
        cwriter_state = csv.writer(state_file)
        cwriter_prob = csv.writer(prob_file)

        cwriter_state.writerow(['state'])  # 写入state.csv的表头
        cwriter_prob.writerow(['prob'])  # 写入prob.csv的表头

        with tqdm(total=num_episodes, desc='Episodes') as pbar:
            for i in range(num_episodes):
                # reset environment from config yaml file
                env.reset(config=config)

                # episode start
                # transition_dict, reward, tt_return, bp_return, \
                #     dtp_return = operate_epoch(config, env, agent, pmi, num_steps, cwriter_state, cwriter_prob)
                transition_dict, reward, tt_return, bp_return, \
                    dtp_return, average_targets, max_targets = operate_epoch(config, env, agent, pmi, num_steps)
                writer.add_scalar('reward', reward, i)
                writer.add_scalar('target_tracking_return', tt_return, i)
                writer.add_scalar('boundary_punishment', bp_return, i)
                writer.add_scalar('duplicate_tracking_punishment', dtp_return, i)
                writer.add_scalar('average_covered_targets', average_targets, i)
                writer.add_scalar('max_covered_targets', max_targets, i)

                # saving return lists
                return_value.save_epoch(reward, tt_return, bp_return, dtp_return, average_targets, max_targets)

                # sample from buffer
                buffer.add(transition_dict)
                # sample_dict = buffer.sample(sample_size)
                sample_dict, indices, _ = buffer.sample(sample_size)

                # update actor-critic network
                actor_loss, critic_loss, td_errors = agent.update(sample_dict)
                writer.add_scalar('actor_loss', actor_loss, i)
                writer.add_scalar('critic_loss', critic_loss, i)

                # update buffer
                buffer.update_priorities(indices, td_errors.abs().detach().cpu().numpy())

                # update pmi network
                if pmi:
                    avg_pmi_loss = pmi.train_pmi(config, torch.tensor(np.array(sample_dict["states"])), env.n_uav)
                    writer.add_scalar('avg_pmi_loss', avg_pmi_loss, i)

                # save & print
                if (i + 1) % frequency == 0:
                    # print some information
                    if pmi:
                        pbar.set_postfix({'episode': '%d' % (i + 1),
                                          'return': '%.3f' % np.mean(return_value.return_list[-frequency:]),
                                          'actor loss': '%f' % actor_loss,
                                          'critic loss': '%f' % critic_loss,
                                          'avg pmi loss': '%f' % avg_pmi_loss})
                    else:
                        pbar.set_postfix({'episode': '%d' % (i + 1),
                                          'return': '%.3f' % np.mean(return_value.return_list[-frequency:]),
                                          'actor loss': '%f' % actor_loss,
                                          'critic loss': '%f' % critic_loss})

                    # save results and weights
                    draw_animation(config=config, env=env, num_steps=num_steps, ep_num=i)
                    agent.save(save_dir=config["save_dir"], epoch_i=i + 1)
                    if pmi:
                        pmi.save(save_dir=config["save_dir"], epoch_i=i + 1)
                    env.save_position(save_dir=config["save_dir"], epoch_i=i + 1)
                    env.save_covered_num(save_dir=config["save_dir"], epoch_i=i + 1)

                # episode end
                pbar.update(1)

    writer.close()

    return return_value.item()


def evaluate(config, env, agent, pmi, num_steps):
    """
    :param config:
    :param pmi: pmi network
    :param num_steps: 每局进行的步数
    :param env:
    :param agent: # 因为所有的无人机共享权重训练, 所以共用一个agent
    :return:
    """
    # initialize saving list
    return_value = ReturnValueOfTrain()

    # reset environment from config yaml file
    env.reset(config=config)

    # episode start
    transition_dict, reward, tt_return, bp_return, dtp_return, average_targets, max_targets = operate_epoch(config, env, agent, pmi, num_steps)

    # saving return lists
    return_value.save_epoch(reward, tt_return, bp_return, dtp_return, average_targets, max_targets)

    # save results and weights
    draw_animation(config=config, env=env, num_steps=num_steps, ep_num=0)
    env.save_position(save_dir=config["save_dir"], epoch_i=0)
    env.save_covered_num(save_dir=config["save_dir"], epoch_i=0)

    return return_value.item()

def run_epoch(config, pmi, env, num_steps):
    """
    :param config:
    :param env:
    :param num_steps:
    :return:
    """
    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': []}
    episode_return = 0
    episode_target_tracking_return = 0
    episode_boundary_punishment_return = 0
    episode_duplicate_tracking_punishment_return = 0
    covered_targets_list = []

    for _ in range(num_steps):
        action_list = []
        # uav_tracking_status = [0] * len(env.uav_list)

        # # each uav makes choices first
        # for uav in env.uav_list:
        #     action, target_index = uav.get_action_by_direction(env.target_list, env.uav_list, uav_tracking_status)  # TODO
        #     uav_tracking_status[target_index] = 1
        #     action_list.append(action)
        for uav in env.uav_list:
            action = uav.get_action_by_direction(env.target_list, env.uav_list)  # TODO
            action_list.append(action)

        next_state_list, reward_list, covered_targets = env.step(config, pmi, action_list)  # TODO

        # use action_list to update the environment
        transition_dict['actions'].extend(action_list)
        transition_dict['rewards'].extend(reward_list['rewards'])

        episode_return += sum(reward_list['rewards'])
        episode_target_tracking_return += sum(reward_list['target_tracking_reward'])
        episode_boundary_punishment_return += sum(reward_list['boundary_punishment'])
        episode_duplicate_tracking_punishment_return += sum(reward_list['duplicate_tracking_punishment'])
        covered_targets_list.append(covered_targets)

    average_covered_targets = np.mean(covered_targets_list)
    max_covered_targets = np.max(covered_targets_list)

    return (transition_dict, episode_return, episode_target_tracking_return,
            episode_boundary_punishment_return, episode_duplicate_tracking_punishment_return,
            average_covered_targets, max_covered_targets)

def run(config, env, pmi, num_steps):
    """
    :param config:
    :param num_steps: 每局进行的步数
    :param env:
    :return:
    """
    # initialize saving list
    return_value = ReturnValueOfTrain()

    # reset environment from config yaml file
    env.reset(config=config)

    # episode start
    transition_dict, reward, tt_return, bp_return, dtp_return, average_targets, max_targets = run_epoch(config, pmi, env, num_steps)

    # saving return lists
    return_value.save_epoch(reward, tt_return, bp_return, dtp_return, average_targets, max_targets)

    # save results and weights
    draw_animation(config=config, env=env, num_steps=num_steps, ep_num=0)
    env.save_position(save_dir=config["save_dir"], epoch_i=0)
    env.save_covered_num(save_dir=config["save_dir"], epoch_i=0)

    return return_value.item()