from tqdm import tqdm
import numpy as np
import torch
import collections
import random
from toolkits import draw_animation


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_on_policy_agent(env, agent, pmi, num_episodes, num_steps, frequency=50):
    """
    :param pmi: pmi network
    :param frequency: 打印消息的频率
    :param num_steps: 每局进行的步数
    :param env:
    :param agent: # TODO 因为所有的无人机共享权重训练, 所以共用一个agent
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
            env.reset()
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

            pmi.train_pmi(torch.tensor(np.array(transition_dict["states"])), env.n_uav)
            if (i + 1) % frequency == 0:
                pbar.set_postfix({'episode': '%d' % (i + 1),
                                  'return': '%.3f' % np.mean(return_list[-frequency:])})
                draw_animation(env, num_steps=num_steps, ep_num=i)

            pbar.update(1)
            
            other_return_list = {
                'target_tracking_return_list' :target_tracking_return_list,
                'boundary_punishment_return_list':boundary_punishment_return_list,
                'duplicate_tracking_punishment_return_list':duplicate_tracking_punishment_return_list
            }

    return return_list, other_return_list


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a,
                                           'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, _lambda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * _lambda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
                