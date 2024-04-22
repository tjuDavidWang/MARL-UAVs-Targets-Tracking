from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import draw_picture


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


def train_on_policy_agent(env, agent, num_episodes, num_steps):
    """
    :param num_steps: 每局进行的步数
    :param env:
    :param agent: # TODO 因为所有的无人机共享权重训练, 所以共用一个agent
    :param num_episodes: 局数
    :return:
    """
    return_list = []
    for i in range(num_episodes):
        with tqdm(total=num_episodes, desc='Iteration %d' % i) as pbar:
            # initial environment
            env.reset()
            episode_return = 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': []}

            all_uav_xs = []
            all_uav_ys = []
            all_target_xs = []
            all_target_ys = []

            # episode start
            for _ in range(num_steps):
                action_list = []
                # 存储无人机飞行轨迹的数组

                uav_xs = []
                uav_ys = []

                # each uav makes choices first
                for uav in env.uav_list:
                    state = uav.get_local_state()
                    action = agent.take_action(state).item()  # size: 1
                    transition_dict['states'].append(state)
                    # action_list.extend(action)
                    action_list.append(action)
                    # 收集xy坐标信息，便于作图

                    uav_xs.append(uav.x)
                    uav_ys.append(uav.y)

                all_uav_xs.append(uav_xs)
                all_uav_ys.append(uav_ys)

                # use action_list to update the environment
                next_state_list, reward_list = env.step(action_list)  # action: List[int]
                transition_dict['actions'].extend(action_list)
                transition_dict['next_states'].extend(next_state_list)
                transition_dict['rewards'].extend(reward_list)

                # 存储目标飞行轨迹的数组
                target_xs = []
                target_ys = []
                for target in env.target_list:
                    # 收集xy坐标信息，便于作图
                    target_xs.append(target.x)
                    target_ys.append(target.y)

                all_target_xs.append(target_xs)
                all_target_ys.append(target_ys)

                # update return
                episode_return += sum(reward_list)
            return_list.append(episode_return)
            agent.update(transition_dict)

            if (i+1) % 10 == 0:
                pbar.set_postfix({'episode': '%d' % (num_episodes * i + 1),
                                  'return': '%.3f' % np.mean(return_list[-10:])})
                # 绘制二维图
                uav = env.uav_list[0]
                draw_picture.draw(all_uav_xs, all_uav_ys, all_target_xs, all_target_ys,
                                  num_steps, env.n_uav, env.m_targets, uav.dp, i)
            pbar.update(1)



    return return_list


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
                