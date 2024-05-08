import argparse
import os.path
from environment import Environment
from actor_critic import ActorCritic
from utils.args_util import get_config
from train import train
from PMINet import PMINetwork
from utils.data_util import save_csv
from utils.draw_util import plot_reward_curve


def print_variables(vdict, name="args"):
    """
    :param vdict: dict, 待打印的字典
    :param name: str, 打印的字典名称
    :return: None
    """
    print("-----------------------------------------")
    print("|This is the summary of {}:".format(name))
    var = vdict
    for i in var:
        if var[i] is None:
            continue
        print("|{:11}\t: {}".format(i, var[i]))
    print("-----------------------------------------")


def main(args):
    # 获取方法所用的参数
    config = get_config(os.path.join("configs", args.method + ".yaml"))
    print_variables(config)

    # 初始化environment, agent
    env = Environment(n_uav=config["environment"]["n_uav"],
                      m_targets=config["environment"]["m_targets"],
                      x_max=config["environment"]["x_max"],
                      y_max=config["environment"]["y_max"],
                      na=config["environment"]["na"])
    agent = ActorCritic(state_dim=12,
                        hidden_dim=config["actor_critic"]["hidden_dim"],
                        action_dim=config["environment"]["na"],
                        actor_lr=float(config["actor_critic"]["actor_lr"]),
                        critic_lr=float(config["actor_critic"]["critic_lr"]),
                        gamma=float(config["actor_critic"]["gamma"]),
                        device=config["devices"][0])  # 只用第一个device

    # 初始化 pmi
    if args.method == "MAAC":
        pmi = None
    else:
        pmi = PMINetwork(hidden_dim=config["pmi"]["hidden_dim"],
                         b2_size=config["pmi"]["b2_size"])

    return_list, other_return_list = train(config=config,
                                           env=env,
                                           agent=agent,
                                           pmi=pmi,
                                           num_episodes=args.num_episodes,
                                           num_steps=args.num_steps,
                                           frequency=args.frequency)
    save_csv(config, return_list, other_return_list)

    plot_reward_curve(config, return_list, "overall_return")
    plot_reward_curve(config, other_return_list["target_tracking_return_list"],
                      "target_tracking_return_list")
    plot_reward_curve(config, other_return_list["boundary_punishment_return_list"],
                      "boundary_punishment_return_list")
    plot_reward_curve(config, other_return_list["duplicate_tracking_punishment_return_list"],
                      "duplicate_tracking_punishment_return_list")


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="")

    # 添加超参数
    parser.add_argument("--hidden_dim", type=int, default=128, help="actor网络和critic网络的隐藏层维数")
    parser.add_argument("--num_episodes", type=int, default=1, help="训练轮数")
    parser.add_argument("--num_steps", type=int, default=100, help="每轮进行步数")
    parser.add_argument("--frequency", type=int, default=50, help="打印信息及保存的频率")
    parser.add_argument("--method", help="", default="MAAC-R", choices=["MAAC", "MAAC-R", "MAAC-G"])
    # 解析命令行参数
    main_args = parser.parse_args()

    # 调用主函数
    main(main_args)
