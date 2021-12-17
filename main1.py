from runner_dgn import Runner_DGN
from runner_maddpg import Runner_maddpg
from runner_ppo import Runner_PPO
from common.arguments import get_args
from common.utils import make_env
import numpy as np
import torch


if __name__ == '__main__':
    # get the params
    args = get_args()
    env, args = make_env(args)
    runner = Runner_maddpg(args, env)
    if args.evaluate:
        runner.evaluate_model()
    else:
        runner.run()

