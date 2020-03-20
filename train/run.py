import argparse
import gym
import os
import sys
import pickle
import time
import datetime
import multiprocessing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from utils.args import *
from utils.replay_memory import Memory
from utils.torch import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy

try:
    path = os.path.join(assets_dir(), 'learned_models/{}_ppo.p'.format(args.env_name))
    models_file=open(path,'r')
    print("pre-trained models loaded.")
    print("model path: ", path)
except IOError:
    print("pre-trained models not found.")

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cpu')

"""environment"""
env = gym.make(args.env_name)
state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
# running_state = ZFilter((state_dim,), clip=5)

"""seeding"""
seed = int(time.time())
np.random.seed(seed)
torch.manual_seed(seed)
env.seed(seed)

"""define actor and critic"""
if path is None:
    if is_disc_action:
        policy_net=DiscretePolicy(state_dim, env.action_space.n)
    else:
        policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std)
    value_net=Value(state_dim)
else:
    print(path)
    policy_net, value_net, _ = pickle.load(open(path, "rb"))

policy_net.to(device)
value_net.to(device)

state = env.reset()

reward_episode = 0

for t in range(10000):
    state_var = tensor(state).unsqueeze(0)

    with torch.no_grad():
        action = policy_net.select_action(state_var)
    next_state, reward, done, _ = env.step(action[0].tolist())
    reward_episode += reward

    env.render()
    time.sleep(0.01)

    if done:
        print("reward:", reward_episode)
        break
    
    state = next_state