#!/usr/bin/env python
from rl.util import set_global_seeds
import argparse, os, json, shutil, gym
from model import *
from envs import NLimbRecorderEnv
from robots import get_robot, get_default_xml

def get_env_id(robot_type, terrain='flat'):
    if robot_type == 'hopper':
        id = 'Hopper-v1'
    elif robot_type == 'walker':
        id = 'Walker-v1'
    elif robot_type == 'ant':
        id = 'Ant-v1'
    else:
        assert False, "Unknown robot type"
    assert terrain in ['flat', 'slope']
    if terrain == 'slope':
        id = 'Incline' + id
    return 'NLimb' + id

def get_init_params(args):
    if args.init_with_default:
        xmlfile = get_default_xml(args.robot)
    elif args.init_with_xml is not None:
        xmlfile = args.init_with_xml
    else:
        return None
    robot = get_robot(args.robot)(xmlfile)
    params = robot.params()
    lmin, lmax = robot.get_param_limits()
    normed_params = 2 * (np.array(params) - np.array(lmin)) / (np.array(lmax) - np.array(lmin)) - 1.0
    return normed_params

def make_env_fn(args):
    env_id = get_env_id(args.robot, args.terrain)
    def env_fn(rank):
        default_xml = get_default_xml(args.robot)
        xmlfile = os.path.join(args.logdir, 'logs', 'robot.xml.{}'.format(rank))
        shutil.copyfile(default_xml, xmlfile)
        set_global_seeds(args.seed + rank)
        env = gym.make(env_id)
        return NLimbRecorderEnv(env, xmlfile, args.robot)
    return env_fn

def make_model_fn(args):
    hiddens = [args.nunits] * args.nlayers
    mean_init = get_init_params(args)
    def model_fn(env):
        obs = layers.Placeholder(tf.float32, env.observation_space.shape, 'obs')
        obs = RunningObsNorm('norm', obs, param_size=len(env.params))
        net = Net('net_pi', obs, hiddens=hiddens)
        actor = Policy('pi', net, ac_space=env.action_space)
        net = Net('net_vf', obs, hiddens=hiddens)
        critic = ValueFunction('vf', net)
        robot = layers.Placeholder(tf.float32, [len(env.params) + 1], 'robot_ph')
        sampler = RobotSampler('sampler', robot, len(env.params), args.ncomponents, mean_init, args.std_init)
        return Model('m', actor, critic, sampler)
    return model_fn

def main(args):
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, 'hyps.json'), 'w') as f:
        json.dump(args.__dict__, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Init experiments.')
    parser.add_argument('logdir', type=str, help='log directory')
    parser.add_argument('-n', '--nenv', type=int, default=8, help='# of threads to create')
    parser.add_argument('-t', '--maxtimesteps', type=int, default=int(1e9), help='max timesteps')
    parser.add_argument('--robot', type=str, default='hopper', help='robot xml to use. Options are [hopper, walker, ant, humanoid]')
    parser.add_argument('--terrain', default='flat', type=str, help='[flat, slope]')
    parser.add_argument('--seed',type=int, default=0, help='random seed')
    parser.add_argument('--gamma',type=float, default=.99, help='gamma')
    parser.add_argument('--lmbda',type=float, default=.95, help='lambda')
    parser.add_argument('--entcoeff',type=float, default=0.0, help='entropy loss coefficient')
    parser.add_argument('--batchsize',type=int, default=64, help='batchsize')
    parser.add_argument('--lr',type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--robot_lr',type=float, default=1e-3, help='robot initial learning rate')
    parser.add_argument('--momentum',type=float, default=.95, help='momentum')
    parser.add_argument('--robot_momentum',type=float, default=.9, help='robot update momentum')
    parser.add_argument('--epochs',type=int, default=4, help='epochs per PPO iteration')
    parser.add_argument('--ppo_clip_param',type=float, default=0.2, help='PPO clip param')
    parser.add_argument('--grad_clip_norm',type=float, default=0.5, help='grad clip norm')
    parser.add_argument('--rollout_length',type=int, default=1024, help='timesteps per batch')
    parser.add_argument('--steps_before_robot_update',type=int, default=1e8, help='start updating the robot at this timestep.')
    parser.add_argument('--steps_after_robot_update',type=int, default=1e8, help='finetune the policy for this many timesteps after freezing the robot distribution.')
    parser.add_argument('--fixed_robot', default=False, action='store_true', help='run roboschool env with a fixed robot')
    parser.add_argument('--init_with_default', default=False, action='store_true', help='initialize robot params with OpenAI params')
    parser.add_argument('--init_with_xml', type=str, default=None, help='initialize robot params from a given xml file')
    parser.add_argument('--ncomponents',type=int, default=8, help='number of components in the robot GMM')
    parser.add_argument('--std_init',type=float, default=.577, help='initial std of comonents in robot distribution')
    parser.add_argument('--nunits',type=int, default=128, help='hidden layer size in policy')
    parser.add_argument('--nlayers',type=int, default=3, help='number of hidden layers')
    parser.add_argument('--chop_freq', default=int(1e8), type=int, help='frequency with which to halve the number of components.')

    main(parser.parse_args())
