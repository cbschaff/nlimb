#!/usr/bin/env python
import argparse, os, json, time
from collections import namedtuple
from algorithm import Algorithm
from init import make_env_fn, make_model_fn
import envs

def main(run_args):
    with open(os.path.join(run_args.logdir, 'hyps.json'), 'r') as f:
        hyps = json.load(f)
    args = namedtuple('Args', hyps.keys())(**hyps)
    env_fn = make_env_fn(args)
    model_fn = make_model_fn(args)

    time.sleep(1)
    alg = Algorithm(run_args.logdir, env_fn, model_fn, args.nenv,
                    args.rollout_length,
                    args.batchsize,
                    epochs_per_iter=args.epochs,
                    lr=args.lr,
                    momentum=args.momentum,
                    ent_coef=args.entcoeff,
                    gamma=args.gamma,
                    lambda_=args.lmbda,
                    clip_norm=args.grad_clip_norm,
                    clip_param=args.ppo_clip_param,
                    robot_lr=args.robot_lr,
                    robot_momentum=args.robot_momentum,
                    fixed_robot=args.fixed_robot,
                    steps_before_robot_update=args.steps_before_robot_update,
                    steps_after_robot_update=args.steps_after_robot_update,
                    chop_freq=args.chop_freq,
                    tmax=args.maxtimesteps)

    alg.train(args.maxtimesteps, run_args.maxseconds, run_args.save_freq)
    alg.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='continue experiments.')
    parser.add_argument('logdir', type=str, help='log directory')
    parser.add_argument('-s', '--maxseconds', type=int, default=None, help='max seconds')
    parser.add_argument('--save_freq', type=int, default=int(5e6), help='timesteps per save')
    main(parser.parse_args())
