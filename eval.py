from deeplearning import tf_util as U
from init import make_env_fn, make_model_fn
from collections import namedtuple
import os, argparse, json
import numpy as np

def eval_robot(args, env, pi):
    rewards = []
    lengths = []
    for j in range(args.nepisodes):
        rewards.append(0)
        lengths.append(0)
        done = False
        ob = env.reset()
        while not done:
            ac = pi.actor.mode(ob[None])[0]
            ob, rew, done, _ = env.step(ac)
            rewards[-1] += rew
            lengths[-1] += 1
    return np.mean(lengths), np.mean(rewards)

def main(args):
    U.reset()

    with open(os.path.join(args.logdir, 'hyps.json'), 'r') as f:
        hyps = json.load(f)
    train_args = namedtuple('Args', hyps.keys())(**hyps)
    env_fn = make_env_fn(train_args)
    model_fn = make_model_fn(train_args)

    env = env_fn(0)
    model = model_fn(env)
    model.build('model', 1, 1)
    model.sampler.build('model', 1, 1)

    sess = U.make_session()
    sess.__enter__()
    U.initialize()
    t = U.Experiment(args.logdir).load(args.ckpt)

    ls = []
    rs = []
    for i in range(args.samples):
        env.update_robot(model.sampler.sample(args.stochastic)[0])
        l,r = eval_robot(args, env, model)
        ls.append(l)
        rs.append(r)
        if not args.stochastic:
            break


    os.makedirs(os.path.join(args.logdir, 'eval'), exist_ok=True)
    with open(os.path.join(args.logdir, 'eval', '{}.json'.format(t)), 'w') as f:
        json.dump({'l':ls, 'r':rs}, f)
    sess.__exit__(None, None, None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a Checkpoint')
    parser.add_argument('logdir', type=str, help='log directory')
    parser.add_argument('-t', '--ckpt', type=int, default=None, help='which checkpoint file to use')
    parser.add_argument('-n', '--nepisodes', type=int, default=1, help='n episodes to show')
    parser.add_argument('-s', '--samples', type=int, default=1, help='# of robots to sample')
    parser.add_argument('--stochastic', type=bool, default=True, help='If false, eval the mode of the robot distribution')
    main(parser.parse_args())
