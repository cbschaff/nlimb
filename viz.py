from deeplearning import tf_util as U
from init import make_env_fn, make_model_fn
from collections import namedtuple
import os, argparse, json
from imageio import imwrite
import subprocess as sp


def main(args):
    U.reset()

    with open(os.path.join(args.logdir, 'hyps.json'), 'r') as f:
        hyps = json.load(f)
    train_args = namedtuple('Args', hyps.keys())(**hyps)
    env_fn = make_env_fn(train_args)
    model_fn = make_model_fn(train_args)

    env = env_fn(0)
    env.unwrapped.set_render_ground(True)
    model = model_fn(env)
    model.build('model', 1, 1)
    model.sampler.build('model', 1, 1)

    sess = U.make_session()
    sess.__enter__()
    U.initialize()
    t = U.Experiment(args.logdir).load(args.ckpt)

    # load mode of design distribution
    env.update_robot(model.sampler.sample(stochastic=False)[0])


    i = 0
    if not args.save:
        env.reset()
        env.render()
    else:
        outdir = './video_tmp'
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(os.path.join(args.logdir, 'video'), exist_ok=True)
    for j in range(args.nepisodes):
        done = False
        ob = env.reset()
        while not done:
            if args.save:
                rgb = env.render('rgb_array')
            ac = model.actor.mode(ob[None])[0]
            ob, rew, done, _ = env.step(ac)
            if args.save:
                imwrite(os.path.join(outdir, '{:05d}.png'.format(i)), rgb)
            i += 1

    if args.save:
        outfile = str(t) + '.mp4'
        sp.call(['ffmpeg', '-r', '60', '-f', 'image2', '-i', os.path.join(outdir, '%05d.png'), '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', os.path.join(outdir, 'out.mp4')])
        sp.call(['mv', os.path.join(outdir, 'out.mp4'), os.path.join(args.logdir, 'video', outfile)])
        sp.call(['rm', '-rf', outdir])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make videos')
    parser.add_argument('logdir', type=str, help='log directory')
    parser.add_argument('-t', '--ckpt', type=int, default=None, help='which checkpoint file to use')
    parser.add_argument('-n', '--nepisodes', type=int, default=1, help='n episodes to show')
    parser.add_argument('--save', default=False, action='store_true', help='save videos')

    main(parser.parse_args())
