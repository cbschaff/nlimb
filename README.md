# Jointly Learning to Construct and Control Agents using Reinforcement Learning

This code accompanies the paper *Jointly Learning to Construct and Control Agents using Reinforcement Learning*.

Arxiv Link: [https://arxiv.org/abs/1801.01432](https://arxiv.org/abs/1801.01432)

## Dependencies
This code is build on top of OpenAI's Roboschool. 

To install Roboschool follow the instructions at [https://github.com/openai/roboschool](https://github.com/openai/roboschool)

All other dependencies can be installed by calling:
`pip install -r requirements.txt`


## Experiments

Experiments are run by calling two scripts: init.py and train.py. 

### init.py
init.py initializes experiment directories and saves hyperparameters. See init.py for a full list of hyperparameters.

`python init.py logdir --hyperparameter value ...`

### train.py
train.py starts or continues an experiment from the latest checkpoint.

`python train.py logdir -s maxseconds --save_freq timesteps_per_checkpoint`

### Replicating the experiments in the paper
Pretrained models from our experiments are included with the release of this code. 

Additionally, for each of the robot morphologies (Hopper, Walker, Ant) and each terrian type (level, inclined), our experiments can be replicated using the following commands

#### Level Terrain
```
python init.py logs/hopper_level --robot hopper --terrain flat
python train.py logs/hopper_level
```

```
python init.py logs/walker_level --robot walker --terrain flat
python train.py logs/walker_level
```

```
python init.py logs/ant_level --robot ant --terrain flat -t 1500000000 --steps_before_robot_update 200000000 --chop_freq 200000000
python train.py logs/ant_level
```

#### Inclined Terrain
```
python init.py logs/hopper_incline --robot hopper --terrain slope
python train.py logs/hopper_incline
```

```
python init.py logs/walker_incline --robot walker --terrain slope
python train.py logs/walker_incline
```

```
python init.py logs/ant_incline --robot ant --terrain slope -t 1500000000 --steps_before_robot_update 200000000 --chop_freq 200000000
python train.py logs/ant_incline
```


## Evaluation and Visualization

The training script creates Tensorboard summaries in `logdir/summaries`. These can be visualized by launching Tensorboard:

`tensorboard --logdir logdir/summaries`

Additionally, two scripts are provided to evaluate and visualize checkpoints: eval.py and viz.py.

eval.py samples robot designs from the design distribution and computes episode statistics averaged over n episodes. Results are saved in `logdir/eval`

`python eval.py logdir -t checkpoint -n nepisodes -s nsamples`

viz.py renders or creates videos of the mode of the design distribution. Videos are saved in `logdir/videos`.

`python viz.py logdir -t checkpoint -n nepisodes --save`
