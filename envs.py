"""
This file defines the environments used in our experiments.

We subclass the roboschool environments to allow for:
 1) abribrary paths to robot xml files
 2) inclined terrain
 2) toggle to disable loading .obj files during training

We define wrappers for these environments which detect
parameters of the controlled robot and allow those parameters
to be modified.
"""
import roboschool
from roboschool.scene_stadium import SinglePlayerStadiumScene
from roboschool.scene_abstract import Scene, cpp_household
from roboschool.gym_mujoco_walkers import *
from gym.envs.registration import register

class SinglePlayerScene(SinglePlayerStadiumScene):
    def __init__(self,*args, render=False, inclined=False, **kwargs):
        super().__init__(*args,**kwargs)
        self.render = render
        self.inclined = inclined

    def episode_restart(self):
        Scene.episode_restart(self)
        stadium_pose = cpp_household.Pose()
        if self.zero_at_running_strip_start_line:
            stadium_pose.set_xyz(27, 21, 0)
        if self.render:
            if self.inclined:
                self.hfield = self.cpp_world.load_thingy('assets/incline_grass.obj', stadium_pose, 1.0, 0, 0xFFFFFF, True)
            else:
                self.stadium = self.cpp_world.load_thingy(
                os.path.join(os.path.dirname(roboschool.__file__), "models_outdoor/stadium/stadium1.obj"),
                stadium_pose, 1.0, 0, 0xFFFFFF, True)

        if self.inclined:
            self.ground_plane_mjcf = self.cpp_world.load_mjcf( "assets/incline_plane.mjcf")
        else:
            self.ground_plane_mjcf = self.cpp_world.load_mjcf( "assets/level_plane.mjcf")


def create_env(BaseClass):
    class Env(BaseClass):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.render_ground = False
            self.inclined_terrain = False

        def create_single_player_scene(self):
            return SinglePlayerScene(gravity=9.8, timestep=0.0165/4, frame_skip=4, render=self.render_ground, inclined=self.inclined_terrain)

        def set_render_ground(self, render=False):
            self.render_ground = render
            if self.scene is not None:
                self.scene.render = render

        def _reset(self):
            """
            Load from abribrary xml files.
            """
            if self.scene is None:
                self.scene = self.create_single_player_scene()
            if not self.scene.multiplayer:
                self.scene.episode_restart()
            # Only this line has been changed. The rest is copied from Roboschool.
            self.mjcf = self.scene.cpp_world.load_mjcf(self.model_xml)
            self.ordered_joints = []
            self.jdict = {}
            self.parts = {}
            self.frame = 0
            self.done = 0
            self.reward = 0
            dump = 0
            for r in self.mjcf:
                if dump: print("ROBOT '%s'" % r.root_part.name)
                if r.root_part.name==self.robot_name:
                    self.cpp_robot = r
                    self.robot_body = r.root_part
                for part in r.parts:
                    if dump: print("\tPART '%s'" % part.name)
                    self.parts[part.name] = part
                    if part.name==self.robot_name:
                        self.cpp_robot = r
                        self.robot_body = part
                for j in r.joints:
                    if dump: print("\tALL JOINTS '%s' limits = %+0.2f..%+0.2f effort=%0.3f speed=%0.3f" % ((j.name,) + j.limits()) )
                    if j.name[:6]=="ignore":
                        j.set_motor_torque(0)
                        continue
                    j.power_coef = 100.0
                    self.ordered_joints.append(j)
                    self.jdict[j.name] = j
            assert(self.cpp_robot)
            self.robot_specific_reset()
            for r in self.mjcf:
                r.query_position()
            s = self.calc_state()    # optimization: calc_state() can calculate something in self.* for calc_potential() to use
            self.potential = self.calc_potential()
            self.camera = self.scene.cpp_world.new_camera_free_float(self.VIDEO_W, self.VIDEO_H, "video_camera")
            return s

        def calc_state(self):
            """
            Zero out foot contact booleans.
            """
            state = super().calc_state()
            state[-len(self.foot_list):] = 0.
            return state

    return Env


def create_inline_env(BaseClass):
    class Env(BaseClass):
        def __init__(self, *args, slope=5., **kwargs):
            super().__init__(*args, **kwargs)
            self.inclined_terrain = True
            self.slope = slope # 5 degrees
            shape = self.observation_space.shape[0] + 1
            high = np.inf*np.ones([shape])
            self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        def robot_specific_reset(self):
            """
            Angle the feet of the Hopper and Walker with the slope
            to avoid starting in collision with the ground.
            """
            if isinstance(self, RoboschoolAnt):
                super().robot_specific_reset()
                return

            for j in self.ordered_joints:
                if j.name.rsplit('_',1)[0] in self.foot_list:
                    j.reset_current_position(np.pi * self.slope / 180. + self.np_random.uniform( low=-0.1, high=0.1 ), 0)
                else:
                    j.reset_current_position(self.np_random.uniform( low=-0.1, high=0.1 ), 0)

            self.feet = [self.parts[f] for f in self.foot_list]
            self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)
            self.scene.actor_introduce(self)
            self.initial_z = None

        def alive_bonus(self, z, pitch):
            """
            Adjust Roboschool height thresholds to account for changes
            in elevation.
            """
            bonus = 0.5 if isinstance(self, RoboschoolAnt) else 1.0
            thresh = 0.26 if isinstance(self, RoboschoolAnt) else 0.8
            thresh += self.body_xyz[0] * np.tan(np.pi * self.slope / 180.)
            return bonus if z > thresh and abs(pitch) < 1.0 else -1

        def _reset(self):
            ob = super()._reset()
            return np.concatenate([ob, [self.slope]])

        def _step(self, a):
            ob, r, done, info = super()._step(a)
            return np.concatenate([ob, [self.slope]]), r, done, info

    return Env



HopperEnv = create_env(RoboschoolHopper)
WalkerEnv = create_env(RoboschoolWalker2d)
AntEnv    = create_env(RoboschoolAnt)
InclineHopperEnv = create_inline_env(HopperEnv)
InclineWalkerEnv = create_inline_env(WalkerEnv)
InclineAntEnv    = create_inline_env(AntEnv)

register(
    id='NLimbHopper-v1',
    entry_point='envs:HopperEnv',
    max_episode_steps=1000,
    reward_threshold=2500.0
    )

register(
    id='NLimbWalker-v1',
    entry_point='envs:WalkerEnv',
    max_episode_steps=1000,
    reward_threshold=2500.0
    )

register(
    id='NLimbAnt-v1',
    entry_point='envs:AntEnv',
    max_episode_steps=1000,
    reward_threshold=2500.0
    )

register(
    id='NLimbInclineHopper-v1',
    entry_point='envs:InclineHopperEnv',
    max_episode_steps=1000,
    reward_threshold=2500.0
    )

register(
    id='NLimbInclineWalker-v1',
    entry_point='envs:InclineWalkerEnv',
    max_episode_steps=1000,
    reward_threshold=2500.0
    )

register(
    id='NLimbInclineAnt-v1',
    entry_point='envs:InclineAntEnv',
    max_episode_steps=1000,
    reward_threshold=2500.0
    )



from gym import Wrapper
from gym.spaces import Box
import numpy as np
from robots import Hopper, Walker, Ant
from collections import deque
from robots import get_robot

class NLimbEnv(Wrapper):
    """
    Environment wrapper for detecting and modifying robot parameters.
    """
    def __init__(self, env, model_xml, robot_type):
        super().__init__(env)
        self.robot = get_robot(robot_type)(model_xml)
        self.unwrapped.model_xml = model_xml

        limits = self.robot.get_param_limits()
        self.lim_min = np.array(limits[0])
        self.lim_max = np.array(limits[1])
        self.params = self._norm_params(self.robot.get_params())
        self.param_names = self.robot.get_param_names()

        assert len(self.observation_space.shape) == 1, "Ob space must be 1 dimensional"
        shape = self.observation_space.shape[0] + len(self.params)
        high = self.observation_space.high[0] * np.ones(shape)
        low = self.observation_space.low[0] * np.ones(shape)
        self.observation_space = Box(low, high, dtype=np.float32)

    def _norm_params(self, params):
        return (2 * (params - self.lim_min) / (self.lim_max - self.lim_min) - 1.0)

    def _unnorm_params(self, params):
        return (params + 1) / 2 * (self.lim_max - self.lim_min) + self.lim_min

    def reset(self):
        ob = self.env.reset()
        ob = np.concatenate([ob, self.params])
        return ob

    def step(self, a):
        ob, reward, done, info = self.env.step(np.clip(a,-1,1))
        ob = np.concatenate([ob, self.params])
        return ob, reward, done, info

    def update_robot(self, params):
        params = np.clip(params[:-1], -1., 1.)
        assert len(params) == len(self.params)
        self.robot.update(self._unnorm_params(params))
        self.params = params


class NLimbRecorderEnv(NLimbEnv):
    """
    Extends NLimbEnv with code to save the performance of
    recent robots.
    """
    def __init__(self, *args, buffer_size=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_buffer = deque(maxlen=buffer_size)
        self.reward_buffer = deque(maxlen=buffer_size)
        self.ep_rews = []
        self.rews = []
        self.unclipped_params = self.params

    def reset(self):
        if len(self.rews) > 0:
            self.ep_rews.append(sum(self.rews))
            self.rews = []
        return super().reset()

    def step(self, a):
        ob, r, done, info = super().step(a)
        self.rews.append(r)
        return ob, r, done, info

    def update_buffer(self):
        if len(self.ep_rews) > 0:
            self.param_buffer.append(self.unclipped_params)
            self.reward_buffer.append(np.mean(self.ep_rews))
            self.rews = []
            self.ep_rews = []

    def update_robot(self, params):
        self.update_buffer()
        self.unclipped_params = params
        super().update_robot(params)
