import numpy as np
import xml.etree.ElementTree as ET

class Geom(object):
    def __init__(self, geom):
        self.xml = geom
        self.params = []

    def get_params(self):
        return self.params.copy()

    def set_params(self, new_params):
        self.params = new_params

    def update_point(self, p, new_params):
        pass

    def update_xml(self):
        pass

    def update(self, new_params):
        self.set_params(new_params)
        self.update_xml()

    def get_smallest_z(self):
        pass

    def get_param_limits(self):
        pass

    def get_param_names(self):
        pass

    def get_volume(self):
        pass

class Sphere(Geom):
    min_radius = .05
    max_radius = .4

    def __init__(self, geom):
        self.xml = geom
        self.params = [float(self.xml.get('size'))] # radius
        self.center = np.array([float(x) for x in self.xml.get('pos').split()])

    def update_point(self, p, new_params):
        return ((p - self.center) * new_params[0] / self.params[0]) + self.center

    def update_xml(self):
        self.xml.set('size', str(self.params[0]))

    def get_smallest_z(self):
        return self.center[2] - self.params[0]

    def get_param_limits(self):
        return [[self.min_radius], [self.max_radius]]

    def get_param_names(self):
        return ['radius']

    def get_volume(self):
        return 4./3. * np.pi * self.params[0] ** 3

class Capsule(Geom):
    min_length = 0.175
    max_length = 0.8
    min_radius = 0.035
    max_radius = 0.085

    def __init__(self, geom):
        self.xml = geom
        fromto = [float(x) for x in self.xml.get('fromto').split()]
        self.p1 = np.array(fromto[:3])
        self.p2 = np.array(fromto[3:])
        length = np.sqrt(np.sum((self.p2 - self.p1) ** 2))
        radius = float(self.xml.get('size'))
        self.params = [length, radius]
        self.axis = (self.p2 - self.p1) / length

    def update_point(self, p, new_params):
        lfac = p.dot(self.axis) * self.axis
        rfac = p - lfac
        return p + lfac * (-1.0 + new_params[0] / self.params[0])# + rfac * (new_params[1] / self.params[1])

    def update_xml(self):
        self.xml.set('fromto', ' '.join([str(x) for x in np.concatenate([self.p1, self.p2])]))
        self.xml.set('size', str(self.params[1])) # radius

    def set_params(self, new_params):
        p1 = self.update_point(self.p1, new_params)
        p2 = self.update_point(self.p2, new_params)
        # update only after computing p1, p2
        self.p1 = p1
        self.p2 = p2
        super().set_params(new_params)

    def get_smallest_z(self):
        return min(self.p1[2], self.p2[2]) - self.params[1]

    def get_param_limits(self):
        return [[self.min_length, self.min_radius], [self.max_length, self.max_radius]]

    def get_param_names(self):
        return ['length','radius']

    def get_volume(self):
        return 4./3. * np.pi * self.params[1]**3 + self.params[0] * np.pi * self.params[1]**2

class Body:
    geoms = {'sphere': Sphere, 'capsule': Capsule} # dictionary of legal geometry types

    def __init__(self, body, worldbody=False):
        self.xml = body
        self.worldbody = worldbody

        geom_xml = body.find('geom') # assume only one geometry per body
        self.geom = self.geoms[geom_xml.get('type')](geom_xml)
        self.joints = [j for j in body.findall('joint') if 'ignore' not in j.get('name')]
        self.parts = [Body(b) for b in body.findall('body')]
        pos = [b.get('pos') for b in body.findall('body')]
        self.part_positions = [np.array([float(x) for x in p.split()]) for p in pos]
        pos = [j.get('pos') for j in self.joints]
        self.joint_positions = [np.array([float(x) for x in p.split()]) for p in pos]
        self.n = len(self.geom.get_params())
        self.n_all_params = len(self.get_params())

        self.zmin = float(self.xml.get("pos").split()[2]) - self.get_height()

    def get_height(self):
        max_height = -self.geom.get_smallest_z()
        for body, pos in zip(self.parts, self.part_positions):
            max_height = max(max_height, body.get_height() - pos[2])
        return max_height

    def update_initial_position(self):
        pos = self.xml.get("pos").split()
        pos[2] = str(self.get_height() + self.zmin)
        self.xml.set("pos", ' '.join(pos))

    def update_xml(self):
        for body, pos in zip(self.parts, self.part_positions):
            body.xml.set('pos', ' '.join([str(x) for x in pos]))

        for joint, pos in zip(self.joints, self.joint_positions):
            joint.set('pos', ' '.join([str(x) for x in pos]))

    def set_body_positions(self, new_params):
        for i, pos in enumerate(self.part_positions):
            self.part_positions[i] = self.geom.update_point(pos, new_params)
        for i, pos in enumerate(self.joint_positions):
            self.joint_positions[i] = self.geom.update_point(pos, new_params)

    def update(self, new_params):
        self.set_body_positions(new_params)
        self.geom.update(new_params)
        self.update_xml()

    def get_params(self):
        params = self.geom.get_params()
        for body in self.parts:
            params += body.get_params()
        return params

    def get_param_limits(self):
        limits = self.geom.get_param_limits()
        for body in self.parts:
            body_limits = body.get_param_limits()
            limits[0] += body_limits[0]
            limits[1] += body_limits[1]
        return limits

    def get_param_names(self):
        name = self.xml.get('name')
        param_names = [name + '-' + p for p in self.geom.get_param_names()]
        for body in self.parts:
            param_names += body.get_param_names()
        return param_names

    def update_params(self, new_params):
        if self.worldbody: assert len(new_params) == self.n_all_params, "Wrong number of parameters"
        self.update(new_params[:self.n])
        remaining_params = new_params[self.n:]
        for body in self.parts:
            remaining_params = body.update_params(remaining_params)
        if self.worldbody:
            self.update_initial_position()
        else:
            return remaining_params

    def get_body_names(self):
        names = [self.xml.get('name')]
        for body in self.parts:
            names += body.get_names()
        return names

    def get_joints(self):
        joints = {}
        for body,pos in zip(self.parts, self.part_positions):
            for j in body.joints:
                joints[j.get('name')] = (self.xml.get('name'), body.xml.get('name'), self.geom, body.geom, pos)
            joints.update(body.get_joints())
        return joints

    def get_volumes(self):
        volumes = {}
        if len(self.joints) > 0:
            for j in self.joints:
                v1 = self.geom.get_volume()
                v2 = sum([b.geom.get_volume() for b in self.parts])
                volumes[j.get('name')] = np.array((v1, v2))
        for body in self.parts:
            volumes.update(body.get_volumes())
        return volumes


class MuJoCoXmlRobot:
    def __init__(self, model_xml):
        self.model_xml = model_xml
        self.tree = ET.parse(self.model_xml)
        worldbody = self.tree.getroot().find('worldbody')
        self.body = Body(worldbody.find('body'), worldbody=True)

    def get_params(self):
        return self.body.get_params()

    def get_param_limits(self):
        return self.body.get_param_limits()

    def get_param_names(self):
        return self.body.get_param_names()

    def get_height(self):
        return self.body.get_height()

    def get_joints(self):
        return self.body.get_joints()

    def get_volumes(self):
        return self.body.get_volumes()

    def update(self, params, xml_file=None):
        if xml_file is None:
            xml_file = self.model_xml
        self.body.update_params(list(params))
        self.tree.write(xml_file)

if __name__ == '__main__':
    robot = MuJoCoXmlRobot('mujoco_assets/hopper.xml')
    params = list(1.0 * np.array(robot.get_params()))
    robot.update(params, 'mujoco_assets/hopper_test.xml')
    assert robot.get_params() == params
    #assert robot.get_height() == 1.31
    print(robot.get_param_limits())
    print(robot.get_param_names())

    robot = MuJoCoXmlRobot('mujoco_assets/walker2d.xml')
    params = [.4,.04,.5,.05,.55,.055,.6,.06,.5,.05,.55,.055,.6,.06]
    robot.update(params, 'mujoco_assets/walker2d_test.xml')
    assert robot.get_params() == params
    assert robot.get_height() == 1.31
    print(robot.get_param_limits())
    print(robot.get_param_names())

    robot = MuJoCoXmlRobot('mujoco_assets/ant.xml')
    params = [.2, .2,.06,.2,.06,.4,.06, .2,.06,.2,.06,.4,.06, .2,.06,.2,.06,.4,.06, .2,.06,.2,.06,.4,.06]
    robot.update(params, 'mujoco_assets/ant_test.xml')
    assert robot.get_params() == params
    assert robot.get_height() == .2
    print(robot.get_param_limits())
    print(robot.get_param_names())

    robot = MuJoCoXmlRobot('mujoco_assets/humanoid.xml')
    params = list(.8 * np.array(robot.get_params()))
    robot.update(params, 'mujoco_assets/humanoid_test.xml')
    assert robot.get_params() == params
    print(robot.get_height())
    #assert robot.get_height() == .6085
    print(robot.get_param_limits())
    print(robot.get_param_names())

    import gym, roboschool
    env = gym.make("RoboschoolHopper-v1")
    env.unwrapped.model_xml = 'mujoco_assets/hopper_test.xml'
    env.reset()
    #env.render()
    import os
    from scipy.misc import imsave
    import subprocess as sp
    outdir = 'xml_vid'
    os.makedirs(outdir, exist_ok=True)
    i = 0
    for _ in range(10):
        env.reset()
        for _ in range(100):
            env.step(env.action_space.sample())
            rgb = env.render('rgb_array')
            imsave(os.path.join(outdir, '{:05d}.png'.format(i)), rgb)
            i+=1
    sp.call(['ffmpeg', '-r', '60', '-f', 'image2', '-i', os.path.join(outdir, '%05d.png'), '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', os.path.join(outdir, 'out.mp4')])
    env.close()
