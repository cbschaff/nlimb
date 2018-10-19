import numpy as np
from xml_parser import MuJoCoXmlRobot

def get_default_xml(robot_type):
    if robot_type == 'hopper':
        return 'assets/hopper.xml'
    elif robot_type == 'walker':
        return 'assets/walker2d.xml'
    elif robot_type == 'ant':
        return 'assets/ant.xml'
    else:
        assert False, "Unknown robot type."

def get_robot(robot_type):
    if 'hopper' == robot_type:
        return Hopper
    if 'walker' == robot_type:
        return Walker
    if 'ant' == robot_type:
        return Ant
    assert False, "unkown robot"


# Add parameter constraints for the different robots
class Hopper(MuJoCoXmlRobot):
    def __init__(self, model_xml):
        super().__init__(model_xml)
        self.default_robot = MuJoCoXmlRobot(get_default_xml('hopper'))
        self.default_params = np.array(self.default_robot.get_params())
        self.lower_limits = 0.5 * self.default_params
        self.upper_limits = 1.5 * self.default_params

    def get_param_limits(self):
        return self.lower_limits, self.upper_limits

class Ant(MuJoCoXmlRobot):
    def __init__(self, model_xml):
        super().__init__(model_xml)
        self.default_robot = MuJoCoXmlRobot(get_default_xml('ant'))
        self.default_params = np.array(self.default_robot.get_params())
        self.lower_limits = 0.5 * self.default_params
        self.upper_limits = 1.5 * self.default_params

    def get_param_limits(self):
        return self.lower_limits, self.upper_limits

class Walker(MuJoCoXmlRobot):
    def __init__(self, model_xml):
        super().__init__(model_xml)
        self.default_robot = MuJoCoXmlRobot(get_default_xml('walker'))
        self.default_params = np.array(self.default_robot.get_params()[:8])
        self.lower_limits = 0.5 * self.default_params
        self.upper_limits = 1.5 * self.default_params

    def get_params(self):
        return super().get_params()[:8]

    def get_param_limits(self):
        return self.lower_limits, self.upper_limits

    def get_param_names(self):
        return super().get_param_names()[:8]

    def update(self, params, xml_file=None):
        params = np.array(params)
        params = np.concatenate([params, params[2:]])
        super().update(params, xml_file)
