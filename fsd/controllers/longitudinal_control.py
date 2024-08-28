from .base import PID
from fsd.registry import CONTROLLERS

@CONTROLLERS.register_module()
class PIDLongitudinal(PID):
    """
    PIDLongitudinal implements longitudinal control using a PID.
    """

    def __init__(self, kp=1.0, ki=0.0, kd=0.0, dt=0.03, ymin=-1.0, ymax=1.0):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        super(PIDLongitudinal, self).__init__(kp=kp,
                                              ki=ki,
                                              kd=kd,
                                              dt=dt,
                                              ymin=ymin,
                                              ymax=ymax)

    def run_step(self, target_speed, measured_speed):
        """
        Execute one step of longitudinal control to reach a certain target speed.

            :param target: desired speed
            :param measurement: current speed
            :return: throttle control in the range [-1, 1] where:
            -1 maximum brake
            +1 maximum throttle
        """
        return super(PIDLongitudinal, self).run_step(target_speed, measured_speed)