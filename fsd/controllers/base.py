import numpy as np 
from collections import deque 
from fsd.registry import CONTROLLERS 

class BaseController(object):
    """
    Base class for controllers.
    """

    def __init__(self):
        """
        Constructor method.
        """
        pass

    def run_step(self, target, measurement):
        """
        Execute one step control to reach a given target.
        """
        raise NotImplementedError("Method run_step must be implemented in child class.")

@CONTROLLERS.register_module()
class PID(BaseController):
    """
    PID implements a basic PID controller.
    """

    def __init__(self, kp=1.0, ki=0.0, kd=0.0, dt=0.03, ymin=-1.0, ymax=1.0):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param kp: Proportional term
            :param kd: Differential term
            :param ki: Integral term
            :param dt: time differential in seconds
        """
        # initialize properties
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._dt = dt
        self._ymin = ymin 
        self._ymax = ymax
        
        # buffer to save error values for plotting and debugging
        self.error_buffer = deque(maxlen=10) 

        # for I
        self.total_error_prev = 0

    @property
    def kp(self):
        return self._kp
    @kp.setter
    def kp(self, value):
        self._kp = value
    
    @property
    def ki(self):
        return self._ki
    @ki.setter
    def ki(self, value):
        self._ki = value
        
    @property
    def kd(self):
        return self._kd
    @kd.setter
    def kd(self, value):
        self._kd = value
        
    @property
    def dt(self):
        return self._dt
    @dt.setter
    def dt(self, value):
        self._dt = value
           
    @property
    def ymin(self):
        return self._ymin
    @ymin.setter
    def ymin(self, value):
        self._ymin = value
    
    @property
    def ymax(self):
        return self._ymax
    @ymax.setter
    def ymax(self, value):
        self._ymax = value
    
    
    def run_step(self, target, measurement):
        """
        Execute one step control to reach a given target.
        """
        return self._pid_control(target, measurement)

    def _pid_control(self, target, measurement):
        """
        Calculate the PID control using the target and measurement.
        """

        error = target - measurement
        total_error = self.total_error_prev + error

        self.error_buffer.append(error)

        if len(self.error_buffer) >= 2:
            _de = (self.error_buffer[-1] - self.error_buffer[-2]) / self.dt
            _ie = total_error * self.dt
        else:
            _de = 0.0
            _ie = 0.0
        
        self.total_error_prev = total_error

        return np.clip((self.kp * error) + (self.kd * _de) + (self.ki * _ie), self.ymin, self.ymax)

    
