import pytest
from fsd.registry import CONTROLLERS


# test with different kps
@pytest.mark.parametrize(
    "kp", [1.0, 2.0, 3.0]
)
def test_pid_controller(kp):
    pid_cfg = dict(
        type='PID',
        kp=kp,
        ki=1.0,
        kd=0.0,
        dt=1.0,
        ymin=-10.0,
        ymax=10.0
    )
    
    pid = CONTROLLERS.build(pid_cfg)
    
    assert pid.kp == kp
    assert pid.ki == 1.0
    assert pid.kd == 0.0
    assert pid.dt == 1.0
    assert pid.total_error_prev == 0
    
    # test run_step
    target = 10.0
    measurement = 6.0
    control = pid.run_step(target, measurement)
    assert pid.total_error_prev == target - measurement
    
    if kp == 1.0:
        assert control - 8.0 < 1e-6
    else:
        assert control - 10 < 1e-6



