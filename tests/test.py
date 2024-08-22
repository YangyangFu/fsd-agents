# import and run all tests
import pytest 

agent_tests = [
    "tests/agents/InterFuser/test_density_map.py"
]

dataset_tests = [
    "tests/datasets/test_carla.py"
]

pytest.main(agent_tests + dataset_tests)