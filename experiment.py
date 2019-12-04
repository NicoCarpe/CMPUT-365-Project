import agent as mc
import matplotlib.pyplot as plt
from rl_glue import RLGlue

# Here we provide you with the true state value and state distribution
true_state_val = np.load('data/true_V.npy')
state_distribution = np.load('data/state_distribution.npy')

def calc_RMSVE(learned_state_val):
    assert(len(true_state_val) == len(learned_state_val) == len(state_distribution))
    MSVE = np.sum(np.multiply(state_distribution, np.square(true_state_val - learned_state_val)))
    RMSVE = np.sqrt(MSVE)
    return RMSVE


def test_TileCoder():
    tests = [[-1.0, 0.01], [0.1, -0.01], [0.2, -0.05], [-1.0, 0.011], [0.2, -0.05]]
    mctc = mc.MountainCarTileCoder(iht_size=1024, num_tilings=8, num_tiles=8)
    t = []
    for test in tests:
        position, velocity = test
        tiles = mctc.get_tiles(position=position, velocity=velocity)
        t.append(tiles)

    print("Your results:")
    for tiles in t:
        print(tiles)

    print()
    print("Expected results:")
    expected = """[0 1 2 3 4 5 6 7]
    [ 8  9 10 11 12 13 14 15]
    [16 17 18 19 20 21 22 23]
    [ 0 24  2  3  4  5  6  7]
    [16 17 18 19 20 21 22 23]
    """
    print(expected)


def run_experiment(environment, agent, environment_parameters, agent_parameters, experiment_parameters):
    rl_glue = RLGlue(environment, agent)

    for 
