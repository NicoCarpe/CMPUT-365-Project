import ExpectedSarsaAgent as ag
import numpy as np
import MountainCarEnvironment as enviro
import matplotlib.pyplot as plt
from rl_glue import RLGlue
import time


def run_experiment(num_runs, num_episodes):
    all_steps = []
    agent = ag.ExpectedSarsaAgent
    env = enviro.MountainEnvironment

    for run in range(num_runs):
        if run % 5 == 0:
            print("RUN: {}".format(run))

        initial_weights = np.random.uniform(-0.001, 0)
        agent_info = {"num_tilings": 8, "num_tiles": 8, "iht_size": 4096, "epsilon": 0.0, "gamma": 1, "alpha": 0.1/8, "initial_weights": initial_weights, "num_actions": 3}
        env_info = {"min_position": -1.2, "max_position": 0.5, "min_velocity": -0.07, "max_velocity": 0.07, "gravity": 0.0025, "action_discount": 0.001}
        rl_glue = RLGlue(env, agent)
        rl_glue.rl_init(agent_info, env_info)
        steps_per_episode = []

        for episode in range(num_episodes):
            start = time.time()
            # 15000 is max steps of the episode
            rl_glue.rl_episode(15000)
            steps_per_episode.append(rl_glue.num_steps)
            # print("Run time: {}".format(time.time() - start))
        print(steps_per_episode)
        
        all_steps.append(np.array(steps_per_episode))
    # alist = []
    # all_steps_length = len(all_steps)
    # for i in range(len(all_steps)):
    #     alist.append(all_steps[all_steps_length - 1])
    #     all_steps_length = all_steps_length - 1
    
    plt.plot(np.mean(np.array(all_steps), axis = 0))
    plt.show()
    np.save("ExpectedSarsa_test", np.array(all_steps))
    # data = np.load("ExpectedSarsa_test.npy", np.array(all_steps))

def main():
    num_runs = 10
    num_episodes = 50
    

    run_experiment(num_runs, num_episodes)
main()






















# # Here we provide you with the true state value and state distribution
# true_state_val = np.load('data/true_V.npy')
# state_distribution = np.load('data/state_distribution.npy')

# def calc_RMSVE(learned_state_val):
#     assert(len(true_state_val) == len(learned_state_val) == len(state_distribution))
#     MSVE = np.sum(np.multiply(state_distribution, np.square(true_state_val - learned_state_val)))
#     RMSVE = np.sqrt(MSVE)
#     return RMSVE


# # Define function to run experiment
# def run_experiment(environment, agent, environment_parameters, agent_parameters, experiment_parameters):

#     rl_glue = RLGlue(environment, agent)

#     # Sweep Agent parameters
#     for num_agg_states in agent_parameters["num_groups"]:
#         for step_size in agent_parameters["step_size"]:

#             # save rmsve at the end of each evaluation episode
#             # size: num_episode / episode_eval_frequency + 1 (includes evaluation at the beginning of training)
#             agent_rmsve = np.zeros(int(experiment_parameters["num_episodes"]/experiment_parameters["episode_eval_frequency"]) + 1)

#             # save learned state value at the end of each run
#             agent_state_val = np.zeros(environment_parameters["num_states"])

#             env_info = {"num_states": environment_parameters["num_states"],
#                         "start_state": environment_parameters["start_state"],
#                         "left_terminal_state": environment_parameters["left_terminal_state"],
#                         "right_terminal_state": environment_parameters["right_terminal_state"]}

#             agent_info = {"num_states": environment_parameters["num_states"],
#                           "num_groups": num_agg_states,
#                           "step_size": step_size,
#                           "discount_factor": environment_parameters["discount_factor"]}

#             print('Setting - num. agg. states: {}, step_size: {}'.format(num_agg_states, step_size))
#             os.system('sleep 0.2')

#             # one agent setting
#             for run in tqdm(range(1, experiment_parameters["num_runs"]+1)):
#                 env_info["seed"] = run
#                 agent_info["seed"] = run
#                 rl_glue.rl_init(agent_info, env_info)

#                 # Compute initial RMSVE before training
#                 current_V = rl_glue.rl_agent_message("get state value")
#                 agent_rmsve[0] += calc_RMSVE(current_V)

#                 for episode in range(1, experiment_parameters["num_episodes"]+1):
#                     # run episode
#                     rl_glue.rl_episode(0) # no step limit

#                     if episode % experiment_parameters["episode_eval_frequency"] == 0:
#                         current_V = rl_glue.rl_agent_message("get state value")
#                         agent_rmsve[int(episode/experiment_parameters["episode_eval_frequency"])] += calc_RMSVE(current_V)

#                 # store only one run of state value
#                 if run == 50:
#                     agent_state_val = rl_glue.rl_agent_message("get state value")

#             # rmsve averaged over runs
#             agent_rmsve /= experiment_parameters["num_runs"]

#             save_name = "{}_agg_states_{}_step_size_{}".format('TD_agent', num_agg_states, step_size).replace('.','')

#             if not os.path.exists('results'):
#                 os.makedirs('results')

#             # save avg. state value
#             np.save("results/V_{}".format(save_name), agent_state_val)

#             # save avg. rmsve
#             np.save("results/RMSVE_{}".format(save_name), agent_rmsve)
