import agent as ag
import numpy as np
import environment as enviro
import matplotlib.pyplot as plt
import rlglue

<<<<<<< HEAD
<<<<<<< HEAD
def calc_RMSVE(learned_state_val):
    assert(len(true_state_val) == len(learned_state_val) == len(state_distribution))
    MSVE = np.sum(np.multiply(state_distribution, np.square(true_state_val - learned_state_val)))
    RMSVE = np.sqrt(MSVE)
    return RMSVE


# Define function to run experiment
def run_experiment(environment, agent, environment_parameters, agent_parameters, experiment_parameters):

    rl_glue = RLGlue(environment, agent)

    # Sweep Agent parameters
    for num_agg_states in agent_parameters["num_groups"]:
        for step_size in agent_parameters["step_size"]:

            # save rmsve at the end of each evaluation episode
            # size: num_episode / episode_eval_frequency + 1 (includes evaluation at the beginning of training)
            agent_rmsve = np.zeros(int(experiment_parameters["num_episodes"]/experiment_parameters["episode_eval_frequency"]) + 1)

            # save learned state value at the end of each run
            agent_state_val = np.zeros(environment_parameters["num_states"])

            env_info = {"num_states": environment_parameters["num_states"],
                        "start_state": environment_parameters["start_state"],
                        "left_terminal_state": environment_parameters["left_terminal_state"],
                        "right_terminal_state": environment_parameters["right_terminal_state"]}

            agent_info = {"num_states": environment_parameters["num_states"],
                          "num_groups": num_agg_states,
                          "step_size": step_size,
                          "discount_factor": environment_parameters["discount_factor"]}

            print('Setting - num. agg. states: {}, step_size: {}'.format(num_agg_states, step_size))
            os.system('sleep 0.2')

            # one agent setting
            for run in tqdm(range(1, experiment_parameters["num_runs"]+1)):
                env_info["seed"] = run
                agent_info["seed"] = run
                rl_glue.rl_init(agent_info, env_info)

                # Compute initial RMSVE before training
                current_V = rl_glue.rl_agent_message("get state value")
                agent_rmsve[0] += calc_RMSVE(current_V)

                for episode in range(1, experiment_parameters["num_episodes"]+1):
                    # run episode
                    rl_glue.rl_episode(0) # no step limit

                    if episode % experiment_parameters["episode_eval_frequency"] == 0:
                        current_V = rl_glue.rl_agent_message("get state value")
                        agent_rmsve[int(episode/experiment_parameters["episode_eval_frequency"])] += calc_RMSVE(current_V)

                # store only one run of state value
                if run == 50:
                    agent_state_val = rl_glue.rl_agent_message("get state value")

            # rmsve averaged over runs
            agent_rmsve /= experiment_parameters["num_runs"]

            save_name = "{}_agg_states_{}_step_size_{}".format('TD_agent', num_agg_states, step_size).replace('.','')

            if not os.path.exists('results'):
                os.makedirs('results')

            # save avg. state value
            np.save("results/V_{}".format(save_name), agent_state_val)

            # save avg. rmsve
            np.save("results/RMSVE_{}".format(save_name), agent_rmsve)
=======
=======
>>>>>>> 79179ac5c96c471932023e8ccb04eae8b153f677

def run_experiment(num_runs, num_episodes, agent_info, env_info):
    all_steps = []
    agent = ag.ExpectedSarsaAgent()
    env = enviro.MountainEnvironment()

    for run in range(num_runs):
        if run % 5 == 0:
            print("RUN: {}".format(run))

        rl_glue = rlglue.RLGlue(env, agent)
        rl_glue.rl_init(agent_info, env_info)
        steps_per_episode = []

        for episode in range(num_episodes):
            # no sure why 15000
            rl_glue.rl_episode(15000)
            steps_per_episode.append(rl_glue.num_steps)
        
        all_steps.append(np.array(steps_per_episode))
    
    plt.plot(np.mean(np.array(all_steps), axis = 0))
    np.save("ExpectedSarsa_test", np.array(all_steps))

def main():
    num_runs = 50
    num_episodes = 200
    initial_weights = np.zeros((3, 4096))
    for action in range(3):
        for i in range(4096):
            initial_weights[action][i] = np.random.uniform(-0.001, 0)
    start_position = np.random.uniform(-0.6, -0.4)
    agent_info = {"num_tilings": 8, "num_tiles": 8, "iht_size": 4096, "epsilon": 0.0, "gamma": 1, "alpha": 0.1/8, "initial_weights": initial_weights, "num_actions": 3}
    env_info = {"min_position": -1.2, "max_position": 0.5, "min_velocity": -0.07, "max_velocity": 0.07, "gravity": 0.0025, "start_position": start_position, "start_velocity": 0.0, "action_discount": 0.001}

    run_experiment(num_runs, num_episodes, agent_info, env_info)
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
<<<<<<< HEAD
>>>>>>> d01e6965c87313bd294849e08b38028485e239d9
=======
>>>>>>> 79179ac5c96c471932023e8ccb04eae8b153f677
