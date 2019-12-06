import ExpectedSarsaAgent as ag
import numpy as np
import MountainCarEnvironment as enviro
import matplotlib.pyplot as plt
from rl_glue import RLGlue
import time


def default_run_experiment(num_runs, num_episodes):
    all_steps = []
    # set agent and evviroment
    agent = ag.ExpectedSarsaAgent
    env = enviro.MountainEnvironment

    total_reward_per_run = []
    # loop through all the runs
    for run in range(num_runs):
        # print which ru we at(every 5 runs)
        # start = time.time()
        if run % 5 == 0:
            print("RUN: {}".format(run))
        
        # set initial_weights to a random float in between -0.001 and 0
        initial_weights = np.random.uniform(-0.001, 0)
        # dictionary for agent_info
        agent_info = {"num_tilings": 8, "num_tiles": 8, "iht_size": 4096, "epsilon": 0.0, "gamma": 1, "alpha": 0.1/8, "initial_weights": initial_weights, "num_actions": 3}
        # dictionary for env_info
        env_info = {"min_position": -1.2, "max_position": 0.5, "min_velocity": -0.07, "max_velocity": 0.07, "gravity": 0.0025, "action_discount": 0.001}
        # called rl_glue
        rl_glue = RLGlue(env, agent)
        rl_glue.rl_init(agent_info, env_info)
        steps_per_episode = []

        # loop through all episodes
        for episode in range(num_episodes):
            # 15000 is max steps of the episode
            rl_glue.rl_episode(15000)
            # append how many steps of each episode and appended into steps_per_episode
            steps_per_episode.append(rl_glue.num_steps)
        # set total_reward
        total_reward = np.sum(steps_per_episode) * -1
        all_steps.append(np.array(steps_per_episode))
        # print("Run time: {}".format(time.time() - start))
        total_reward_per_run.append(total_reward)
    # set mean of the total reward over 50 runs
    data = np.mean(total_reward_per_run)
    # set standard error
    data_std_err = np.std(total_reward_per_run, axis=0) / np.sqrt(num_runs)
    # set title, xlabel and ylabel for the plot
    plt.title("Expected Sarsa MountainCar (Default Parameters)", fontdict={'fontsize': 16, 'fontweight' : 25}, pad=15.0)
    plt.xlabel("Epsiode", labelpad=5.0)
    plt.ylabel("Steps per Episode (averaged over " + str(num_runs) + " runs)", labelpad=10.0)
    plt.plot(np.mean(np.array(all_steps), axis = 0))
    plt.show()
    np.save("ExpectedSarsa_test", np.array(all_steps))
    # print out mean and standard error
    print("mean: ", data)
    print("standard error: ", data_std_err)


def better_run_experiment(num_runs, num_episodes):
    # Same as last function
    all_steps = []
    agent = ag.ExpectedSarsaAgent
    env = enviro.MountainEnvironment

    total_reward_per_run = []
    for run in range(num_runs):
        start = time.time()
        if run % 5 == 0:
            print("RUN: {}".format(run))

        initial_weights = np.random.uniform(-0.001, 0)
        agent_info = {"num_tilings": 32, "num_tiles": 4, "iht_size": 4096, "epsilon": 0.1, "gamma": 1, "alpha": 0.7/32, "initial_weights": initial_weights, "num_actions": 3}
        env_info = {"min_position": -1.2, "max_position": 0.5, "min_velocity": -0.07, "max_velocity": 0.07, "gravity": 0.0025, "action_discount": 0.001}
        rl_glue = RLGlue(env, agent)
        rl_glue.rl_init(agent_info, env_info)
        steps_per_episode = []


        for episode in range(num_episodes):
            # 15000 is max steps of the episode
            rl_glue.rl_episode(15000)
            steps_per_episode.append(rl_glue.num_steps)
        total_reward = np.sum(steps_per_episode) * -1
        all_steps.append(np.array(steps_per_episode))
        print("Run time: {}".format(time.time() - start))
        total_reward_per_run.append(total_reward)
    
    data = np.mean(total_reward_per_run)
    data_std_err = np.std(total_reward_per_run, axis=0) / np.sqrt(num_runs)
    plt.title("Expected Sarsa MountainCar (Alternate Parameters)", fontdict={'fontsize': 16, 'fontweight' : 25}, pad=15.0)
    plt.xlabel("Epsiode", labelpad=5.0)
    plt.ylabel("Steps per Episode (averaged over " + str(num_runs) + " runs)", labelpad=10.0)
    plt.plot(np.mean(np.array(all_steps), axis = 0))
    plt.show()
    np.save("ExpectedSarsa_test", np.array(all_steps))
    print("mean: ", data)
    print("standard error: ", data_std_err)


def main():
    # set number of runs
    num_runs = 50
    # set number of epsiodes
    num_episodes = 200
    # user input 1 or 2 to select whcih parameters setting
    action = int(input("1 for default parameters setting, 2 for alternate parameters setting: "))
    if action == 1:
        default_run_experiment(num_runs, num_episodes)
    elif action == 2:
        better_run_experiment(num_runs, num_episodes)
main()
