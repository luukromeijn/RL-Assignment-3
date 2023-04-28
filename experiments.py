'''Experiments. Add experiments to run by their number (whitespace separated.
# E.g. 'python experiments.py 1 2'. Specifying no number runs all.'''

import sys
import time
import numpy as np
import tensorflow as tf
from catch import Catch
from policybased import policy_based_learning, PolicyBasedAgent
from helper import tf_control_memorygrowth, LearningCurvePlot, make_results_dir, smooth

# Some preparation
results_dir = make_results_dir('results')
log_file = open(results_dir + "/experiment.log","w")
sys.stdout = log_file

# (DEFAULT) HYPERPARAMETERS
# Experimental
n_repetitions = 1 # ?
budget = 100
n_episodes = 5 
smoothing = 15 
# Environment
rows = 7
columns = 7
speed = 1.0
max_steps = 250
max_misses = 10
observation_type = 'pixel'
seed = None
# Agent
n_actions = 3
learning_rate = 0.005
gamma = 0.95
eta = 1
n_step = max_steps + 1 
baseline_subraction = False

def average_over_repetitions(n_repetitions, env, agent, budget, n_episodes):
    rewards, actor_history, critic_history = [], [], []
    for rep in range(n_repetitions):
        print("Repetition", rep)
        tf.keras.backend.clear_session() # Prevent RAM overflow
        r, actor, critic = policy_based_learning(env, agent, budget, n_episodes)
        rewards.append(r) 
        actor_history.append(actor)
        critic_history.append(critic)
        log_file.flush()
    mean_r, std_r = np.mean(rewards, axis=0), np.std(rewards, axis=0)
    mean_act, std_act = np.mean(actor_history, axis=0), np.std(actor_history, axis=0)
    mean_crit, std_crit = np.mean(critic_history, axis=0), np.std(critic_history, axis=0)
    output = [mean_r, std_r, mean_act, std_act, mean_crit, std_crit]
    output = [smooth(array, smoothing, 1) for array in output]
    return output

# ALL EXPERIMENTS
def exp_learning_rate():
    exp_name = 'Learning rates'
    print("Running experiment:", exp_name)
    env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
                max_misses=max_misses, observation_type=observation_type, seed=seed)
    fig_r = LearningCurvePlot(xlabel='Epoch')
    fig_a = LearningCurvePlot(xlabel='Epoch', ylabel='Loss')
    fig_c = LearningCurvePlot(xlabel='Epoch', ylabel='Loss')
    for learning_rate in [0.001, 0.01, 0.1]:
        print(learning_rate)
        log_file.flush()
        t0 = time.time()
        s = env.reset() 
        agent = PolicyBasedAgent(s.shape, n_actions, learning_rate, gamma, eta, n_step, baseline_subtraction=True)
        curves = average_over_repetitions(n_repetitions, env, agent, budget, n_episodes)
        fig_r.add_curve(curves[0], var=curves[1], label='α=' + str(learning_rate))
        fig_a.add_curve(curves[2], var=curves[3], label='α=' + str(learning_rate))
        fig_c.add_curve(curves[4], var=curves[5], label='α=' + str(learning_rate))
        print(time.time()-t0, "seconds passed.")
    fig_r.save(exp_name + ' (rewards).png', results_dir)
    fig_a.save(exp_name + ' (actor).png', results_dir)
    fig_c.save(exp_name + ' (critic).png', results_dir)

# A dictionary mapping to all experiments
experiments = {'1':exp_learning_rate}

# Calling the experiments as specified by user
if len(sys.argv) < 2:
    for number in experiments:
        print("Starting experiment", number)
        log_file.flush()
        experiments[number]()
for number in sys.argv[1:]:
  try:
    print("Starting experiment", number)
    log_file.flush() 
    experiments[number]()
  except:
    print('Provide a number 1-TODO to start experiment.') # TODO

log_file.close()