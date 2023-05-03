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
results_dir = make_results_dir('experiments-3-5')
log_file = open(results_dir + "/experiment.log","w")
sys.stdout = log_file

# class mock_logfile:

#     def flush():
#         pass

#     def close():
#         pass

# log_file = mock_logfile

# (DEFAULT) HYPERPARAMETERS
# Experimental
budget = 1000
n_episodes = 1  
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
n_step = 10000000
baseline_subtraction = False

def run_experiment(env, agent, budget, n_episodes, smoothing, model_save=None):
    tf.keras.backend.clear_session() # Prevent RAM overflow
    rewards, actor_history, critic_history = policy_based_learning(env, agent, budget, n_episodes, model_save)
    log_file.flush()
    mean_r, std_r = np.mean(rewards, axis=1), np.std(rewards, axis=1)
    mean_act, std_act = np.mean(actor_history, axis=1), np.std(actor_history, axis=1)
    mean_crit, std_crit = np.mean(critic_history, axis=1), np.std(critic_history, axis=1)
    output = [mean_r, std_r, mean_act, std_act, mean_crit, std_crit]
    output = [smooth(array, smoothing, 1) for array in output]
    return output

# ALL EXPERIMENTS
def exp_hyperparameter_search():
    
    baseline_subtraction = True
    n_step = 5
    exp_name = 'Hyperparameter search'
    print("Running experiment:", exp_name)
    
    fig_r = LearningCurvePlot(xlabel='Epoch')
    fig_a = LearningCurvePlot(xlabel='Epoch', ylabel='Loss')
    fig_c = LearningCurvePlot(xlabel='Epoch', ylabel='Loss')

    env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
                max_misses=max_misses, observation_type=observation_type, seed=seed)
    
    for learning_rate in [0.005, 0.01, 0.05]:
      for eta in [0, 0.1, 0.2]:
        print(learning_rate, eta)
        log_file.flush()
        t0 = time.time()
        s = env.reset() 
        agent = PolicyBasedAgent(s.shape, n_actions, learning_rate, gamma, eta, n_step, baseline_subtraction)
        try:
            curves = run_experiment(env, agent, budget, n_episodes, smoothing)
        except:
            print("An error occurred.")
            continue
        fig_r.add_curve(curves[0], var=curves[1], label='α='+ str(learning_rate)+'; η='+str(eta))
        fig_a.add_curve(curves[2], var=curves[3], label='α='+ str(learning_rate)+'; η='+str(eta))
        fig_c.add_curve(curves[4], var=curves[5], label='α='+ str(learning_rate)+'; η='+str(eta))
        fig_r.save(exp_name + ' (rewards).png', results_dir)
        fig_a.save(exp_name + ' (actor).png', results_dir)
        fig_c.save(exp_name + ' (critic).png', results_dir)
        print(time.time()-t0, "seconds passed.")

    return 

def exp_comparison():
    
    exp_name = 'Comparison'
    print("Running experiment:", exp_name)
    
    settings = {'REINFORCE': (10000000, False),
                'Baseline sub.': (10000000, True),
                'Bootstrap (n=1)': (1, False),
                'Bootstrap (n=5)': (5, False),
                'Base + boot (n=5)': (5, True)}
    
    n_episodes = 3
    learning_rate = 0.01
    eta = 0.1
    
    fig_r = LearningCurvePlot(xlabel='Epoch')
    fig_a = LearningCurvePlot(xlabel='Epoch', ylabel='Loss')
    fig_c = LearningCurvePlot(xlabel='Epoch', ylabel='Loss')

    env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
                max_misses=max_misses, observation_type=observation_type, seed=seed)
    
    for setting in settings:
        print(setting)
        n_step, baseline_subtraction = settings[setting]
        log_file.flush()
        t0 = time.time()
        s = env.reset() 
        agent = PolicyBasedAgent(s.shape, n_actions, learning_rate, gamma, eta, n_step, baseline_subtraction)
        try:
            curves = run_experiment(env, agent, budget, n_episodes, smoothing)
        except:
            print("An error occurred.")
            continue
        fig_r.add_curve(curves[0], var=curves[1], label=setting)
        fig_a.add_curve(curves[2], var=curves[3], label=setting)
        fig_c.add_curve(curves[4], var=curves[5], label=setting)
        fig_r.save(exp_name + ' (rewards).png', results_dir)
        fig_a.save(exp_name + ' (actor).png', results_dir)
        fig_c.save(exp_name + ' (critic).png', results_dir)
        print(time.time()-t0, "seconds passed.")

    return 


# A dictionary mapping to all experiments
experiments = {'1':exp_hyperparameter_search, 
               '2':exp_comparison}

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