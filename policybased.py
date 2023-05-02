'''REINFORCE/actor-critic with(out) bootstrapping and/or baseline subtraction.
Optional flags: --bootstrapping or --baselinesubtraction.'''

# Imports
import sys
import tensorflow as tf
from keras import Model
from keras.models import Sequential, clone_model
from keras.layers import Dense, Input, Flatten
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from catch import Catch
from helper import LearningCurvePlot, tf_control_memorygrowth

tau = 0.000001 # A small number to prevent undefined logs
# Entropy function
h = lambda P: -tf.tensordot(P, tf.math.log(P + tau), 1)

class PolicyBasedAgent:

  def __init__(self, input_shape, n_actions, learning_rate=0.005, gamma=0.95, eta=1.0, n_step=np.inf, baseline_subtraction=False):
    '''A policy-based, deep reinforcement learning agent. 
    
    Parameters
    ----------
    input_shape (tuple)
    n_actions (int)
    learning_rate (float): alpha
    gamma (float): discount factor
    eta (float): weight for entropy regularization
    n_step (int): after how many future steps a bootstrap estimator is used
    baseline_subtraction (bool)'''
    
    # Setting the hyperparameters
    self.learning_rate = learning_rate
    self.n_actions = n_actions
    self.gamma = gamma
    self.eta = eta
    self.optimizer = Adam(self.learning_rate)
    self.n_step = n_step
    self.baseline_subtraction = baseline_subtraction
    
    # Actor and critic share the same base model architecture (but not weights)
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Flatten())
    model.add(Dense(12, activation='relu'))
    model.add(Dense(12, activation='relu'))
    self.actor = clone_model(model)
    self.actor.add(Dense(self.n_actions, activation='softmax'))
    self.critic = clone_model(model)
    self.critic.add(Dense(self.n_actions, activation='linear'))

  def select_action(self, s):
    '''Selects action based on probability distribution'''
    pred = self.actor(np.expand_dims(s,0))[0].numpy()
    a = np.random.choice(self.n_actions, p=pred)
    return a, pred[a]

  def update_model(self, traces):
    '''Calculates loss and takes optimization step'''
    with tf.GradientTape(persistent=True) as tape:
      policy_loss, value_loss = self.loss_function(traces)
      avg_policy_loss = tf.math.reduce_mean(policy_loss)
      avg_value_loss = tf.math.reduce_mean(value_loss)
    
    policy_gradient = tape.gradient(avg_policy_loss, self.actor.trainable_variables)
    self.optimizer.apply_gradients(zip(policy_gradient, self.actor.trainable_variables))
    if self.baseline_subtraction or self.n_step < 10000000: # Check if critic is used
      value_gradient = tape.gradient(avg_value_loss, self.critic.trainable_variables)
      self.optimizer.apply_gradients(zip(value_gradient, self.critic.trainable_variables))

    return policy_loss, value_loss

  def loss_function(self, traces):
    ''''Calculates loss for critic and actor'''

    policy_loss, value_loss = [], []
    for trace in traces:
      
      states, actions, rewards = trace[0], trace[1], trace[2]
      policies, values = self.actor(states), self.critic(states)
      trace_p_loss, trace_v_loss = 0, 0
      
      for i in range(len(states)):
        policy, value = policies[i], values[i]
        # (N-step) future discounted reward
        R = np.sum([(self.gamma**k)*rewards[i+k] for k in 
                      range(min(self.n_step,len(states)-i))])
        # Insert bootstrap estimate if end of trace is not reached after N steps
        if i + self.n_step < len(states):
          R += (self.gamma**self.n_step) * self.critic(tf.expand_dims(states[i+self.n_step],0))[0][actions[i+self.n_step]]
        if self.baseline_subtraction:
          R -= value[actions[i]]
        # Policy loss = R * log(pi(a_t|s_t)) + entropy regularization
        trace_p_loss -= R * tf.math.log(policy[actions[i]] + tau) + self.eta*h(policy)
        # Value loss = MSE
        trace_v_loss += (R - value[actions[i]])**2

      policy_loss.append(trace_p_loss)
      value_loss.append(trace_v_loss)

    return policy_loss, value_loss


def policy_based_learning(env, agent, budget, n_episodes, save_in_dir=None):
  '''Learns a policy-based agent'''

  training_rewards, actor_history, critic_history = [], [], []
  for epoch in range(budget):
    traces = []
    trace_rewards = []
    for episode in range(n_episodes): # Generate random episodes
      s = env.reset()
      done = False
      states, actions, rewards = [], [], []
      while not done: # Generate trace
        a, prob = agent.select_action(s)
        s_next, r, done, info = env.step(a)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = s_next
      trace_rewards.append(np.sum(rewards))
      traces.append([tf.stack(states), tf.stack(actions), tf.stack(rewards)])
    if __name__ == '__main__':
      print("Epoch", epoch, "avg reward:", np.mean(trace_rewards))
    policy_loss, value_loss = agent.update_model(traces) # Update the model using the generated traces
    actor_history.append(policy_loss)
    critic_history.append(value_loss)
    training_rewards.append(trace_rewards)

  if save_in_dir is not None:
    agent.actor.save(save_in_dir + '/model')

  return np.array(training_rewards), np.array(actor_history), np.array(critic_history)

if __name__ == '__main__':

    tf_control_memorygrowth() # For DSLab servers

    # (DEFAULT) HYPERPARAMETERS
    # Environment
    rows = 7
    columns = 7
    speed = 1.0
    max_steps = 250
    max_misses = 10
    observation_type = 'pixel' # 'vector'
    seed = None
    # Agent
    budget = 1000
    n_episodes = 5
    n_actions = 3
    learning_rate = 0.005
    gamma = 0.95
    eta = 1
    n_step = 10000000 # default = no bootstrapping
    baseline_subtraction = False

    for arg in sys.argv:
      if arg == '--bootstrapping':
        print("Bootstrapping on.")
        n_step = 5
      if arg == '--baselinesubtraction':
        print("Baseline subtraction on.")
        baseline_subtraction = True

    # Initialize environment and Q-array
    env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
                max_misses=max_misses, observation_type=observation_type, seed=seed)
    s = env.reset()
    agent = PolicyBasedAgent(s.shape, n_actions, learning_rate, gamma, eta, n_step, baseline_subtraction)

    rewards, actor_history, critic_history = policy_based_learning(env, agent, budget, n_episodes)
    rewards = np.mean(rewards, axis=1)
    actor_history = np.mean(actor_history, axis=1)
    critic_history = np.mean(critic_history, axis=1)
    
    # Save results
    fig = LearningCurvePlot('Policy based learning', 'Epochs', 'Rewards (avg)')
    fig.add_curve(rewards)
    fig.save('rewards.png')
    fig = LearningCurvePlot('Actor history', 'Epochs', 'Loss')
    fig.add_curve(actor_history)
    fig.save('actor_history.png')
    fig = LearningCurvePlot('Critic history', 'Epochs', 'Loss')
    fig.add_curve(critic_history)
    fig.save('critic_history.png')