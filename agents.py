# TODO docstring

# Imports
import tensorflow as tf
from keras import Model
from keras.models import Sequential, clone_model
from keras.layers import Dense, Input, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from catch import Catch

# Entropy function
h = lambda P: -tf.tensordot(P, tf.math.log(P), 1)

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
    baseline_subraction (bool)'''
    
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
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
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
    
    policy_gradient = tape.gradient(policy_loss, self.actor.trainable_variables)
    self.optimizer.apply_gradients(zip(policy_gradient, self.actor.trainable_variables))
    value_gradient = tape.gradient(value_loss, self.critic.trainable_variables)
    self.optimizer.apply_gradients(zip(value_gradient, self.critic.trainable_variables))

    return policy_loss, value_loss


  def loss_function(self, traces):
    ''''Calculates loss for critic and actor'''
    
    policy_loss, value_loss = 0, 0
    for trace in traces:
      states, actions, rewards = trace[0], trace[1], trace[2]
      for i in range(len(states)):
        policy, value = self.actor(states[i])[0], self.critic(states[i])[0]
        # (N-step) future discounted reward
        R = np.sum([(self.gamma**k)*rewards[i+k] for k in 
                      range(min(self.n_step,len(states)-i))])
        # Insert bootstrap estimate if end of trace is not reached after N steps
        if i + self.n_step < len(states):
          R += self.gamma**self.n_step * self.critic(states[i+self.n_step])[0][actions[i+self.n_step]]
        if self.baseline_subtraction:
          R -= value[actions[i]]
        # Policy loss = R * log(pi(a_t|s_t)) + entropy regularization
        policy_loss += R * tf.math.log(policy[actions[i]]) + self.eta*h(policy)
        # Value loss = MSE
        value_loss += (R - value[actions[i]])**2

    # policy_loss = 1/len(traces)*policy_loss  
    return -policy_loss, value_loss

def policy_based_learning(env, agent, budget, n_episodes):
  '''Learns a policy-based agent'''

  avg_rewards, actor_history, critic_history = [], [], []
  for epoch in range(budget):
    traces = []
    epoch_reward = 0
    for episode in range(n_episodes): # Generate random episodes
      s = env.reset()
      done = False
      states, actions, rewards = [], [], []
      while not done: # Generate trace
        a, prob = agent.select_action(s)
        s_next, r, done, info = env.step(a)
        states.append(np.expand_dims(s,0))
        actions.append(a)
        rewards.append(r)
        s = s_next
      total_reward = np.sum(rewards)
      epoch_reward += total_reward
      traces.append([states, actions, rewards])
    print("Epoch", epoch, "avg reward:", epoch_reward/n_episodes)
    policy_loss, value_loss = agent.update_model(traces) # Update the model using the generated traces
    actor_history.append(policy_loss)
    critic_history.append(value_loss)
    avg_rewards.append(epoch_reward/n_episodes)

  return avg_rewards, actor_history, critic_history

if __name__ == '__main__':

    # From https://www.tensorflow.org/guide/gpu 
    # (sometimes needed to prevent OOM on dslab servers)
    tf.keras.backend.clear_session()
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
          logical_gpus = tf.config.list_logical_devices('GPU')
          print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
          # Memory growth must be set before GPUs have been initialized
          print(e)

    # Environment parameters
    rows = 7
    columns = 7
    speed = 1.0
    max_steps = 250
    max_misses = 10
    observation_type = 'pixel' # 'vector'
    seed = None

    # Agent hyperparameters
    budget = 1000
    n_episodes = 5

    # Initialize environment and Q-array
    env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
                max_misses=max_misses, observation_type=observation_type, seed=seed)
    s = env.reset()
    agent = PolicyBasedAgent(input_shape=s.shape, n_actions=3, learning_rate=0.005, gamma=0.95, eta=1, n_step=5, baseline_subtraction=True)

    rewards, actor_history, critic_history = policy_based_learning(env, agent, budget, n_episodes)
    plt.plot(rewards)
    plt.xlabel('Epochs')
    plt.ylabel('Average rewards')
    plt.title('Policy based learning')
    plt.savefig('rewards.png')
    plt.clf()
    plt.plot(actor_history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Policy based learning')
    plt.savefig('actor_history.png')
    plt.clf()
    plt.plot(critic_history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Policy based learning')
    plt.savefig('critic_history.png')

