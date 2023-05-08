from policybased import PolicyBasedAgent
from catch import Catch
import tensorflow as tf

print(tf.__version__)


# Path to saved Actor network
model_path = 'results\experiments-7-5\Vector\model'

# Specify environment variations
rows = 7
columns = 7
speed = 1.0
max_steps = 250
max_misses = 10
observation_type = 'vector' # 'vector'
seed = None

# Initialize environment
env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
            max_misses=max_misses, observation_type=observation_type, seed=seed)
s = env.reset()
step_pause = 0.3 # the pause between each plot
env.render(step_pause)

# Initialize agent
agent = PolicyBasedAgent(s.shape, 3)
agent.actor = tf.saved_model.load(model_path)

done = False
while not done:
    a, probs = agent.select_action(s)
    s, r, done, _ = env.step(a)
    print(r)
    env.render(step_pause)