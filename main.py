import tensorflow as tf

# From https://www.tensorflow.org/guide/gpu 
# (sometimes needed to prevent OOM on dslab servers)
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

# BACKUP
  # def update_model(self, traces):
  #   '''Calculates loss and takes optimization step'''
  #   with tf.GradientTape() as tape:
  #     loss = self.loss_function(traces)
    
  #   gradients = tape.gradient(loss, self.model.trainable_variables)
  #   before = self.model.trainable_variables
  #   self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
  #   after = self.model.trainable_variables

  # def loss_function(self, traces):
  #   '''Cumulative reward * trace probability'''
  #   loss = 0
  #   for trace in traces:
  #     R = 0
  #     for i in range(len(trace)-1,-1,-1):
  #       s, a, r = np.expand_dims(trace[i][0],0), trace[i][1], trace[i][2]
  #       pred = self.model(s)[0]
  #       R = r + self.gamma*R
  #       loss += R*tf.math.log(pred[a]) + self.eta*h(pred) # Entropy regularization
  #   # loss = -1/len(traces)*loss
  #   return -loss

    # # Building the actor model
    # model = Sequential()
    # input = Input(shape=input_shape)
    # flatten = Flatten()(input)
    # dense_1 = Dense(12, activation='relu')(flatten)
    # dense_2 = Dense(12, activation='relu')(dense_1)
    # policy = Dense(self.n_actions, activation='softmax')(dense_2)
    # value = Dense(self.n_actions, activation='linear')(dense_2)
    # self.actor = Model(inputs=[input], outputs=[policy, value])

    # # Building the critic model
    # self.critic = clone_model(self.actor)