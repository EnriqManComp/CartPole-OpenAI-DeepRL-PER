import tensorflow as tf
from keras import backend

class duelingDQN():

    def __init__(self,input_dim, fc1_units, fc2_units, action_dim, lr):
        X_input = tf.keras.layers.Input(input_dim)
        X = X_input

        X = tf.keras.layers.Dense(units=256)(X)
        X = tf.keras.layers.ReLU()(X)

        V = tf.keras.layers.Dense(units=fc1_units)(X)
        V = tf.keras.layers.ReLU()(V)
        V = tf.keras.layers.Dense(1, activation=None)(V)
        V = tf.keras.layers.Lambda(lambda s: backend.expand_dims(s[:, 0], -1), output_shape=(action_dim,))(V)

        A = tf.keras.layers.Dense(units=fc2_units)(X)
        A = tf.keras.layers.ReLU()(A)
        A = tf.keras.layers.Dense(action_dim, activation=None)(A)
        A = tf.keras.layers.Lambda(lambda a: a[:, :] - backend.mean(a[:, :], keepdims=True), output_shape=(action_dim,))(A)

        X = tf.keras.layers.Add()([V, A])
            
        self.model = tf.keras.Model(inputs = X_input, outputs = X)    
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')

        self.tau = 0.001 

    def soft_update(self, q_net_weights):
        
        new_weights = [self.tau * current + (1 - self.tau) * target for current, target in zip(q_net_weights, self.model.get_weights())]
        self.model.set_weights(new_weights)  
        
        