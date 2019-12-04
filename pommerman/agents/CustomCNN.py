import tensorflow as tf

from stable_baselines.a2c.utils import linear
from stable_baselines.common.policies import ActorCriticPolicy

class CustomCNN(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomCNN, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse)
        size=11
        bp = 3*size**2 #board partition
        with tf.variable_scope("model", reuse=reuse):
            obs = self.processed_obs
            self.board1, self.misc = tf.split(obs, [bp, -1], 1)
            
            self.board = tf.reshape(self.board1, (-1, size, size, 3))
            self.conv1 = tf.layers.conv2d(self.board, 64, 2, activation=tf.nn.relu, name='conv1')
            self.conv2 = tf.layers.conv2d(self.conv1, 32, 2, activation=tf.nn.relu, name='conv2')
            self.fc0 = tf.contrib.layers.flatten(self.conv2)
            self.fc1 = tf.concat((self.fc0, self.misc), -1)
            self.fc1 = tf.layers.dense(self.fc1, 1024, name = 'fc1')
            self.actions = tf.layers.dense(self.fc1, 6)   
            self.valueUM = tf.layers.dense(self.fc1, 128) #??

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(self.actions, self.valueUM, init_scale=0.01)

        self._value_fn = linear(self.valueUM, 'vf', 1)
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})