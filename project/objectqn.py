import uuid
import time
import pickle
import sys
import gym.spaces
import itertools
import numpy as np
import random
import shutil
import os

import tensorflow                as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python.framework import ops
from collections import namedtuple
from dqn_utils import *

import cv2
import math

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])


def attention_model(objects_in, num_slots,  scope, reuse=False, beta=20.0):

    with tf.variable_scope(scope, reuse=reuse):
        out = objects_in
        
        out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_slots, activation_fn=None)
        
        softm = tf.exp(out * beta) / tf.reduce_sum(tf.exp(out * beta), 1)
        att = tf.reduce_mean(softm * objects_in, 1)
        return att

def vision_model(img_in, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)

        return out

def policy_model(representations, num_actions, scope, reuse=False):
    out = representations
    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

   return out

def template_matching(last_frame, templates, threshold=0.5):

    # ALL TEMPLATE MATCHING STUFF
    all_object_locs = []
    for num in range(0,len(self.templates)):
        res = cv2.matchTemplate(last_frame,self.templates[num],cv2.TM_CCOEFF_NORMED)
        object_locs = np.where(res >= threshold)
        object_locs = list(zip(list(object_locs[0]), list(object_locs[1])))
        suppressed_locs = cluster_detections(object_locs)
        labeled_objects = [(x,y, num) for x,y in suppressed_locs]
        all_object_locs += labeled_objects

    print("object_locs", all_object_locs)
    
    return all_object_locs

def cluster_detections( object_locs, radius=3.0):
  detections = {}

  for i, (x_loc, y_loc) in enumerate(object_locs):
      key = (x_loc, y_loc)
      if key in detections:
          clusters = detections[key]
      else:
          clusters = [key]
      
      for x2, y2 in object_locs[i+1:]:
          dist = math.hypot(x2 - x_loc, y2 - y_loc)
          if dist <= radius:
              key2 = (x2, y2)
              if key2 not in clusters:
                  if key2 in detections:
                      clusters2 = detections[key2]
                      for tmp_key in clusters2:
                          if tmp_key not in clusters:
                              clusters.append(tmp_key)
                      
                      detections[key2] = clusters
                  else:
                      clusters.append(key2)
                      detections[key2] = clusters

  final_objects = []
  for key, clusters in detections.items():
      inverse_clusters = list(zip(*clusters))
      x = round(np.mean(list(inverse_clusters[0])))
      y = round(np.mean(list(inverse_clusters[1])))
      point = (x, y)
      if point not in final_objects:
          final_objects.append(point)

  return final_objects

class QLearner(object):

  def __init__(
    self,
    env,
    q_func,
    optimizer_spec,
    session,
    exploration=LinearSchedule(1000000, 0.1),
    stopping_criterion=None,
    replay_buffer_size=1000000,
    batch_size=32,
    gamma=0.99,
    learning_starts=50000,
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=10000,
    grad_norm_clipping=10,
    rew_file=None,
    double_q=True,
    lander=False,
    mod_file=None,
    src_task=None,
    target_task=None,
    preload=False):
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    session: tf.Session
        tensorflow session to use.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    double_q: bool
        If True, then use double Q-learning to compute target values. Otherwise, use vanilla DQN.
        https://papers.nips.cc/paper/3964-double-q-learning.pdf
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    self.target_update_freq = target_update_freq
    self.optimizer_spec = optimizer_spec
    self.batch_size = batch_size
    self.learning_freq = learning_freq
    self.learning_starts = learning_starts
    self.stopping_criterion = stopping_criterion
    self.env = env
    self.session = session
    self.exploration = exploration
    self.rew_file = str(uuid.uuid4()) + '.pkl' if rew_file is None else rew_file
    self.mod_file = "tmp_model.ckpt" if mod_file is None else mod_file
    self.src_task = src_task
    self.target_task = target_task
    self.preload = preload
    
    template_dir = "/home/dguillory/workspace/homework/templates"

    files = [ os.path.join(template_dir, f) for f in os.listdir(template_dir) if os.path.isfile(os.path.join(template_dir, f))]
    self.templates = [cv2.imread(f, 0) for f in files]
    self.template_cnt = len(self.templates)

    ###############
    # BUILD MODEL #
    ###############
    
    if True:
        if len(self.env.observation_space.shape) == 1:
            # This means we are running on low-dimensional observations (e.g. RAM)
            input_shape = self.env.observation_space.shape
        else:
            img_h, img_w, img_c = self.env.observation_space.shape
            input_shape = (img_h, img_w, frame_history_len * img_c)
            print("INPUT SHAPE")
            print(input_shape)
        self.num_actions = self.env.action_space.n
    
        # set up placeholders

        # placeholder for templates
        self.template_loc_ph = tf.placeholder(
                tf.float32,
                shape=(None, None, 2)
                name="templates"
                )

        self.template_class_ph = tf.placeholder(
                tf.uint8, shape=(None, None, 1),
                name="object_class")


        # placeholder for current observation (or state)
        self.obs_t_ph              = tf.placeholder(
            tf.float32 if lander else tf.uint8, [None] + list(input_shape))
        # placeholder for current action
        self.act_t_ph              = tf.placeholder(tf.int32,   [None])
        # placeholder for current reward
        self.rew_t_ph              = tf.placeholder(tf.float32, [None])
        # placeholder for next observation (or state)
        self.obs_tp1_ph            = tf.placeholder(
            tf.float32 if lander else tf.uint8, [None] + list(input_shape))
        # placeholder for end of episode mask
        # this value is 1 if the next state corresponds to the end of an episode,
        # in which case there is no Q-value at the next state; at the end of an
        # episode, only the current state reward contributes to the target, not the
        # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
        self.done_mask_ph          = tf.placeholder(tf.float32, [None])
    
        # casting to float on GPU ensures lower data transfer times.
        if lander:
          obs_t_float = self.obs_t_ph
          obs_tp1_float = self.obs_tp1_ph
        else:
          obs_t_float   = tf.cast(self.obs_t_ph,   tf.float32) / 255.0
          obs_tp1_float = tf.cast(self.obs_tp1_ph, tf.float32) / 255.0
    
        # Here, you should fill in your own code to compute the Bellman error. This requires
        # evaluating the current and next Q-values and constructing the corresponding error.
        # TensorFlow will differentiate this error for you, you just need to pass it to the
        # optimizer. See assignment text for details.
        # Your code should produce one scalar-valued tensor: total_error
        # This will be passed to the optimizer in the provided code below.
        # Your code should also produce two collections of variables:
        # q_func_vars
        # target_q_func_vars
        # These should hold all of the variables of the Q-function network and target network,
        # respectively. A convenient way to get these is to make use of TF's "scope" feature.
        # For example, you can create your Q-function network with the scope "q_func" like this:
        # <something> = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)
        # And then you can obtain the variables like this:
        # q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
        # Older versions of TensorFlow may require using "VARIABLES" instead of "GLOBAL_VARIABLES"
        # Tip: use huber_loss (from dqn_utils) instead of squared error when defining self.total_error
        ######
    
        # YOUR CODE HERE
        self.tot_errs = []

        # Templates

        if self.objects:
            # Do Processing for template matching
            object_one_hot = tf.one_hot(self.template_class_ph, self.template_cnt)
            object_candidates = tf.concat([self.template_loc_ph, object_one_hot], 2)
            att_rep = attention_model(object_candidates, self.num_slots, scope="attention_model", reuse=False)

        if self.vision:
            # Do Processing for visual parts
            vision_rep = vision_model(obs_t_float, scope="vision_model", reuse=False)

        if self.objects and self.vision:
            rep = tf.concat([vision_rep, att_rep], 2))
        elif self.objects:
            rep = att_rep
        else:
            rep = vision_rep
        


        base_model = policy_model(rep, self.num_actions, scope="q_func", reuse=False)
        one_hot = tf.one_hot(self.act_t_ph, self.num_actions)
        base_q = tf.reduce_sum(tf.multiply(base_model, one_hot),  axis=1)
      
        if double_q:
            print("Double DQN")
            target_q_net = q_func(obs_tp1_float, self.num_actions, scope="q_func", reuse=True)
            max_action = tf.argmax(target_q_net, axis=1)
            target_q_est = q_func(obs_tp1_float, self.num_actions, scope="target_q_func", reuse=False)
            action_filter = tf.one_hot(max_action, self.num_actions)
            max_q = tf.reduce_sum(tf.multiply(action_filter, target_q_est), axis=1)
        else:
            max_q = tf.reduce_max(q_func(obs_tp1_float, self.num_actions, scope="target_q_func", reuse=False), axis=1)
    
        q_contribution = tf.multiply(1.0 - self.done_mask_ph, gamma * max_q)
        target_q = self.rew_t_ph + q_contribution  
        target_q = tf.stop_gradient(target_q)
        self.target_q = target_q
        self.base_q = base_q
        self.total_error = tf.reduce_mean(huber_loss(target_q - base_q)) 
        self.best_action = tf.argmax(base_model, axis=1)[0]
        
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_q_func") 
    
        ######
    
        # construct optimization op (with gradient clipping)
        self.learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
        optimizer = self.optimizer_spec.constructor(learning_rate=self.learning_rate, **self.optimizer_spec.kwargs)
        self.train_fn = minimize_and_clip(optimizer, self.total_error,
                     var_list=q_func_vars, clip_val=grad_norm_clipping)
        

        #saver_dict = {v.name: v for v in tf.global_variables()}
        self.saver = tf.train.Saver() #saver_dict)
        
        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_fn = []
        for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_fn.append(var_target.assign(var))
        self.update_target_fn = tf.group(*update_target_fn)
    
    # construct the replay buffer
    self.replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len, lander=lander)
    self.replay_buffer_idx = None

    ###############
    # RUN ENV     #
    ###############

    self.model_initialized = False
    self.num_param_updates = 0
    self.mean_episode_reward      = -float('nan')
    self.best_mean_episode_reward = -float('inf')
    self.last_obs = self.env.reset()
    self.log_every_n_steps = 10000
    self.log_data = []
    self.start_time = None
    self.t = 0
    self._last_save = 0


 
  
  def stopping_criterion_met(self):
    return self.stopping_criterion is not None and self.stopping_criterion(self.env, self.t)

  def step_env(self):
    ### 2. Step the env and store the transition
    # At this point, "self.last_obs" contains the latest observation that was
    # recorded from the simulator. Here, your code needs to store this
    # observation and its outcome (reward, next observation, etc.) into
    # the replay buffer while stepping the simulator forward one step.
    # At the end of this block of code, the simulator should have been
    # advanced one step, and the replay buffer should contain one more
    # transition.
    # Specifically, self.last_obs must point to the new latest observation.
    # Useful functions you'll need to call:
    # obs, reward, done, info = env.step(action)
    # this steps the environment forward one step
    # obs = env.reset()
    # this resets the environment if you reached an episode boundary.
    # Don't forget to call env.reset() to get a new observation if done
    # is true!!
    # Note that you cannot use "self.last_obs" directly as input
    # into your network, since it needs to be processed to include context
    # from previous frames. You should check out the replay buffer
    # implementation in dqn_utils.py to see what functionality the replay
    # buffer exposes. The replay buffer has a function called
    # encode_recent_observation that will take the latest observation
    # that you pushed into the buffer and compute the corresponding
    # input that should be given to a Q network by appending some
    # previous frames.
    # Don't forget to include epsilon greedy exploration!
    # And remember that the first time you enter this loop, the model
    # may not yet have been initialized (but of course, the first step
    # might as well be random, since you haven't trained your net...)

    #####

    # YOUR CODE HERE
    last_obs = self.last_obs 
    frame_idx = self.replay_buffer.store_frame(last_obs)    
    
    if self.model_initialized:
        epsilon = self.exploration.value(self.t)
        if random.random() < epsilon:
            action = random.randint(0, self.num_actions - 1)
            
        else:
            net_in = self.replay_buffer.encode_recent_observation()

            input_encoding = np.expand_dims(net_in, 0)
            last_frame = net_in[:, :, 3]
            objects = template_matching(last_frame, self.templates) #add threshold arg
            template_loc = np.array([[x, y] for x, y, l in objects])
            template_class = np.array([l, for x, y, l in objects]) 
            action = self.session.run([self.best_action], feed_dict={self.obs_t_ph: input_encoding,
                                                                     self.template_loc_ph: template_loc,
                                                                     self.template_class_ph: template_class})[0]
            #print("selected action")
            #print(action)
                        
    else:
        action = random.randint(0, self.num_actions - 1)
    
    obs, reward, done, info = self.env.step(action)
    self.replay_buffer.store_effect(frame_idx, action, reward, done)
    
    if done:
        self.last_obs = self.env.reset()
    else:
        self.last_obs = obs

    
        


  def update_model(self):
    ### 3. Perform experience replay and train the network.
    # note that this is only done if the replay buffer contains enough samples
    # for us to learn something useful -- until then, the model will not be
    # initialized and random actions should be taken
    if (self.t > self.learning_starts and \
        self.t % self.learning_freq == 0 and \
        self.replay_buffer.can_sample(self.batch_size)):
      # Here, you should perform training. Training consists of four steps:
      # 3.a: use the replay buffer to sample a batch of transitions (see the
      # replay buffer code for function definition, each batch that you sample
      # should consist of current observations, current actions, rewards,
      # next observations, and done indicator).
      # 3.b: initialize the model if it has not been initialized yet; to do
      # that, call
      #    initialize_interdependent_variables(self.session, tf.global_variables(), {
      #        self.obs_t_ph: obs_t_batch,
      #        self.obs_tp1_ph: obs_tp1_batch,
      #    })
      # where obs_t_batch and obs_tp1_batch are the batches of observations at
      # the current and next time step. The boolean variable model_initialized
      # indicates whether or not the model has been initialized.
      # Remember that you have to update the target network too (see 3.d)!
      # 3.c: train the model. To do this, you'll need to use the self.train_fn and
      # self.total_error ops that were created earlier: self.total_error is what you
      # created to compute the total Bellman error in a batch, and self.train_fn
      # will actually perform a gradient step and update the network parameters
      # to reduce total_error. When calling self.session.run on these you'll need to
      # populate the following placeholders:
      # self.obs_t_ph
      # self.act_t_ph
      # self.rew_t_ph
      # self.obs_tp1_ph
      # self.done_mask_ph
      # (this is needed for computing self.total_error)
      # self.learning_rate -- you can get this from self.optimizer_spec.lr_schedule.value(t)
      # (this is needed by the optimizer to choose the learning rate)
      # 3.d: periodically update the target network by calling
      # self.session.run(self.update_target_fn)
      # you should update every target_update_freq steps, and you may find the
      # variable self.num_param_updates useful for this (it was initialized to 0)
      #####

      # YOUR CODE HERE
      obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self.replay_buffer.sample(self.batch_size)
     
 
      if not self.model_initialized:
          """
          #with tf.variable_scope("q_func", reuse=tf.AUTO_REUSE):
              #tqf = tf.get_variable("target_q_func/action_value/fully_connected_1/weights", shape=[512, 8])
              #qf = tf.get_variable("q_func/action_value/fully_connected_1/weights", shape=[512, 8])
              #tqf = tf.assign(tqf, tf.placeholder(tf.float32, [None, 18]), validate_shape=False)
              #qf = tf.assign(qf, tf.placeholder(tf.float32, [None, 18]), validate_shape=False)
          """

          print("LOADING A SAVED MODEL")
          #self.saver.restore(self.session, "/home/dguillory/workspace/homework/models/SpaceInvadersNoFrameskip-v4.ckpt")
          #self.saver.restore(self.session, "/home/dguillory/workspace/homework/models/ZaxxonNoFrameskip-v4.ckpt")
          #self.save_model()
          #1/0
          
          initialize_interdependent_variables(self.session, tf.global_variables(), {
              self.obs_t_ph: obs_batch,
              self.obs_tp1_ph: next_obs_batch})
          
          if self.preload and self.src_task is not None:
              #chkp.print_tensors_in_checkpoint_file("/home/dguillory/workspace/homework/test_models/" + self.src_task + "/variables/variables", tensor_name='', all_tensors=True)
              print ("BREAK After test models \n\n")
              #chkp.print_tensors_in_checkpoint_file("/home/dguillory/workspace/homework/models/" + self.src_task + ".ckpt", tensor_name='', all_tensors=True)
              #1/0
              print(tf.global_variables())
              #print([n.name for n in tf.get_default_graph().as_graph_def().node])

              #uncomment
              variables_to_restore = [var for var in tf.global_variables() if "convnet" in var.name]
              saver = tf.train.Saver(variables_to_restore)

              saver.restore(self.session, "/home/dguillory/workspace/homework/models/" + self.src_task +".ckpt")
              
              #print("/home/dguillory/workspace/homework/test_models/" + self.src_task)
              #tf.saved_model.loader.load(self.session, [tf.saved_model.tag_constants.TRAINING], "/home/dguillory/workspace/homework/test_models/" + self.src_task)
              print("AFTER LOAD")
              print(tf.global_variables())
              #print([n.name for n in tf.get_default_graph().as_graph_def().node])

          
          self.model_initialized = True
      
      gradients, tot_err, target_q, base_q = self.session.run([self.train_fn, self.total_error, self.target_q, self.base_q], feed_dict={
                                                                               self.obs_t_ph: obs_batch,
                                                                               self.act_t_ph: act_batch,
									       self.rew_t_ph: rew_batch,
									       self.obs_tp1_ph: next_obs_batch,
									       self.done_mask_ph: done_mask,
									       self.learning_rate: self.optimizer_spec.lr_schedule.value(self.t)
                                                                               })
      self.tot_errs.append(tot_err)
      if self.num_param_updates % 10000 == 0:
          print("total err: ")
          print( np.mean(self.tot_errs))
          self.tot_errs = []
          #print("target q: " + str(target_q))
          #print("base q: " + str(base_q))
      if self.num_param_updates % self.target_update_freq == 0:
          self.session.run([self.update_target_fn], feed_dict={
						       self.obs_t_ph: obs_batch,
						       self.act_t_ph: act_batch,
						       self.rew_t_ph: rew_batch,
						       self.obs_tp1_ph: next_obs_batch,
						       self.done_mask_ph: done_mask,
						       self.learning_rate: self.optimizer_spec.lr_schedule.value(self.t) 
						       })

      self.num_param_updates += 1

    self.t += 1

  def save_model(self):
      b = tf.saved_model.builder.SavedModelBuilder("/tmp/" + self.mod_file)
      b.add_meta_graph_and_variables(
	  self.session,
	  tags=[tf.saved_model.tag_constants.TRAINING],
	  assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
          strip_default_attrs=True,
	  clear_devices=True)
      b.save()
      dirpath = "/home/dguillory/workspace/homework/test_models/" + self.mod_file
      if os.path.exists(dirpath) and os.path.isdir(dirpath):
          shutil.rmtree(dirpath)
      shutil.move("/tmp/" + self.mod_file, dirpath)
      print("Done Saving")

  def log_progress(self):
    episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()

    if len(episode_rewards) > 0:
      self.mean_episode_reward = np.mean(episode_rewards[-100:])


    if  False: #self.t > (self.log_every_n_steps + self._last_save)  and self.mean_episode_reward > self.best_mean_episode_reward and self.model_initialized:
        print("SAVING SAVING")
        print(self.mod_file)
        self._last_save = self.t
        self.saver.save(self.session, "/home/dguillory/workspace/homework/models/" + self.mod_file + ".ckpt")
        """
        tf.saved_model.simple_save(self.session, "/tmp/" + self.mod_file,
                                   inputs={"obs_t_ph": self.obs_t_ph},
                                   outputs={"base_q": self.base_q})

        shutil.rmtree("/home/dguillory/workspace/homework/test_models/" + self.mod_file)
        shutil.move("/tmp/" + self.mod_file, "/home/dguillory/workspace/homework/test_models" + self.mod_file)
        """
        print("Done Saving")


    if len(episode_rewards) > 100:
      self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)

    if self.t % self.log_every_n_steps == 0 and self.model_initialized:
      print("Timestep %d" % (self.t,))
      print("mean reward (100 episodes) %f" % self.mean_episode_reward)
      print("best mean reward %f" % self.best_mean_episode_reward)
      print("episodes %d" % len(episode_rewards))
      print("exploration %f" % self.exploration.value(self.t))
      print("learning_rate %f" % self.optimizer_spec.lr_schedule.value(self.t))
      if self.start_time is not None:
        print("running time %f" % ((time.time() - self.start_time) / 60.))

      self.start_time = time.time()

      sys.stdout.flush()

      self.log_data.append({"timestep": self.t, "mean": self.mean_episode_reward, "best": self.best_mean_episode_reward})

      with open(self.rew_file, 'wb') as f:
        pickle.dump(self.log_data, f, pickle.HIGHEST_PROTOCOL)

def learn(*args, **kwargs):
  alg = QLearner(*args, **kwargs)
  while not alg.stopping_criterion_met():
    alg.step_env()
    # at this point, the environment should have been advanced one step (and
    # reset if done was true), and self.last_obs should point to the new latest
    # observation
    alg.update_model()
    alg.log_progress()

