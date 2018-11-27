import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import os
import uuid
import argparse

import dqn
from dqn_utils import *
from atari_wrappers import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def build_arg_parser():
    parser = argparse.ArgumentParser("Run atari games using DQN Networks")
    parser.add_argument("--game", required=True, choices=["pong", "zaxxon", "spaceinvaders", "airraid",
                                           "assault", "beamrider", "carnival", "phoenix",
                                           "riverraid"])
    parser.add_argument("--ddqn", action="store_true")
    parser.add_argument("--model", default="atari", choices=["visual", "atari", "frozen", "finetune"])
    parser.add_argument("--source", choices=["pong", "zaxxon", "spaceinvaders", "airraid",
                                             "assault", "beamrider", "carnival", "phoenix",
                                             "riverraid"])
    parser.add_argument("--explore", default=1.0, type=float)
    parser.add_argument("--cuda", default="3", type=str)
    return parser


def atari_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out

def atari_visual_freeze(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        out = tf.stop_gradient(out, name="freeze_image")
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
            #dummy = layers.fully_connected(out, num_outputs=18,         activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out


def atari_learn(env,
                session,
                num_timesteps,
                task="TEST_MODEL",
                model=atari_model,
                source=None,
                explore=1.0):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0,   explore),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    uid = str(uuid.uuid4())
    preload = False
    if source is not None:
        rew_file = task + "_from_" + source + uid + ".pkl"
        mod_file = task + "_from_" + source + uid 
        preload = True
    else:
        rew_file =  task  + uid + ".pkl"
        mod_file =  task + uid 

    print("MODEL FILE")
    print(mod_file)

    dqn.learn(
        env=env,
        q_func=model,
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10,
        double_q=True,
        rew_file=rew_file,
        mod_file=mod_file,
        target_task=task,
        src_task=source,
        preload=preload
    )
    env.close()

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session

def get_env(task, seed):
    #env = gym.make('PongNoFrameskip-v4')
    #env = gym.make('ZaxxonNoFrameskip-v4')
    env = gym.make(task)

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/hw3_vid_dir2/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)

    return env

def main():
    
    args = build_arg_parser().parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    if args.game == "pong":
        task = 'PongNoFrameskip-v4'
    elif args.game == "assault":
        task = 'AssaultNoFrameskip-v4'
    elif args.game == 'zaxxon':
        task = 'ZaxxonNoFrameskip-v4'
    elif args.game == 'spaceinvaders':
        task = 'SpaceInvadersNoFrameskip-v4'
    elif args.game == 'airraid':
        task = 'AirRaidNoFrameskip-v4'
    elif args.game == 'beamrider':
        task = 'BeamRiderNoFrameskip-v4'
    elif args.game == 'carnival':
        task = "CarnivalNoFrameskip-v4"
    elif args.game == "phoenix":
        task = "PhoenixNoFrameskip-v4"
    elif args.game == "riverraid":
        task = "RiverraidNoFrameskip-v4"

    
    second_task = None
    if args.model == "visual":
        if args.source == "pong":
            second_task = 'PongNoFrameskip-v4'
        elif args.source == "assault":
            second_task = 'AssaultNoFrameskip-v4'
        elif args.source == 'zaxxon':
            second_task = 'ZaxxonNoFrameskip-v4'
        elif args.source == 'spaceinvaders':
            second_task = 'SpaceInvadersNoFrameskip-v4'
        elif args.source == 'airraid':
            second_task = 'AirRaidNoFrameskip-v4'
        elif args.source == 'beamrider':
            second_task = 'BeamRiderNoFrameskip-v4'
        elif args.source == 'carnival':
            second_task = "CarnivalNoFrameskip-v4"
        elif args.source == "phoenix":
            second_task = "PhoenixNoFrameskip-v4"
        elif args.source == "riverraid":
            second_task = "RiverraidNoFrameskip-v4"
        
        model = atari_visual_freeze 
    else:
        model = atari_model


    # Get Atari games.
    #task = gym.make('PongNoFrameskip-v4')
    #task = gym.make('Zaxxon-v4')

    # Run training
    seed = random.randint(0, 9999)
    print('random seed = %d' % seed)
    env = get_env(task, seed)
    session = get_session()
    atari_learn(env, session, num_timesteps=2e8, task=task, model=model, source=second_task, explore=args.explore)

if __name__ == "__main__":
    main()
