'''An example to show how to set up an pommerman game programmatically'''
import os, shutil
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pommerman
from pommerman import agents
from pommerman.agents.nn_agent import NNAgent
from pommerman.agents.cnn_model import CNNBatchNorm

import torch

def setup_episode_dirs(base_dir, episode_num):
    '''
    Creates directories to record episode playouts.
    Clears any information that was previously there.
    '''
    png_dir= base_dir + '/png_logs/' + str(episode_num)
    json_dir = base_dir + '/json_logs/' + str(episode_num)
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)
    else:
        shutil.rmtree(png_dir)
        if not os.path.exists(png_dir):
            os.makedirs(png_dir)
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    else:
        shutil.rmtree(json_dir)
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)
    print('record png dir:', png_dir)
    print('record json dir:', json_dir)
    return png_dir, json_dir

def main():
    '''Simple function to bootstrap a game.
       
       Use this as an example to set up your training env.
    '''
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    # Create a set of agents (exactly four)    

    shape=(14,11,11)
    n_actions=6
    n_filters_per_layer=64
    n_cnn_layers=4
    nn_model=CNNBatchNorm(input_feature_shape=shape, n_actions=n_actions, n_filters_per_layer=n_filters_per_layer, n_cnn_layers=n_cnn_layers)
    nn_model2=CNNBatchNorm(input_feature_shape=shape, n_actions=n_actions, n_filters_per_layer=n_filters_per_layer, n_cnn_layers=n_cnn_layers)
    nn_path='./trained_agent/skynet_variations/ppo_CNN4_64_6407.pt'
    nn_path2='./trained_agent/skynet_variations/ppo_CNN4_64_955.pt'
    nn_model.load_state_dict(torch.load(nn_path, map_location=lambda storage, loc: storage))
    nn_model2.load_state_dict(torch.load(nn_path2, map_location=lambda storage, loc: storage))
    selection='softmax'
    
    nn_agent = NNAgent(nn_model, action_selection=selection, is_training=False)
    nn_agent2 = NNAgent(nn_model2, action_selection=selection, is_training=False)
    env_id="tournament-v0"
    agent_list = [
        nn_agent,
        #nn_agent2
        agents.SimpleAgent()
        #agents.StaticAgent(),
        #agents.SlowRandomAgentNoBomb(),
        #agents.PlayerAgent(),
        #agents.RandomAgent(),
    ]
    env = pommerman.make(env_id, agent_list)

    base_dir = '.'    
    # Run the episodes just like OpenAI Gym
    for i_episode in range(1):
        png_dir, json_dir = setup_episode_dirs(base_dir,i_episode)
        state = env.reset()
        done = False
        while not done:
            env.render(record_pngs_dir=png_dir, record_json_dir=json_dir)
            #env.save_json(json_dir) #use this instead of env.render to only save JSON files without doing rendering.
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
        print('Episode {} finished'.format(i_episode))
        print("Final Result: ", info)
        pommerman.utility.join_json_state(json_dir, ['skynet6407','skynet955'], finished_at=0, config=env_id, info=info)
    env.close()


if __name__ == '__main__':
    main()
