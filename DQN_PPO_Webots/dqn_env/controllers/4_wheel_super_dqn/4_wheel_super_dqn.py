"""4_wheel_super_dqn controller."""

import random
import numpy as np
import torch
from supervisor_controller import FourWheelCarSupervisor
from DQN_Agent import DQNAgent
from utilities import save_loss_plot, save_reward_plot, save_transition_plot, save_x_y_plot, clear_files_in_folder
import sys


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set a seed value
seed_value = 20  # You can change this to any integer

set_seed(seed_value)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clear_files_in_folder()

supervisor_fourwheel = FourWheelCarSupervisor()

agent = DQNAgent(supervisor_fourwheel.observation_space, supervisor_fourwheel.action_space, device)

episode_count = 0
episode_limit = 1000                                  # Change the amount of episodes running here
update_epsilon = 10                                 # Change the amount of updating epsilon here
solved = False

while not solved and episode_count < episode_limit:
    state = supervisor_fourwheel.reset()
    episodeScore = 0
    actions = []
    states = []
    coor_x_y = []
    q_values = []
    next_q_values = []
    expect_q_values = []
    losses = []
    
    for step in range(supervisor_fourwheel.steps_per_episode):
        states.append(state)
        action = agent.select_action(state)
        actions.append(action)
        next_state, reward, done, info = supervisor_fourwheel.step([action])
        coor_x_y.append(info)
        

        agent.memory.push(state, action, reward, next_state, done)
        state = next_state
        
        episodeScore += reward
        qv, next_q, expect_q, loss = agent.optimize_model()
        if len(qv) == 0 and len(next_q) == 0 and len(expect_q) == 0 and len(loss) == 0 :
            pass
        else:
            q_values.append(qv[0])
            next_q_values.append(next_q[0])
            expect_q_values.append(expect_q[0])
            losses.append(loss[0]) 

        if done:
            supervisor_fourwheel.reset_pedestrian()
            break            
        
    if episode_count % update_epsilon == 0:
        agent.epsilon = max(agent.EPSILON_END, agent.epsilon * agent.EPSILON_DECAY)
    else:
        agent.epsilon = agent.epsilon

    # print(agent.epsilon)
    
    supervisor_fourwheel.pedestrian.getField('customData').setSFString('reset')
    supervisor_fourwheel.target_first_touch = False
    supervisor_fourwheel.out_of_road = False
    supervisor_fourwheel.hit_the_box = False
    supervisor_fourwheel.hit_the_pedestrian = False
    supervisor_fourwheel.lanes_input = []
    supervisor_fourwheel.previous_distance_to_target = 2.733


    if episode_count % agent.TARGET_UPDATE == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())

    episode_count += 1
    print(f"Episode {episode_count}, Total Reward: {episodeScore}, Epsilon: {agent.epsilon:.2f}")
    
    save_transition_plot(states,actions,episode_count)
    save_x_y_plot(coor_x_y, episode_count)
    save_loss_plot(q_values, next_q_values, expect_q_values, losses, episode_count)
    save_reward_plot(episodeScore, episode_count)
    
    q_values.clear()
    next_q_values.clear()
    expect_q_values.clear()
    losses.clear()

    
