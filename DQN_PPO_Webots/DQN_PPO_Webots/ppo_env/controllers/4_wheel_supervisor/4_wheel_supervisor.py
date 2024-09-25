import numpy as np
from numpy import convolve, ones, mean
from supervisor_controller import FourWheelCarSupervisor
from PPO_Agent import PPOAgent, Transition
from utilities import save_reward_plot, save_transition_plot, save_x_y_plot, clear_files_in_folder
import sys
import numpy as np
import torch
import random


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

# Call the function to set the seed
set_seed(seed_value)

clear_files_in_folder()

# initializes the FourWheelCarSupervisor class
supervisor_fourwheel = FourWheelCarSupervisor()
# Wrap the four wheel supervior in the custom keyboard printer
# supervisor_key = KeyboardControllerFourWheel(supervisor_fourwheel)
# set the agent to PPO algorithm
agent = PPOAgent(supervisor_fourwheel.observation_space, supervisor_fourwheel.action_space)

episode_count = 0
episode_limit = 1000                    # Change the amount of episodes running here
average_episode_action_probs = []       # Change the amount of updating epsilon here
update_epsilon = 10
rewards = [] # get the results of rewards in all episodes

# Run outer loop until the episode limit is reached or the task is solved
while episode_count < episode_limit:
    state = supervisor_fourwheel.reset() # reset robot and get starting observation
    episodeScore = 0 # initialize reward
    action_probs = [] # This list holds the probability of each chosen action
    actions = [] #get actions in each episode
    states = [] # get states in each episode
    coor_x_y = [] # get x,y coordinates in each episode
    # Inner loop is the episode loop
    for step in range(supervisor_fourwheel.steps_per_episode):
        # In training mode the agent samples from the probability distribution, naturally implementing exploration
        selected_action, action_prob = agent.work(state, type_="selectAction") # OK
        # Save the current selected_action's probability
        states.append(state)
        actions.append(selected_action)
        action_probs.append(action_prob) # OK
        supervisor_fourwheel.is_done_steps = step
        # Step the supervisor to get the current selected_action reward, the new state and whether we reached the
        # done condition
        new_state, reward, done, info = supervisor_fourwheel.step([selected_action])
        
        coor_x_y.append(info)
        # Save the current state transition in agent's memory
        trans = Transition(state, selected_action, action_prob, reward, new_state)
        agent.store_transition(trans)
        
        episodeScore += reward # Deduct the points in every step

        # print(f'accummulated reward in each step - {supervisor_fourwheel.episodeScore:.3f}')
        if done:
            # rewards.append(episodeScore)
            agent.train_step(batch_size = step + 1) 
            supervisor_fourwheel.reset_pedestrian()
            # solved = supervisor_fourwheel.solved() # Check Whether the episode is solved
            break
        
        state = new_state
        # print("step = {}".format(step))
        
    # if supervisor_fourwheel.test: # If test flag is externally set to True, agent is deployed
    #     break
        
    # The average action probability tells us how confident the agent was of its actions.
    # By looking at this we can check whether the agent is converging to a certain policy.
    if episode_count % update_epsilon == 0:
        agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)
        # print(agent.epsilon)
    avgaction_prob = mean(action_probs)
    average_episode_action_probs.append(avgaction_prob)
    # print("Average Action Probability", avgaction_prob)
    save_transition_plot(states,actions,action_probs,episode_count)
    save_x_y_plot(coor_x_y,episode_count)
    episode_count += 1 # Increment episode counter 
    agent.episode_count = episode_count
    print("Episode #", episode_count, "Score:",episodeScore)
    save_reward_plot(episodeScore, episode_count)

    # np.convolve is used as a moving average, see https://stackoverflow.com/a/22621523
    # moving_avg_n = 10
    # plot_data(convolve(supervisor_fourwheel.episode_score_list, ones((moving_avg_n,))/moving_avg_n, mode='valid'),
             # "episode", "episode score", "Episode scores over episodes")
    # plot_data(convolve(average_episode_action_probs, ones((moving_avg_n,))/moving_avg_n, mode='valid'),
             # "episode", "average episode action probability", "Average episode action probability over episodes")

    # if not solved:
        # print("Task is not solved.")

    supervisor_fourwheel.pedestrian.getField('customData').setSFString('reset')
    supervisor_fourwheel.target_first_touch = False
    supervisor_fourwheel.out_of_road = False
    supervisor_fourwheel.hit_the_box = False
    supervisor_fourwheel.hit_the_pedestrian = False
    supervisor_fourwheel.lanes_input = []
    supervisor_fourwheel.previous_distance_to_target = 2.733
    supervisor_fourwheel.reset()
    # else: 
    #     print("Task is solved")
    #     supervisor_fourwheel.close()
    #     break
        
    # if not solved and not supervisor_fourwheel.test:
        # print("Reached episode limit and task was not solved.")
    # else:
        # if not solved:
            # print("Task is not solved, deploying agent for testing...")
        # elif solved:
            # print("Task is solved, deploying agent for testing...")
supervisor_fourwheel.close()

    # print("Press R to reset")
    # state = supervisor_key.reset()
    # supervisor_fourwheel.test = True
    # supervisor_fourwheel.episodeScore = 500
    # while True:
        # selected_action, action_prob = agent.work(state, type_="selectActionMax")
        # state, reward, done, _ = supervisor_key.step([selected_action])
        # supervisor_fourwheel.episodeScore += reward  # Accumulate episode reward
        
        # if done:
            # print("Rewards =", supervisor_fourwheel.episodeScore)
            # supervisor_fourwheel.episodeScore = 500
            # state = supervisor_key.reset()


    