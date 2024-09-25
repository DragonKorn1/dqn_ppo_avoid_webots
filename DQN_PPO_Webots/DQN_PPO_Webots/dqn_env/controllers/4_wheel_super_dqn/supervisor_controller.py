import numpy as np

from deepbots.supervisor import CSVSupervisorEnv
from utilities import normalize_to_range
from math import sqrt

class FourWheelCarSupervisor(CSVSupervisorEnv):
    """
    FourwheelCarsupoervisor acts as an environment having all the appropriate methods such as get_reward().

    Taken from https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py and modified for Webots.
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves forwards and backwards. The pendulum
        starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's
        velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem described
        by Barto, Sutton, and Anderson
    Observation:
        Type: Box(7)
        Num	 Observation                        Min            Max
        0   front robot distance sensor        0 (0 m)       1000 (4 m)
        1   left steering Position             -0.7 Radian   0.7  Radian
        2   right steering Position            -0.7 Radian   0.7  Radian
        3   front left wheel driving velocity  0 radian/s    10   radian/s
        4   front right wheel driving velocity 0 radian/s    10   radian/s
        5   rear left wheel driving velocity   0 radian/s    10   radian/s
        6   rear right wheel driving velocity   0 radian/s    10   radian/s

    Actions:
        Type: Discrete(4)
        Num	Action
        0	Accelerate
        1	Brake
        2  Turn left
        3  Turn Right

        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is
        pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the
        cart underneath it
    Reward:
        Initial Reward is + 500 and reduce -1 for each step
    Starting State:
        [703.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    Episode Termination:
        - the car moves outside of the road (the robot will fail)
        - the car crashes the box (the distance sensor detects less than 25 (10 cm))
        - Episode length is greater than 400 steps
        - Solved Requirements (the reward is always positive over 10 episoide)
    """
    def __init__(self):
        """
        References to robot and the pole endpoint are initialized here, used for building the observation.
        When in test mode (self.test = True) the agent stops being trained and picks actions in a non-stochastic way.
        """
        super().__init__()
        self.observation_space = 14
        self.action_space = 4
        
        self.robot = self.getFromDef("FOUR-WH-ROBOT")
        self.target = self.getFromDef("TARGET")
        self.obstacle = self.getFromDef("OBS")
        self.pedestrian = self.getFromDef("PEDESTRIAN")        
        
        self.message_received = None # Variable to save the messages received from the robot
        
        self.steps_per_episode = 150 # How many steps to run each episode (changing this messes up the solved condition)
        self.episode_score = 0 # Score Accumulated during an episode
        self.episode_score_list = [] # A list to save all the episode scores, used to check if task is solved
        self.lanes_input = []
        self.target_first_touch = False
        self.hit_the_box = False
        self.out_of_road = False
        self.hit_the_pedestrian = False
        self.robot_reached_1_5m = False
        self.previous_distance_to_target = 2.784
        
    def check_robot_position(self):
        robot_x_position = round(self.robot.getPosition()[0], 3)
        if robot_x_position >= 1.65:
            self.robot_reached_1_5m = True
            self.pedestrian.getField("customData").setSFString("start_moving")
            # print("Sent start_moving")

    def reset_pedestrian(self):
        if self.pedestrian:
            self.pedestrian.getField('customData').setSFString('reset')
            # print("Sent Reset")

    def get_observations(self):
        """
        This get_observation implementation builds the required observation for the CartPole problem.
        All values apart from pole angle are gathered here from the robot and target objects.
        The distance sensor, the two-front-wheeled steering positions ,and the four-wheeled driving velocity value are taken from the message sent by the robot.
        The distance sensor is normalized appropriately to [0,1]
        The steering positions are normalized appropriately to [-1, 1]
        The driving velocity are normalized appropriately to [0, 1]

        :return: Observation: [distance sensor, left steering Position, right steering Position, front left wheel driving velocity, front right wheel driving velocity, rear left wheel driving velocity,rear right wheel driving velocity]
        :rtype: list
        """
        self.check_robot_position()
        self.message_received = self.handle_receiver() # update message received from robot, which contains distance sensor, left steering Position, right steering Position, front left wheel driving velocity, front right wheel driving velocity, rear left wheel driving velocity,rear right wheel driving velocity        
        
        # print(self.message_received)
        
        if self.message_received is not None:
        
            distance_sensor_norm = round(normalize_to_range(float(self.message_received[0]), 0.0, 1000.0, 0.0, 1.0), 3)
            steer_left_pos_norm = round(normalize_to_range(float(self.message_received[1]), -0.5, 0.5, -1.0, 1.0), 3)
            steer_right_pos_norm = round(normalize_to_range(float(self.message_received[2]), -0.5, 0.5, -1.0, 1.0), 3)
            front_left_wheel_norm = round(normalize_to_range(float(self.message_received[3]), 0.0, 10, 0.0, 1.0), 3)
            front_right_wheel_norm = round(normalize_to_range(float(self.message_received[4]), 0.0, 10, 0.0, 1.0), 3)
            rear_left_wheel_norm = round(normalize_to_range(float(self.message_received[5]), 0.0, 10, 0.0, 1.0), 3)
            rear_right_wheel_norm = round(normalize_to_range(float(self.message_received[6]), 0.0, 10, 0.0, 1.0), 3)
        else:
            
            distance_sensor_norm = float(0.703)
            steer_left_pos_norm = float(0.0)
            steer_right_pos_norm = float(0.0)
            front_left_wheel_norm = float(0.0)
            front_right_wheel_norm = float(0.0)
            rear_left_wheel_norm = float(0.0)
            rear_right_wheel_norm = float(0.0)
        # get the position of robot's distance
        robot_x_position = self.robot.getPosition()[0]
        robot_y_position = self.robot.getPosition()[1]
        robot_z_posotion = self.robot.getPosition()[2]
        # get the position of target's distance
        target_x_position = self.target.getPosition()[0]
        target_y_position = self.target.getPosition()[1]
        target_z_position = self.target.getPosition()[2]
        # get the postion of the obstacle
        obs_x_position = self.obstacle.getPosition()[0]
        obs_y_position = self.obstacle.getPosition()[1]
        obs_z_position = self.obstacle.getPosition()[2]
        # get the position of the pedestrian 
        pes_x_position = self.pedestrian.getPosition()[0]
        pes_y_position = self.pedestrian.getPosition()[1]

        robot_rotation = self.robot.getField("rotation").getSFRotation()
        _, _, _, robot_angle = robot_rotation
        
        # Normalize the robot coordinates
        robot_x_norm = round(normalize_to_range(float(robot_x_position), 0.0, 3.0, 0.0, 1.0), 3)
        robot_y_norm = round(normalize_to_range(float(robot_y_position), -0.5, 0.5, -1.0, 1.0), 3)

        #normalize the distance between the robot and the target
        distance_r_tar = round(sqrt(((robot_x_position - target_x_position) ** 2) + ((robot_y_position - target_y_position) ** 2)),2)
        distance_r_tar_norm = round(normalize_to_range(float(distance_r_tar), 0.0, 3.5, 0.0, 1.0), 2)

        #normalize the distance between the robot and the obstacle
        distance_r_obs = round(sqrt(((robot_x_position - obs_x_position) ** 2) + ((robot_y_position - obs_y_position) ** 2) ), 2)
        distance_r_obs_norm = round(normalize_to_range(float(distance_r_obs), 0.0, 3.5, 0.0, 1.0), 2)

        #normalize the distance between the robot and the 
        distance_r_pes = round(sqrt(((robot_x_position - pes_x_position) ** 2 + ((robot_y_position - pes_y_position) ** 2))), 2)
        distance_r_pes_norm = round(normalize_to_range(float(distance_r_pes), 0.0, 3.5, 0.0, 1.0))

        robot_angle_norm = round(normalize_to_range(float(robot_angle), -3.14, 3.14, -1.0, 1.0), 3)

        if int(self.message_received[7]) < int(self.message_received[8]):
            self.lanes_input = [1.0,0.0,0.0] # left lane
        elif int(self.message_received[7]) == int(self.message_received[8]) and int(self.message_received[7]) > 0 and int(self.message_received[8]) > 0:
            self.lanes_input = [0.0,1.0,0.0] # center lane
        elif int(self.message_received[7]) > int(self.message_received[8]):
            self.lanes_input = [0.0,0.0,1.0] # right lane
        else:
            pass
        
        # print(self.lanes_input)
        # normalize the box detection to the input of the nueral network
        box_detected = 1.0 if self.message_received[9] == 'True' else 0.0

        # normalize the pedestrian detection to the input of the nueral network
        pedestrian_detected = 1.0 if self.message_received[10] == 'True' else 0.0

        # return [robot_x_position, robot_y_position, robot_z_posotion, target_x_position, target_y_position, target_z_position,
                # distance_sensor_norm, steer_left_pos_norm, steer_right_pos_norm, front_left_wheel_norm, front_right_wheel_norm,
                # rear_left_wheel_norm, rear_right_wheel_norm]
                
        # return [robot_x_norm, robot_y_norm ,distance_sensor_norm, steer_left_pos_norm, 
        #         front_left_wheel_norm, distance_r_tar_norm, distance_r_obs_norm, 
        #         self.lanes_input[0], self.lanes_input[1], self.lanes_input[2], detected, robot_angle_norm]
        
        return [robot_x_norm, robot_y_norm, distance_sensor_norm, 
                steer_left_pos_norm, front_left_wheel_norm, 
                distance_r_tar_norm, distance_r_obs_norm, distance_r_pes_norm,
                self.lanes_input[0], self.lanes_input[1], self.lanes_input[2], 
                box_detected, pedestrian_detected, robot_angle_norm]


        
    def get_default_observation(self):
        """
        Simple implementation returning the default observation which is a zero vector in the shape
        of the observation space.
        :return: Starting observation zero vector
        :rtype: list
        """
        return [0.0 for _ in range(self.observation_space)]
    
    def get_reward(self, action=None):
        """
        Reward is -1 for each step taken, including the termination step.
        :param action: Not used, defaults to None
        :type action: None, optional
        :rtype: int
        """
        robot_x_position = round(self.robot.getPosition()[0],3)
        robot_y_position = round(self.robot.getPosition()[1],3)
        target_x_position = round(self.target.getPosition()[0],3)
        target_y_position = round(self.target.getPosition()[1],3)
        obs_x_position = round(self.obstacle.getPosition()[0],3)
        obs_y_position = round(self.obstacle.getPosition()[1],3)       
        ped_x_position = round(self.pedestrian.getPosition()[0],3)
        ped_y_position = round(self.pedestrian.getPosition()[1],3)

        distance_r_tar = round(sqrt(((robot_x_position - target_x_position) ** 2) + ((robot_y_position - target_y_position) ** 2)),3)
        distance_r_obs = round(sqrt(((robot_x_position - obs_x_position) ** 2) + ((robot_y_position - obs_y_position) ** 2) ), 3)
        distance_r_ped = round(sqrt(((robot_x_position - ped_x_position) ** 2) + ((robot_y_position - ped_y_position) ** 2) ), 3)
        
        reward = 0
        reward_out = 0
        reward_goal = 0
        reward_bonus = 0
        reward_hit_box = 0
        reward_hit_ped = 0
        reward_close = 0

        # Reward for getting closer to the target
        distance_improvement = self.previous_distance_to_target - distance_r_tar
        reward_distance = distance_improvement * 15  # Adjust this multiplier as needed
        if reward_distance == 0.0: # If the robot does not move.
            reward_distance -= 0.05

        # Penalty for being out of road
        if abs(robot_y_position) > 0.43:
            # out_of_road_penalty = max((abs(robot_y_position) - 0.43) * 5, 2)
            out_of_road_penalty = min(abs(2.733 - distance_r_tar) * 10, 2)
            self.out_of_road = True
            reward_out = out_of_road_penalty

        # Reward for reaching target
        if robot_x_position >= 3.50 and not self.target_first_touch:
            self.target_first_touch = True
            reward_goal = 20
            # if robot_y_position <= 0.07 and robot_y_position > 0.3:
            #     reward_bonus =  10 - (robot_y_position + 0.3) * 43.4783
            # else:
            #     reward_bonus =  (robot_y_position + 0.3) * 43.4783
            reward_bonus = 0

        # Penalty for hitting the box
        if (robot_x_position >= 2.165 and robot_x_position <= 2.77) and robot_y_position >= 0.00 and not self.hit_the_box:
            hit_box_penalty = 7  
            self.hit_the_box = True
            reward_hit_box = hit_box_penalty

        # Penalty for being very close to the obstacle
        if distance_r_obs < 0.6:
            close_to_obstacle_penalty = (0.6 - distance_r_obs) * 1
            reward_close = close_to_obstacle_penalty

        # Penalty if the robot hit the pedestrian
        if distance_r_ped < 0.25:
            hit_pedestrian_penalty = 8
            self.hit_the_pedestrian = True
            reward_hit_ped = hit_pedestrian_penalty
        
        # print(f'reward_distance - {reward_distance}')
        # print(f'reward_out - {reward_out}')
        # print(f'reward_goal - {reward_goal}')
        # print(f'reward_bonus - {reward_bonus}')
        # print(f'reward_hit_box - {reward_hit_box}')
        # print(f'reward_hit_ped - {reward_hit_ped}')
        # print(f'distance_r_ped - {distance_r_ped:.3f}')
        # print(f'reward_close - {reward_close}')

        # Small penalty for time
        reward = reward_distance - reward_out + reward_goal + reward_bonus - reward_hit_box - reward_close - reward_hit_ped

        # reward = reward_distance *0.2
        # - reward_out * 0.2
        # + reward_goal * 0.3
        # + reward_bonus * 0.1
        # - reward_hit_box * 0.3
        # - reward_close *0.05
        # - reward_hit_ped * 0.4

        # reward_clipped = np.clip(reward, -7, 7)    


        # Update previous distance
        self.previous_distance_to_target = distance_r_tar

        # if not self.first_reward:
        #     reward -= 1
        #     self.first_reward = True
        
        # if robot_y_position > 0.43:
        #     # print('out of road left')
        #     # return -2
        #     reward -= 3
            
        # if robot_y_position < -0.43:
        #     # print('out of road left')
        #     # return -1
        #     reward -= 1
            
        # if robot_x_position >= 2.7 and robot_y_position <= -0.07 and not self.target_first_touch:
        #     self.target_first_touch = True
        #     # print('reach the target')
        #     # return 10
        #     reward += 10
        
        # if  distance_r_tar < 2.0 and distance_r_tar >= 1.0 and not self.two_m_target_first_touch :
        #     # print('distance to goals is less than two meters')
        #     self.two_m_target_first_touch = True
        #     # return 1
        #     reward += 2
        
        # if  distance_r_tar < 1.0 and distance_r_tar >= 0.5 and not self.one_m_target_first_touch:
        #     # print('distance to goals is less than one meters')
        #     self.one_m_target_first_touch = True
        #     # return 2
        #     reward += 5
        
        # if distance_r_tar < 0.5 and not self.half_m_target_first_touch:
        #     # print('distance to goals is less than 0.5 meters')
        #     self.half_m_target_first_touch = True
        #     # return 3
        #     reward += 6
                        
        # # if robot_x_position - self.robot_x_pos_old > 0.002:
        #     # self.robot_x_pos_old = robot_x_position
        #     # return -2
        
        # # hit the box
        # if robot_x_position >= 2.05 and robot_y_position >= -0.007 and not self.hit_the_box:
        #     self.hit_the_box = True
        #     # return -4
        #     reward -= 7
        
        # # if distance_r_tar < 0.5 and distance_r_obs < 0.32:
        #     # return -2
        #     # reward -= 2

            
        return reward 
        
    # def step(self, action):
        # observation, reward, is_done, info = self.step(action)
        # return observation, reward, is_done, info
            
    def is_done(self):
        """
        An episode is done if the car is on the target, the car is off the road , or the car crashes the box 

        :return: True if termination conditions are met, False otherwise
        :rtype: bool
        """
        if self.message_received is not None:
            distance_sensor_done = (round(float(self.message_received[0]),2))
            
        else:
            distance_sensor_done = 703.0
        
            # if the car reaches the target, the episode stops
        if self.target_first_touch:
            return True
                
            # if the robot is out of the road (the robot will go out of the lane), the episode stops
        if self.out_of_road:
            return True
            
            # if the robot crashes the box by detecting sensor ( < 8cm), the episode stops     
        if distance_sensor_done < 8.0:
            return True
        

        # if the robot crashes the pedestrian ,the episode stops     
        if self.hit_the_pedestrian:
            return True

        # if the robot crashes the box by hitting the box , the episode stops     
        if self.hit_the_box:
            return True


        return False
            
    def get_info(self):
        """
        Dummy implementation of get_info.
        
        robot_x_position = round(self.robot.getPosition()[0],3)
        robot_y_position = round(self.robot.getPosition()[1],3)
        
        
        :return: None
        :rtype: None
        """
        
        robot_x_position = round(self.robot.getPosition()[0],3)
        robot_y_position = round(self.robot.getPosition()[1],3)
        pes_x_position = round(self.pedestrian.getPosition()[0], 3)
        pes_y_position = round(self.pedestrian.getPosition()[1], 3)
        info = [robot_x_position, robot_y_position, pes_x_position, pes_y_position]
        
        return info

    def render(self, mode="human"):
        """
        Dummy implementation of render.
        """
        pass

    def solved(self):
        if len(self.episode_score_list) > 5:         # over 10 trials thus for
            if np.mean(self.episode_score_list[-5:]) > 11: # if the reward scores are over 0 in 10 successive episode ==> the case is solved.
                return True
        
        return False
        """
        This method checks whether the CartPole task is solved, so training terminates.
        Solved condition requires that the average episode score of last 10 episodes is over 0.

        :return: True if task is solved, False otherwise
        :rtype: bool
        """
        # return True
        
        # if len(self.episode_score_list) > 10:
        # if np.mean(self.episode_score_list[-10:] > 0): 
                # return True
                
