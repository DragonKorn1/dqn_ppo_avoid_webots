import numpy as np
from deepbots.robots import CSVRobot

class FourWheeledCar(CSVRobot):

    def __init__(self):
    
        # the constructor gets the distance sensor and position sensor references and enables it and also initialize the wheels and steering.
    
        super().__init__()
        # initialize and enable distance sensor
        self.distance_sensor = self.getDevice('distance_sensor') 
        self.distance_sensor.enable(self.timestep)
        # initialize and enable steering position sensors
        self.steering_left_position_sensor = self.getDevice('left_steer_sensor')
        self.steering_left_position_sensor.enable(self.timestep)
        self.steering_right_position_sensor = self.getDevice('right_steer_sensor')
        self.steering_right_position_sensor.enable(self.timestep)
        #initialize and enable driving position sensors
        self.wheels_front_left_position_sensor = self.getDevice('front_left_wheel_sensor')
        self.wheels_front_left_position_sensor.enable(self.timestep)
        self.wheels_front_right_position_sensor = self.getDevice('front_right_wheel_sensor')
        self.wheels_front_right_position_sensor.enable(self.timestep)
        self.wheels_rear_left_position_sensor = self.getDevice('rear_left_wheel_sensor')
        self.wheels_rear_left_position_sensor.enable(self.timestep)
        self.wheels_rear_right_position_sensor = self.getDevice('rear_right_wheel_sensor')
        self.wheels_rear_right_position_sensor.enable(self.timestep)
        #initialize and enable camera and recognition
        self.camera = self.getDevice('camera')
        self.camera.enable(self.timestep)
        self.camera.recognitionEnable(self.timestep)
        # self.camera.getRecognitionObjects()
        
        self.gps = self.getDevice('gps')
        self.gps.enable(self.timestep)

        # self.emitter_robot = self.getDevice('emitter')
        # self.start_signal_sent = False
        # self.initial_position = 0.285

        # initialize the wheel and steering motors
        self.wheels = [None for _ in range(4)]
        self.steer = [None for _ in range(2)]
        self.wheels[0] = self.getDevice('front_left_wheel')
        self.wheels[1] = self.getDevice('front_right_wheel')
        self.wheels[2] = self.getDevice('rear_left_wheel')
        self.wheels[3] = self.getDevice('rear_right_wheel')

        self.steer[0] = self.getDevice('left_steer')
        self.steer[1] = self.getDevice('right_steer')
    
        self.reset_speed_motors()
        self.reset_position_steer()
        self.motor_speed = 12.0
        self.steer_position = 0.0


    def reset_speed_motors(self):
    
    # This method initialize the four wheels, storing the references inside a list and setting the starting positions and velocities.
    # setting the initial postion and velocity of the wheels
        self.wheels[0].setPosition(float('inf'))
        self.wheels[0].setVelocity(12.0)
        # self.wheels[0].setVelocity(0.0)
        self.wheels[1].setPosition(float('inf'))
        self.wheels[1].setVelocity(12.0)
        # self.wheels[1].setVelocity(0.0)
        self.wheels[2].setPosition(float('inf'))
        self.wheels[2].setVelocity(12.0)
        # self.wheels[2].setVelocity(0.0)
        self.wheels[3].setPosition(float('inf'))
        self.wheels[3].setVelocity(12.0)
        # self.wheels[3].setVelocity(0.0)
        
    def reset_position_steer(self):
        
        self.steer[0].setPosition(0.0)
        self.steer[1].setPosition(0.0)


        # for i in range(2):
            # self.wheels[i].setPosition(0.0)
            
    def create_message(self):
    
        """
        This method packs the robot's observation into a list of strings to be sent to the supervisor.
        The message contains only the Distance Sensor value, ie. the distance between the robot and the obstacles.
        From Webots documentation:
        'The getValue function returns the most recent value measured by the specified position sensor. Depending on
        the type, it will return a value in radians (angular position sensor) or in meters (linear position sensor).'

        :return: A list of strings with the robot's observations.
        :rtype: list
        """
        x = None
        y = None
        w = None
        h = None
        
        current_position = self.gps.getValues()

        # print(current_position)

        # current_position = self.getFromDef("FOUR-WH-ROBOT").getPosition()
        # x_distance_moved = current_position[0] - self.initial_position

        # if x_distance_moved >= 2.0 and not self.start_signal_sent:
        #     self.emitter_robot.send("start_moving".encode('utf-8'))
        #     self.start_signal_sent = True
        #     print("Sent start signal moving")


        def detect_lanes(img, height, width):
            
            roi_height = height // 2
            roi_y = height - roi_height
            
            #List to store left and right lane points
            left_points = []
            right_points = []
            
            #Scan the region of interest
            for y in range(roi_y, height, 5):
                left_x = None
                right_x = None
                for x in range(width):
                    # get pixel color (the lane on dark background)
                    r = self.camera.imageGetRed(img, width, x, y)
                    g = self.camera.imageGetGreen(img, width, x, y)
                    b = self.camera.imageGetGreen(img, width, x, y)
                    
                    # Check if pixel is bright(white) (part of a lane marking)
                    if r > 200 and g > 200 and b > 200:
                        if x < width // 2 and (left_x is None or x < left_x):
                            left_x = x
                        elif x >= width // 2 and (right_x is None or x > right_x):
                            right_x = x
                            
                if left_x is not None:
                    left_points.append((left_x, y))
                if right_x is not None:
                    right_points.append((right_x, y))
                    
            return len(left_points), len(right_points)
    
        def detect_obstacle(recognitions, width, height):
            
            box_detected = False
            pedestrian_detected = False

            for obj in recognitions:
                # print(f'model - {obj.getModel()}')
                if obj.getModel() == 'CustomWoodenBox':
                
                # find the center and the size of the box
                    box_detected = True
                    # pos_on_image = obj.getPositionOnImage()
                    # size_on_image = obj.getSizeOnImage()
                    # x = int(pos_on_image[0] - size_on_image[0]/2)
                    # y = int(pos_on_image[1] - size_on_image[1]/2)
                    # w = int(size_on_image[0])
                    # h = int(size_on_image[1])
                    
                else:
                    # x = -1
                    # y = -1
                    # w = -1
                    # h = -1
                    box_detected = False
                
                if obj.getModel() == 'mini_pedestrian':
                    pedestrian_detected = True
                else:
                    pedestrian_detected = False


            return box_detected, pedestrian_detected
            
        img = self.camera.getImage()
        left_points, right_points = detect_lanes(img, self.camera.getHeight(), self.camera.getWidth())
        box_detected, pedestrian_detected = detect_obstacle(self.camera.getRecognitionObjects(), self.camera.getWidth(), self.camera.getHeight())
                
        message = [str(self.distance_sensor.getValue()), # get Vaีlue from distance sensor
                   str(self.steering_left_position_sensor.getValue()),str(self.steering_right_position_sensor.getValue()), #get Value from steering position sensors
                   str(self.wheels_front_left_position_sensor.getValue()), str(self.wheels_front_right_position_sensor.getValue()), # get Value from front driving position sensors
                   str(self.wheels_rear_left_position_sensor.getValue()), str(self.wheels_rear_right_position_sensor.getValue()), # get Value from rear driving position sensors
                   str(left_points), str(right_points), str(box_detected), str(pedestrian_detected)] # get Values from camera
        
        return message
    
    def use_message_data(self, message):
        
        """
        This method unpacks the supervisor's message, which contains the next action to be executed by the robot.
        In this case it contains an integer denoting the action, either 0 , 1 , 2 and 3. with 0 being forward, 
        1 being backward, 2 to turn left and 3 to turn right movement. The corresponding motor_speed value is applied to the wheels.

        :param message: The message the supervisor sent containing the next action.
        :type message: list of strings
        """
        action = message[0]

        if action == "RESET":
            self.reset_speed_motors()
            self.reset_position_steer()
            self.motor_speed = 12.0
            self.steer_position = 0.0
            # print("Reset")
            return

        action = int(action) 

        assert action == 0 or action == 1 or action == 2 or action == 3 or action == 4, "4 Wheeled Car controller gets incorrect action value:" + str(action)
        
        if action == 0:
            self.motor_speed += 0.10
            # print("Motor Speed = {:.2f}".format(self.motor_speed))
            if (self.motor_speed >= 15.0):
                self.motor_speed = 15.0

            for i in range(len(self.wheels)):
                self.wheels[i].setPosition(float('inf'))
                self.wheels[i].setVelocity(self.motor_speed)
            
        elif action == 1:
            self.motor_speed -= 0.10
            # print("Motor Speed = {:.2f}".format(self.motor_speed))
            if (self.motor_speed <= 0.0):
                self.motor_speed = 0.0

            for i in range(len(self.wheels)):
                self.wheels[i].setPosition(float('inf'))
                self.wheels[i].setVelocity(self.motor_speed)
                
        elif action == 2:
            # print("Steering Position = {:.2f}".format(self.steer_position))
            self.steer_position += 0.10
            if (self.steer_position >= 0.6):
                self.steer_position = 0.6
            
            for i in range(len(self.steer)):
                self.steer[i].setPosition(self.steer_position)

        else:
            # print("Steering Position = {:.2f}".format(self.steer_position))
            self.steer_position -= 0.10
            if (self.steer_position <= -0.6):
                self.steer_position = -0.6
            
            for i in range(len(self.steer)):
                self.steer[i].setPosition(self.steer_position)

        # else: 
        #     # print("Steering Position = {:.2f}".format(self.steer_position))
        #     pass

robot_controller = FourWheeledCar()
robot_controller.run()
    
    