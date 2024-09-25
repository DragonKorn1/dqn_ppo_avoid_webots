from controller import Supervisor
import math

class CustomPedestrian(Supervisor):
    def __init__(self):
        super(CustomPedestrian, self).__init__()
        self.timeStep = int(self.getBasicTimeStep())
        
        # Get the robot node
        self.robot_node = self.getSelf()
        # Initialize the receiver

        # Get the controller arguments
        controller_args_field = self.robot_node.getField("controllerArgs")
        if controller_args_field:
            controller_args_count = controller_args_field.getCount()
            if controller_args_count > 0:
                try:
                    self.speed = float(controller_args_field.getMFString(0))
                except ValueError:
                    print("Invalid speed argument. Using default.")
                    self.speed = 0.02
            else:
                self.speed = 0.02  # default speed if no argument is provided
        else:
            self.speed = 0.02  # default speed if field doesn't exist

        print(f"Speed: {self.speed}")

        # Step Counter
        self.step_counter = 0

        # Initialize position
        self.initial_position = [2.84, 0.3, 0.15]
        self.x = self.initial_position[0]
        self.y = self.initial_position[1]
        self.z = self.initial_position[2]
        # self.first_touch = False
        self.start_moving = False
        # Store initial contact points (assumed to be with the ground)
        # self.initial_contacts = 0
        
    # def get_contact_point_ids(self):
        # contact_points = self.robot_node.getContactPoints(True)
        # contact_count = 0
        # for point in contact_points:
            # contact_count += 1
                
        # print(contact_count)
            
        # return contact_count

    # def check_collision(self):        
        # print(f'initial_contacts - {self.initial_contacts}')
        # current_contacts = self.get_contact_point_ids()
        # print(f'current_contacts - {current_contacts}')
        # new_contacts = current_contacts - self.initial_contacts
        # print(f'new_contacts - {new_contacts}')
        
        # if new_contacts > 0:
            # print("collision detected")
            # return True 
        
        # return False

    def reset_position(self):
        self.x, self.y, self.z = self.initial_position
        translation_field = self.robot_node.getField("translation")
        if translation_field:
            translation_field.setSFVec3f(self.initial_position)
        else:
            print("Error: Could not find 'Translation' field ")

        self.start_moving = False

    def run(self):
        while self.step(self.timeStep) != -1:
            # if self.receiver.getQueueLength() > 0:
            #     message = self.receiver.getData().decode('utf-8')
            #     self.receiver.nextPacket()
            #     if message == 'start_moving':
            #         self.start_moving = True
            #         print("Received start moving signal")                
            custom_data = self.robot_node.getField("customData").getSFString()
            # print(f'Custom Data from pedestrian - {custom_data}')
            if custom_data == 'start_moving':
                self.start_moving = True
            elif custom_data == 'reset':
                self.reset_position()
                # print("Pedestrian reset Position")

            if self.start_moving:
                self.y += self.speed
                new_position = [self.x, self.y, self.z]
                self.robot_node.getField("translation").setSFVec3f(new_position)


            # if not self.first_touch:
                # self.initial_contacts = self.get_contact_point_ids()
                # self.first_touch = True
            # else: 
                # if self.check_collision():
                    # Collision handling logic here
                    # print("Robot has collided with an object. Stopping.")
                    # break
                # print(self.step_counter)
                # if self.step_counter >= 30:
                #     # Update position
                #     self.y += self.speed
            
                #     # Update robot's position
                #     new_position = [self.x, self.y, self.z]
                #     self.robot_node.getField("translation").setSFVec3f(new_position)

                # self.step_counter += 1


# Create the robot instance and run it
controller = CustomPedestrian()
controller.run()