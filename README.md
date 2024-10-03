## DQN PPO Pedestrian Avoidance on Webots

Pedestrian Avoidance Simulation on Webots with deepbots.

Webots is released under the terms of the [Apache 2.0 license agreement](https://cyberbotics.com/doc/guide/webots-license-agreement).

# Prerequisite: (according to [Webots User Guide](https://cyberbotics.com/doc/guide/system-requirements))

Minimum Computer Specification: 
 - CPU: 2 GHz dual core CPU clock speed 
 - RAM: 2 GB of RAM.
 - GPU: NVIDIA or AMD OpenGL at least 512 MB of RAM. (NVIDIA GPUs are recommended)
 - CUDA version: 11.8
 - CUDNN version: 8.9.7

# Installation Guide and Simulation procedures
If you would like to run this simulation. This is the steps to install before the simulation.

***The instruction video how to install and run the simulation is displayed from this [link](https://youtu.be/C7BN6PAKfh8)

1. Download Webots 2023b  ([Windows](https://cyberbotics.com/doc/guide/installation-procedure#installation-on-windows), [Linux](https://cyberbotics.com/doc/guide/installation-procedure#installation-on-linux), and [MacOS](https://cyberbotics.com/doc/guide/installation-procedure#installation-on-macos)).
2. Install Webots 2023b.
3. Install Visual Studio Code from this [link](https://code.visualstudio.com/)
4. Install [python](https://www.python.org/downloads/) (version 3.8 or later)
5. Install pip package installer from this [link](https://pip.pypa.io/en/stable/installation)
6. Install numpy and opencv-python package installer 
```
    pip3 install numpy opencv-python 
```
7. Install PyTorch from this [link](https://pytorch.org)
8. Install Jupeyter Lab, Jupyter Notebook, Voila, and IpyKernel 
```
    pip3 install jupyterlab notebook voila ipykernel
```
9. Install deepbots package 
```
    pip3 install deepbots
```
# Code structure
The structure of the simulation code is classified in 3 main parts.
 - worlds: the worlds folder contains the simulation environment and controller file configurations to simulate the scenario.
 - controllers: the controller folder includes the Robot-Supervisor scheme coding to manipulate both the robot's and the supervisor's configuration and learning algorithms as well as the pedestrian robot control. These are seperated into 3 parts.
   - 4_wheel_supervisor: this folder contains the supervisor configurations and each algorithm code for the robot to learn the scenario.
   - mini_pedestrian: the folder involve the pedestrian control of the pedestrian robot in the scenario.
   - super_robot: this folder includes the robot's configuration file.
 - proto: this proto folder contains the physical configuration of the obstacle and pedestrian and also segmentation settings for the camera.

