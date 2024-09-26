# DQN_PPO_Pedestrian_avoidance_webots

Pedestrian Avoidance Simulation on Webots with deepbots. This is the prototype code for Pedestrian Avoidance.

Prerequisite: (according to [Webots User Guide](https://cyberbotics.com/doc/guide/system-requirements))

Minimum Computer Specification: 
CPU: 2 GHz dual core CPU clock speed 
RAM: 2 GB of RAM.
GPU: NVIDIA or AMD OpenGL at least 512 MB of RAM. (We reccommed NVIDIA GPU for CUDA)
CUDA version: 11.8
CUDNN version: 8.9.7

If you would like to run this simulation. This is the steps to install before the simulation.

1. Download Webots 2023b  ([Windows](https://cyberbotics.com/doc/guide/installation-procedure#installation-on-windows), [Linux](https://cyberbotics.com/doc/guide/installation-procedure#installation-on-linux), and [MacOS](https://cyberbotics.com/doc/guide/installation-procedure#installation-on-macos)).
2. Install Webots 2023b.
3. Install [python](https://www.python.org/downloads/) (version 3.8 or later)
5. Install pip package installer from this [link](https://pip.pypa.io/en/stable/installation)
4. Install numpy and opencv-python package installer 
```
    pip3 install numpy opencv-python 
```
6. Install PyTorch from this [link](https://pytorch.org)
7. Install Jupeyter Lab, Jupyter Notebook, Voila, and IpyKernel 
```
    pip3 install jupyterlab notebook voila ipykernel
```
8. Install deepbots package 
```
    pip3 install deepbots
```
9. After installations, Download and follow the video instructions to run the simulation from this [link](https://youtu.be/C7BN6PAKfh8)