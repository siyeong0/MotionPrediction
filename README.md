# Motion Prediction from sparse sensor
[Reference project](https://sunny-codes.github.io/projects/questenvsim.html)
## Setup
- OS : Ubuntu 18.04 or 20.04
### Gym
1. Create python virtual environment with python 3.8
2. Install pytorch
3. Install [Isaac Gym](https://developer.nvidia.com/isaac-gym)
    ```bash
    cd issacgym/python
    pip install -e .
    ```
4. Install Issac Gym Envs
    ```bash
    git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git
    cd IsaacGymEnvs
    pip install -e .
    ```
5. Install rsl_rl
    ```bash
    git clone https://github.com/leggedrobotics/rsl_rl.git
    cd rsl_rl && git checkout v1.0.2
    pip install -e .
    ```
6. Clone this repository
    ```bash
    git clone https://github.com/siyeong0/MotionPrediction.git
    ```
### PoseLib
1. Create python virtual environment with python 3.7
2. Install [Isaac Gym](https://developer.nvidia.com/isaac-gym)
    ```bash
    cd issacgym/python
    pip install -e .
    ```
3. Install Issac Gym Envs
    ```bash
    git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git
    cd IsaacGymEnvs
    pip install -e .
    ```
4. Install bpy
    ```bash
    wget https://files.pythonhosted.org/packages/f4/fd/749f83e5006a3ecd2b134b20b42b7d5140e90d0ff3b45111abcb75027e80/bpy-2.91a0-cp37-cp37m-manylinux2014_x86_64.whl
    pip install bpy-2.91a0-cp37-cp37m-manylinux2014_x86_64.whl && bpy_post_install
    ```
5. Install FBX SDK
    - [FBX SDK 2020.2.1 Python](https://www.autodesk.com/content/dam/autodesk/www/adn/fbx/2020-2-1/fbx202021_fbxpythonsdk_linux.tar.gz)
    ```bash
    tar -zxvf fbx202021_fbxpythonsdk_linux.tar.gz
    chmod ugo+x fbx202021_fbxpythonsdk_linux ./fbx202021_fbxpythonsdk_linux ~/fbxsdk
    cp fbxsdk/lib/Python37_x64/* /anaconda3/envs/__name__/lib/python3.7/site-packages
    ```
## Dataset
- [CMU Mocap dataset](https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture/the-motionbuilder-friendly-bvh-conversion-release-of-cmus-motion-capture-database)
