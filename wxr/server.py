from aiohttp import web
import socketio
import time
import threading
import numpy as np
import asyncio

from isaacgym import gymapi

import torch

from mtss_sim.mtss_sim import MotionTrackingSim
from mtss_cfg.mtss_cfg import MtssCfg, MtssPPOCfg

from isaacgymenvs.utils.torch_jit_utils import quat_rotate

from wxr.common import *
from wxr.util import *

sio = socketio.AsyncServer()
app = web.Application()
sio.attach(app)

# server buffers
assign_table = np.arange(MAX_NUM_ENVS)
id_table = {}
ass_idx = 0

buffer_lock = threading.Lock()

head_state_buf = torch.zeros((STATE_BUFFER_SIZE, MAX_NUM_ENVS, 7), dtype=torch.float32, device='cpu')
left_hand_state_buf = torch.zeros((STATE_BUFFER_SIZE, MAX_NUM_ENVS, 3), dtype=torch.float32, device='cpu')
right_hand_state_buf = torch.zeros((STATE_BUFFER_SIZE, MAX_NUM_ENVS, 3), dtype=torch.float32, device='cpu')

reset_buf = torch.full((MAX_NUM_ENVS,), fill_value=2, dtype=torch.long)
reset_idx_buf = torch.zeros((MAX_NUM_ENVS), dtype=torch.long)
head_offset = torch.zeros((MAX_NUM_ENVS), dtype=torch.float32)
height_scale = torch.ones((MAX_NUM_ENVS), dtype=torch.float32)

# initilize buffers to valid values
head_state_buf[:,:,2] = 1.3     # default head height
head_state_buf[:,:,6] = 1.0     # make quaternion to valid value
left_hand_state_buf[:,:,1] = 0.18       # leftward is +y ,it looks at +x axis
left_hand_state_buf[:,:,2] = 0.83 - 1.3  # default hand height, this buffer represent head's coordinate
right_hand_state_buf[:,:,1] = -0.18     # rightward is -y ,it looks at +x axis
right_hand_state_buf[:,:,2] = 0.83 - 1.3 # default hand height, this buffer represent head's coordinate

# event handlers
@sio.event
async def user_msg(sid, data):
    print("sid: ", sid)
    print("data: ", data)
    await sio.emit('message', data, sid)   # client only transmit
    
@sio.event
async def all_msg(sid, data):
    print("sid: ", sid)
    print("data: ", data)
    await sio.emit('message', data)   # broadcase transmit
    
@sio.event
async def connect(sid, environ, auth):
    global ass_idx
    print('connect ', sid)
    buffer_lock.acquire()
    env_id = assign_table[ass_idx]
    id_table[sid] = env_id
    buffer_lock.release()
    ass_idx += 1
    
@sio.event
async def disconnect(sid):
    global ass_idx
    print('disconnect ', sid)
    ass_idx -= 1
    buffer_lock.acquire()
    env_id = id_table[sid]
    del(id_table[sid])
    assign_table[ass_idx] = env_id
    buffer_lock.release()

# message receivers
_sim_prev_idx = -1
@sio.on('runGym')
async def run_isaac_gym(sid, dummyData):
    # extern global variables
    global _sim_prev_idx
    # simulation instance
    env_cfg = MtssCfg()
    env_cfg.wirte("a.txt")
    rl_cfg = MtssPPOCfg()
    sim = MotionTrackingSim(env_cfg, 
                            MAX_NUM_ENVS, 
                            rl_cfg, 
                            MODEL_PATH, 
                            gymapi.SIM_PHYSX, 
                            SIM_DEVICE, False)
    # constants
    motion_time_stride = env_cfg.env.time_stride
    
    # sim step input buffers
    NUM_PAST_FRAMES = env_cfg.env.num_past_frame
    NUM_FUTURE_FRAMES = env_cfg.env.num_future_frame
    BUF_SIZE = 1 + NUM_PAST_FRAMES + NUM_FUTURE_FRAMES
    head_input_buf = torch.zeros((BUF_SIZE, MAX_NUM_ENVS, 7))
    left_hand_input_buf = torch.zeros((BUF_SIZE, MAX_NUM_ENVS, 3))
    right_hand_input_buf = torch.zeros((BUF_SIZE, MAX_NUM_ENVS, 3))
    
    # main simulation loop
    while True:
        # TODO: 
        curr_time = round_to_sliced_time(time.time())
        time_offset = curr_time - get_init_time() - motion_time_stride * (env_cfg.env.num_future_frame + 1)
        idx = round(time_offset * SIM_FPS) % STATE_BUFFER_SIZE
        
        # syncronize with wxr
        if idx == _sim_prev_idx:
            continue
        _sim_prev_idx = idx
        
        # reset environments
        reset_env_ids = (reset_buf == 1).nonzero(as_tuple=False).flatten().to('cuda')
        if len(reset_env_ids) > 0:
            reset_root_state = torch.zeros((len(reset_env_ids), 13), dtype=torch.float32)
            for reset_env_id in reset_env_ids:
                reset_idx = reset_idx_buf[reset_env_id]
                reset_root_state[:, 0:7] = head_state_buf[reset_idx, reset_env_id]
                reset_root_pos = reset_root_state[:, 0:3]
                reset_root_quat = reset_root_state[:, 3:7]
                reset_root_pos[:, 2] = 0.90    # default root pos when dofs are zeroset
                reset_root_quat[:, 0:2] = 0.0  # extract yaw only
                reset_root_quat = torch.nn.functional.normalize(reset_root_quat, dim=1)
                
                head_state_buf[:, reset_env_id, :] = head_state_buf[reset_idx, reset_env_id]
                left_hand_state_buf[:, reset_env_id, :] = left_hand_state_buf[reset_idx, reset_env_id]
                right_hand_state_buf[:, reset_env_id, :] = right_hand_state_buf[reset_idx, reset_env_id]
                
            sim.reset_idx(reset_env_ids)
            sim.init_env_state(reset_env_ids, reset_root_state.to(SIM_DEVICE))
            reset_buf[reset_env_ids] = 0
        
        # step simulation
        base_idx = idx - NUM_PAST_FRAMES
        for fidx in range(BUF_SIZE):
            tidx = (base_idx + fidx) % STATE_BUFFER_SIZE
            head_state = torch.clone(head_state_buf[tidx, :, 0:7])
            head_pos = head_state[:, 0:3]
            head_pos[:, 2] += head_offset[:]
            head_quat = head_state[:, 3:7]
            lhp = left_hand_state_buf[tidx, :, 0:3]
            rhp = right_hand_state_buf[tidx, :, 0:3]
            scaled_lhp = lhp# * height_scale.unsqueeze(1).repeat(1, 3)
            scaled_rhp = rhp# * height_scale.unsqueeze(1).repeat(1, 3)
            glob_left_hand_pos = quat_rotate(head_quat, scaled_lhp) + head_pos
            glob_right_hand_pos = quat_rotate(head_quat, scaled_rhp) + head_pos
            
            head_input_buf[fidx, :, :] = head_state
            left_hand_input_buf[fidx, :, :] = glob_left_hand_pos
            right_hand_input_buf[fidx, :, :] = glob_right_hand_pos
        
        sim.step(head_input_buf,
                 left_hand_input_buf,
                 right_hand_input_buf)
        
        root_state = sim.root_state.to('cpu')
        link_state = sim.link_state.to('cpu')
        dof_state = sim.dof_state.to('cpu') * torch.pi
        
        env_idx = 1
        # joint quaternions
        root            = root_state[0,3:7].numpy().astype(np.float32)
        abodmen         = euler_to_quat(dof_state[env_idx,0:3,0].numpy()).astype(np.float32)
        neck            = euler_to_quat(dof_state[env_idx,3:6,0].numpy()).astype(np.float32)
        head            = euler_to_quat(np.array([0.0, 0.0, 0.0])).astype(np.float32)
        right_shoulder  = euler_to_quat(dof_state[env_idx,6:9,0].numpy()).astype(np.float32)
        right_elbow     = euler_to_quat(np.array([0.0, dof_state[0,9,0], 0.0])).astype(np.float32)
        left_shoulder   = euler_to_quat(dof_state[env_idx,10:13,0].numpy()).astype(np.float32)
        left_elbow      = euler_to_quat(np.array([0.0, dof_state[0,13,0], 0.0])).astype(np.float32)
        right_hip       = euler_to_quat(dof_state[env_idx,14:17,0].numpy()).astype(np.float32)
        right_knee      = euler_to_quat(np.array([0.0, dof_state[0,17,0], 0.0])).astype(np.float32)
        right_ankle     = euler_to_quat(dof_state[env_idx,18:21,0].numpy()).astype(np.float32)
        left_hip        = euler_to_quat(dof_state[env_idx,21:24,0].numpy()).astype(np.float32)
        left_knee       = euler_to_quat(np.array([0.0, dof_state[0,24,0], 0.0])).astype(np.float32)
        left_ankle      = euler_to_quat(dof_state[env_idx,25:28,0].numpy()).astype(np.float32)
        quat_arr = bytes(np.concatenate((root, right_hip, right_knee, right_ankle, 
                                         left_hip, left_knee, left_ankle, abodmen, 
                                         left_shoulder, left_elbow, neck, head, 
                                         right_shoulder, right_elbow), axis=0))
                
        root_pos = list(root_state[0,0:3].numpy().astype(float))
        head_pos = list(link_state[0,2,0:3].numpy().astype(float))
        
        skeletonData = {
            'quatArr' : quat_arr,
            'rootPos' : root_pos,
            'headPos' : head_pos,
            'bodypart' : 'body',
        }
        
        # emit skeleton data to wxr
        await sio.emit('vrMotionPredBodyMoving', skeletonData)
        await asyncio.sleep(0.0001)


@sio.on('userHeadSensorData')
async def get_head_state(sid, sensorData):
    # sensor datas
    head_pos = sensorData['pos']
    head_rot = sensorData['rot']
    head_pos = wxr_to_isaac(head_pos)
    head_rot = wxr_to_isaac(head_rot)
    
    # find user's environment id and buffer's curruent time index
    env_id = id_table[sid]
    idx = get_curr_idx()
        
    # fill buffers
    ### lock ###
    buffer_lock.acquire()
    if (reset_buf[env_id] == 2):
        head_offset[env_id] = 1.3 - head_pos[2]
        height_scale[env_id] = 1.3 / 1.8
        reset_idx_buf[env_id] = idx
        reset_buf[env_id] = 1
    # position
    head_state_buf[idx, env_id, 0:3] = torch.Tensor(head_pos)
    head_state_buf[idx, env_id, 3:7] = torch.Tensor(euler_to_quat(head_rot))
    buffer_lock.release()
    ### lock ###
    await asyncio.sleep(0.0001)
    
@sio.on('userHandSensorData')
async def get_hand_state(sid, sensorData):
     # sensor datas
    left_hand_pos = sensorData['left']
    right_hand_pos = sensorData['right']
    left_hand_pos = wxr_to_isaac(left_hand_pos)
    right_hand_pos = wxr_to_isaac(right_hand_pos)
    
    # find user's environment id and buffer's curruent time index
    env_id = id_table[sid]
    idx = get_curr_idx()

    # fill buffers
    ### lock ###
    buffer_lock.acquire()
    left_hand_state_buf[idx, env_id, 0:3] = torch.Tensor(left_hand_pos)
    right_hand_state_buf[idx, env_id, 0:3] = torch.Tensor(right_hand_pos)
    buffer_lock.release()
    ### lock ###
    await asyncio.sleep(0.0001)
    
# web server runner
def run_web_server(app, host, port):
    web.run_app(app, host=host, port=port)
    
def gym_starter():
    time.sleep(5)
    cio = socketio.Client()
    cio.connect("http://192.168.1.183:8000/")
    cio.emit("runGym", "a")

class Server:
    def __init__(self):
        pass
    
    def run(self):
        server_thread = threading.Thread(target=run_web_server, args=(app, '192.168.1.183', 8000))
        starter_thread = threading.Thread(target=gym_starter)
        starter_thread.start()
        server_thread.run()