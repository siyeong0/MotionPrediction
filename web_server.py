import numpy as np
import threading

from wxr.common import *
from wxr.util import *

from wxr.gym_runner import GymRunner

from isaacgymenvs.utils.torch_jit_utils import quat_mul

import torch

async_mode = None

from flask import Flask, render_template
import socketio

sio = socketio.Server(
    async_mode=async_mode,
    cors_allowed_origins=[
        'http://192.168.1.183:8000',
        'https://admin.socket.io',
    ])

app = Flask(__name__)
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)
app.config['SECRET_KEY'] = 'secret'
thread = None
gym = GymRunner(MAX_NUM_ENVS, MODEL_PATH, 'cuda')

head_state_buf = torch.zeros((STATE_BUFFER_SIZE, MAX_NUM_ENVS, 7), dtype=torch.float32, device='cpu')
left_hand_state_buf = torch.zeros((STATE_BUFFER_SIZE, MAX_NUM_ENVS, 3), dtype=torch.float32, device='cpu')
right_hand_state_buf = torch.zeros((STATE_BUFFER_SIZE, MAX_NUM_ENVS, 3), dtype=torch.float32, device='cpu')

# initilize buffers to valid values
head_state_buf[:,:,2] = 1.4     # default head height
head_state_buf[:,:,6] = 1.0     # make quaternion to valid value
left_hand_state_buf[:,:,1] = 0.18       # leftward is +y ,it looks at +x axis
left_hand_state_buf[:,:,2] = 0.83 - 1.4  # default hand height, this buffer represent head's coordinate
right_hand_state_buf[:,:,1] = -0.18     # rightward is -y ,it looks at +x axis
right_hand_state_buf[:,:,2] = 0.83 - 1.4 # default hand height, this buffer represent head's coordinate

buffer_lock = threading.Lock()

def background_thread():
    while True:
        sio.sleep(0.001)
        # copy state buffers
        buffer_lock.acquire()
        hsb = torch.clone(head_state_buf)
        lhb = torch.clone(left_hand_state_buf)
        rhb = torch.clone(right_hand_state_buf)
        buffer_lock.release()
        
        # step simulation
        step_ret = gym.step(hsb, lhb, rhb)
        
        if not step_ret:
            continue
        else:
            root_state, link_state = step_ret
        
        env_idx = 0
        
        def to_wxr_skeleton(quat, base_trans_euler=[0.0, 0.0, 0.0], degree=True):
            base = np.array(base_trans_euler) / 180.0 * np.pi if degree else np.array(base_trans_euler)
            base_quat = torch.Tensor(euler_to_quat(base))
            quat = quat_mul(quat, base_quat)
            return isaac_to_wxr_quat(quat.numpy()).astype(np.float32)
        
        #root            = to_wxr_skeleton(link_state[env_idx,0,3:7])
        root            = to_wxr_skeleton(root_state[env_idx, 3:7], [180,180,0])
        torso           = to_wxr_skeleton(link_state[env_idx,1,3:7], [180,180,0])
        neck            = to_wxr_skeleton(link_state[env_idx,2,3:7], [180,180,0])
        right_upper_arm = to_wxr_skeleton(link_state[env_idx,3,3:7], [0,180,0])
        right_lower_arm = to_wxr_skeleton(link_state[env_idx,4,3:7], [0,180,0])
        right_hand      = to_wxr_skeleton(link_state[env_idx,5,3:7], [0,180,0])
        left_upper_arm  = to_wxr_skeleton(link_state[env_idx,6,3:7], [0,180,0])
        left_lower_arm  = to_wxr_skeleton(link_state[env_idx,7,3:7], [0,180,0])
        left_hand       = to_wxr_skeleton(link_state[env_idx,8,3:7], [0,180,0])
        right_thigh     = to_wxr_skeleton(link_state[env_idx,9,3:7], [0,180,0])
        right_shin      = to_wxr_skeleton(link_state[env_idx,10,3:7], [0,180,0])
        right_foot      = to_wxr_skeleton(link_state[env_idx,11,3:7], [0,120,0])
        left_thigh      = to_wxr_skeleton(link_state[env_idx,12,3:7], [0,180,0])
        left_shin       = to_wxr_skeleton(link_state[env_idx,13,3:7], [0,180,0])
        left_foot       = to_wxr_skeleton(link_state[env_idx,14,3:7], [0,120,0])
        head            = neck
        quat_arr = bytes(np.concatenate((root, right_thigh, right_shin, right_foot, 
                                         left_thigh, left_shin, left_foot, torso, 
                                         left_upper_arm, left_lower_arm, neck, head, 
                                         right_upper_arm, right_lower_arm), axis=0))
        # bb = np.array([0,0,0,1],dtype=np.float32)
        # quat_arr = bytes(np.concatenate((bb, bb, bb, bb, 
        #                                  bb, bb, bb, bb, 
        #                                  bb, bb, bb, bb, 
        #                                  bb, bb), axis=0))
        # quat_arr = bytes(np.concatenate((root, left_thigh, left_shin, left_foot, 
        #                                  right_thigh, right_shin, right_foot, torso, 
        #                                  right_upper_arm, right_lower_arm, neck, head, 
        #                                  left_upper_arm, left_lower_arm), axis=0))
        
        root_pos = list(isaac_to_wxr(root_state[0,0:3].numpy()).astype(float))
        head_pos = list(isaac_to_wxr(link_state[0,2,0:3].numpy()).astype(float))
        
        skeletonData = {
            'quatArr' : quat_arr,
            'rootPos' : root_pos,
            'headPos' : head_pos,
            'bodypart' : 'body',
        }
        
        # emit skeleton data to wxr
        sio.emit('vrMotionPredBodyMoving', skeletonData)

@app.route('/')
def index():
    global thread
    if thread is None:
        thread = sio.start_background_task(background_thread)
    return render_template('index.html')

@sio.event
def user_msg(sid, data):
    print("sid: ", sid)
    print("data: ", data)
    sio.emit('message', data, sid)   # client only transmit
    
@sio.event
def all_msg(sid, data):
    print("sid: ", sid)
    print("data: ", data)
    sio.emit('message', data)   # broadcase transmit
    
# server buffers
assign_table = np.arange(MAX_NUM_ENVS)
id_table = {}
ass_idx = 0
    
@sio.event
def connect(sid, environ, auth):
    global ass_idx
    print('connect ', sid)
    buffer_lock.acquire()
    env_id = assign_table[ass_idx]
    id_table[sid] = env_id
    buffer_lock.release()
    ass_idx += 1
    
@sio.event
def disconnect(sid):
    buffer_lock.acquire()
    env_id = id_table[sid]
    del(id_table[sid])
    assign_table[ass_idx] = env_id
    buffer_lock.release()
    
@sio.on('userHeadSensorData')
def get_head_state(sid, sensorData):
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
    if (gym.reset_buf[env_id] == 2):
        gym.head_offset[env_id] = 1.4 - head_pos[2]
        gym.height_scale[env_id] = 1.4 / 1.8
        gym.reset_idx_buf[env_id] = idx
        gym.reset_buf[env_id] = 1
    # position
    #print(head_pos)
    head_state_buf[idx, env_id, 0:3] = torch.Tensor(head_pos) + gym.head_offset[env_id]
    print(torch.Tensor(head_pos) + gym.head_offset[env_id])
    head_state_buf[idx, env_id, 3:7] = torch.Tensor(euler_to_quat(head_rot))
    buffer_lock.release()
    ### lock ###
    sio.sleep(0.001)
    
@sio.on('userHandSensorData')
def get_hand_state(sid, sensorData):
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
    sio.sleep(0.001)
    
if __name__ == '__main__':
    
    
    
    if sio.async_mode == 'threading':
        # deploy with Werkzeug
        gym_thread = threading.Thread(target=background_thread)
        gym_thread.start()
        #thread = sio.start_background_task(background_thread)
        app.run(host='192.168.1.183', port=8000, threaded=True)
    elif sio.async_mode == 'eventlet':
        # deploy with eventlet
        import eventlet
        import eventlet.wsgi
        eventlet.wsgi.server(eventlet.listen(('', 5000)), app)
    elif sio.async_mode == 'gevent':
        # deploy with gevent
        from gevent import pywsgi
        try:
            from geventwebsocket.handler import WebSocketHandler
            websocket = True
        except ImportError:
            websocket = False
        if websocket:
            pywsgi.WSGIServer(('', 5000), app,
                              handler_class=WebSocketHandler).serve_forever()
        else:
            pywsgi.WSGIServer(('', 5000), app).serve_forever()
    elif sio.async_mode == 'gevent_uwsgi':
        print('Start the application through the uwsgi server. Example:')
        print('uwsgi --http :5000 --gevent 1000 --http-websockets --master '
              '--wsgi-file app.py --callable app')
    else:
        print('Unknown async_mode: ' + sio.async_mode)