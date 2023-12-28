import numpy as np
import time
from scipy.spatial.transform import Rotation

from wxr.common import *

def round_to_sliced_time(t):
    return round(t * SIM_FPS) / SIM_FPS

def wxr_to_isaac(p):
    x = p[0]
    y = p[1]
    z = p[2]
    r = np.zeros(3, dtype=np.float32)
    r[0] = -z
    r[1] = -x
    r[2] = y
    
    return r

def isaac_to_wxr(p):
    x = p[0]
    y = p[1]
    z = p[2]
    r = np.zeros(3, dtype=np.float32)
    r[0] = -y
    r[1] = z
    r[2] = -x
    
    return r

init_time = round_to_sliced_time(time.time())
def get_init_time():
    return init_time

def get_curr_idx():
    curr_time = round_to_sliced_time(time.time())
    time_offset = curr_time - init_time
    idx = round(time_offset * SIM_FPS) % STATE_BUFFER_SIZE
    return idx

def euler_to_quat(e):
    rot = Rotation.from_euler('xyz', e)
    q = rot.as_quat()
    return q
    
    