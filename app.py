from aiohttp import web
import socketio
import struct
import numpy as np

## creates a new Async Socket IO Server
sio = socketio.AsyncServer()
## Creates a new Aiohttp Web Application
app = web.Application()
# Binds our Socket.IO server to our Web App instance
sio.attach(app)

## If we wanted to create a new websocket endpoint,
## use this decorator, passing in the name of the
## event we wish to listen out for
@sio.event
async def user_msg(sid, data):
    print("sid: ", sid)
    print("data: ", data)
#    await sio.emit('message', data)   # broadcase transmit
    await sio.emit('message', data, sid)   # client only transmit

@sio.event
async def all_msg(sid, data):
    print("sid: ", sid)
    print("data: ", data)
    await sio.emit('message', data)   # broadcase transmit

@sio.event
async def connect(sid, environ, auth):
    print('connect ', sid)

@sio.event
async def disconnect(sid):
    print('disconnect ', sid)

@sio.on('join')
async def join(sid, data):
    print(str(data))
    
@sio.on('hmd')
async def hmd_state(sid, state):
    hmd_state = np.array(struct.unpack('ffffff', state))
    print(np.array(struct.unpack('ffffff', state)))
    hmd_pos = hmd_state[0:3]
    hmd_rot = hmd_state[3:6]
    

## We kick off our server
if __name__ == '__main__':
    web.run_app(app, host='192.168.1.183', port=8000)