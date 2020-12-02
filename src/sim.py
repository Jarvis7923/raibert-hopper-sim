
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons

import numpy as np 
from scipy.integrate import odeint

from enum import Enum
import threading
import time


from src.vis import vis

class msg_type(Enum):
    error = 0
    system = 1
    info = 2

def show_loginfo(msgtype, msg, end='\n'):
    RESET = '\033[0m'
    if msgtype is msg_type.error:
        expr1 = '\033[1;31m[ERROR]'+ RESET
        msg = ' \033[1m' + msg + RESET
    elif msgtype is msg_type.system:
        expr1 = '\033[1;35m[SYSTEM]'+ RESET
        msg = ' \033[4m' + msg + RESET
    elif msgtype is msg_type.info:
        expr1 = '\033[1;32m[INFO]'+ RESET
        msg = ' ' + msg + RESET

    print(expr1 + msg, end=end)

class sim:
    def __init__(self, 
                 dt=0.001,
                 g=9.81, 
                 damping=1e-5,
                 vis=True
                ):
 
        show_loginfo(msg_type.system, "Simulation initializing ... ")
        self._init_physics(dt, g, damping)
        self._stop = False
        self._vis_flag = vis

    def __del__(self):
        show_loginfo(msg_type.system, "Simulation Ends... ")

    def _init_physics(self, dt, g, damping):
        self._dt, self._g, self._damping = dt, g, damping
        show_loginfo(
            msg_type.info, "Physics Parameters: \n\t fixed time step: {0} sec\n\t gravity: {1} m/s^2\n\t body damping: {2} N/(m/s)".format(dt, g, damping))

    def _init_graphics(self):
        show_loginfo(msg_type.system, "Graphics Initializing... ")
        self._fig = plt.figure()
        self._vis = vis(self._fig, self._rd)
        show_loginfo(msg_type.system, "Graphics Ready... ")

    def spawn(self, rd, pos):
        rd.g, rd.damping = self._g, self._damping
        self._rd = rd
        self._rd.set_state(pos)
        self._rd.dt = self._dt

        if self._vis_flag:
            self._init_graphics()
        show_loginfo(msg_type.info, "Robot spawn at:\n\t{0}".format(pos))
    
    def run(self):
        show_loginfo(msg_type.system, "Dynamics loop initializing ... ")
        if self._vis_flag:
            self._thread = threading.Thread(target=self._run_dynamics, args=())
            self._thread.start()
            self._vis.show()
        
            self._stop = True
            self._thread.join()
            show_loginfo(msg_type.system, "Simulation terminating ... ")
        else:
            self._run_dynamics()
            show_loginfo(msg_type.system, "Simulation terminating ... ")

    def _run_dynamics(self):
        time.sleep(2)
        show_loginfo(msg_type.system, "Dynamics start")
        dt = self._dt
        while not(self._stop):
            t0 =  time.time()
            tspan = [0, dt]
            tau = self._rd.controller_func()
            s0 = self._rd.state
            if s0[1] < 0:
                show_loginfo(msg_type.error, "Dynamics error!")
                break
            # s += dt*np.array(self._rd.model(s, tau=tau))
            sol = odeint(self._rd.model, s0, tspan, args=(tau,))
            self._rd.set_state(sol[-1])
            self._rd.time_elapsed += dt
            # show_loginfo(msg_type.info, "curent state:  {0}".format(self._rd.state), end='\r')
            t1 = time.time()
            interval = dt - (t1 - t0)
            if interval > 0 : 
                time.sleep(interval)
        print()
        show_loginfo(msg_type.system, "Dynamics loop terminating... ")
    


