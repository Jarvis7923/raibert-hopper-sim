import numpy as np 

from src.sim import sim
from src.rd import rd

if __name__ == "__main__":
    r = rd()
    MB, MF, IB, IF, L, la, c, ks, kt, x0s = r.params
    
    s = sim(dt=0.002, damping=1e-5, g=9.81)
    # s = sim(dt=0.0008, damping=0, g=0.0)
    s.spawn(r, [0, 2.0, 0.6*np.pi/8.0, -0.6*np.pi/8.0, x0s, 0.0, 0, 0, 0, 0])
    # s.spawn(r, [0, 2.5, 0.0, 0.0, x0s, 0.0, 0, 0, 0, 0])
    s.run()
