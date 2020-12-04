
import numpy as np 

from enum import Enum

def sRb(q):
    sq, cq = np.sin(q), np.cos(q)
    return np.array([
        [cq, -sq],
        [sq, cq]
    ])


class rphase(Enum):
    """ enumerate for different phases in the jumping locomotion
    """
    TD = 0 # touchdown
    SQD = 1 # squat down
    BOTTOM = 2 # bottom
    STU = 3 # stand up 
    LO = 4 # liftoff
    RIS = 5 # rising
    TOP = 6 # top
    FAL = 7 # falling


class rd:
    def __init__(self, 
                 MB=10.0,
                 MF=1.0,
                 IB=10.0, 
                 IF=1.0, 
                 L=1.3,  
                 la=0.05,  
                 c=0.3,  
                 ks=1000.2,  
                 kt=10000.0,  
                 x0s=1.0,
                 la_min = 0.1,
                 la_max = 0.5,
                ):
        self.time_elapsed = 0.0
        self.dt = 0.001
        self.params = [MB, MF, IB, IF, L, la, c, ks, kt, x0s]
        self.g = 9.81
        self.damping = 1e-5
        self.Th, self.dTh, self.ddTh = [], [], []
        self.contact_point = None
        self.status = None

        self.qTD, self.thLO =  0.0, 0.0
        self.la_min, self.la_max = la_min, la_max

        self.la_tar, self.la0, self.la_t0 = 0.05, 0.0, 0.0

        self.dxD, self.x_lengthen = 0.0, 0.02

        self.test = 0.0

    def set_state(self, state):
        self.Th = np.array(state[:5])
        self.dTh = np.array(state[5:])

    @property
    def state(self):
        return np.r_[self.Th, self.dTh]
        return np.r_[self.Th, self.dTh].tolist()

    def model(self, state, t=None, tau=0):
        MB, MF, IB, IF, L, la, c, ks, kt, x0s = self.params
        g, damping = self.g, self.damping
        
        x, y, th, q, l = self.Th
        dx, dy, dth, dq, dl = self.dTh
        
        cq, sq, cth, sth = np.cos(q), np.sin(q), np.cos(th), np.sin(th)
        cqth, sqth =  np.cos(q+th), np.sin(q+th)
        
        k = ks if (l < x0s) and (l > 0.0) else kt 

        M = np.array([
            [MB+MF,0,MF*(c*cth+cqth*(-(L/2)+la+l)),MF*cqth*(-(L/2)+la+l),MF*sqth],
            [0,MB+MF,MF*(c*sth+(-(L/2)+la+l)*sqth),MF*(-(L/2)+la+l)*sqth,-MF*cqth],
            [MF*(c*cth+cqth*(-(L/2)+la+l)),MF*(c*sth+(-(L/2)+la+l)*sqth),IB+IF+MF*(c*cth+cqth*(-(L/2)+la+l))**2+MF*(c*sth+(-(L/2)+la+l)*sqth)**2,IF+(L**2*MF)/4-L*la*MF+la**2*MF-1/2*c*(L-2*la)*MF*cq-MF*(L-2*la-c*cq)*l+MF*l**2,c*MF*sq],
            [MF*cqth*(-(L/2)+la+l),MF*(-(L/2)+la+l)*sqth,IF+(L**2*MF)/4-L*la*MF+la**2*MF-1/2*c*(L-2*la)*MF*cq-MF*(L-2*la-c*cq)*l+MF*l**2,IF+1/4*(L-2*la)**2*MF-(L-2*la)*MF*l+MF*l**2,0],
            [MF*sqth,-MF*cqth,c*MF*sq,0,MF]
        ], dtype=np.float)
        
        h = np.array([
            1/2*MF*(-2*c*sth*dth**2+cqth*(4*dl*dq+4*dl*dth)+sqth*(L*dq**2-2*la*dq**2-2*l*dq**2+2*L*dq*dth-4*la*dq*dth-4*l*dq*dth+L*dth**2-2*la*dth**2-2*l*dth**2)),
            1/2*(2*g*MB+2*g*MF+2*c*MF*cth*dth**2+sqth*(4*MF*dl*dq+4*MF*dl*dth)+cqth*(-L*MF*dq**2+2*la*MF*dq**2+2*MF*l*dq**2-2*L*MF*dq*dth+4*la*MF*dq*dth+4*MF*l*dq*dth-L*MF*dth**2+2*la*MF*dth**2+2*MF*l*dth**2)),
            1/2*MF*(2*c*g*sth+(-g*L+2*g*la+2*g*l)*sqth-2*L*dl*dq+4*la*dl*dq+4*l*dl*dq-2*L*dl*dth+4*la*dl*dth+4*l*dl*dth+cq*(4*c*dl*dq+4*c*dl*dth)+sq*(c*L*dq**2-2*c*la*dq**2-2*c*l*dq**2+2*c*L*dq*dth-4*c*la*dq*dth-4*c*l*dq*dth)),
            -(1/2)*MF*(L-2*la-2*l)*(g*sqth+2*dl*dq+2*dl*dth+c*sq*dth**2),
            1/2*(-2*k*x0s-2*g*MF*cqth+2*k*l+L*MF*dq**2-2*la*MF*dq**2-2*MF*l*dq**2+2*L*MF*dq*dth-4*la*MF*dq*dth-4*MF*l*dq*dth+L*MF*dth**2-2*la*MF*dth**2-2*c*MF*cq*dth**2-2*MF*l*dth**2),
        ], dtype=np.float)
        
        # damping
        bs = 0.00001 if (l < x0s) and (l > 0.0) else 125.0
        d = np.array([
            [damping, 0, 0, 0, 0],
            [0, damping, 0, 0, 0],
            [0, 0, damping, 0, 0],
            [0, 0, 0, damping, 0],
            [0, 0, 0, 0, bs]
        ], dtype=np.float)
        
        # contact 
        peef = np.array([
            c*sth+(la+l)*sqth+x,
            -c*cth-cqth*(+la+l)+y,
        ], dtype=np.float)

        if peef[1] < 0:
            if self.contact_point is None:
                self.contact_point = np.array([peef[0], 0.0])
            
            Jeef = np.array([
                [1,0,c*cth+cqth*(la+l),cqth*(la+l),sqth],
                [0,1,c*sth+(la+l)*sqth,(la+l)*sqth,-cqth],
            ], dtype=np.float)

            veef = Jeef @ self.dTh

            bc = np.array([[275.0, 0],[0, 275.0]], dtype=np.float)
            kc = np.array([[30000.0, 0.0],[0.0, 30000.0]],dtype=np.float)
            fc = - kc @ (peef - self.contact_point) - bc @ veef
            tau_c = Jeef.T @ fc 
            
            # fc, kc, bc, mu = np.zeros(2), 10000.0, 125.0, 0.0
            # fc[1] = - kc * (peef[1] - self.contact_point[1]) - bc * veef[1]
            # fc[0] = - np.sign(veef[0])*mu*fc[1]
            # tau_c = Jeef.T @ fc 
            
            # invM = np.linalg.inv(M)
            # ddTheta = invM @ (tau - h - d @ self.dTh)
            # A = Jeef
            # P = np.eye(A.shape[1]) - invM @ A.T @ np.linalg.inv(A @ invM @ A.T) @ A 
            # ddTheta = P @ ddTheta

        else:
            self.contact_point = None 
            tau_c = np.zeros(5)

        # invM = np.linalg.inv(M)
        # ddTheta = invM @ (tau - h - d @ self.dTh + tau_c) 
        
        ddTheta = np.linalg.solve(M, tau - h - d @ self.dTh + tau_c)

        return np.r_[self.dTh, ddTheta]


    def controller_func(self):
        tau = np.zeros(5) 
        
        MB, MF, IB, IF, L, la, c, ks, kt, x0s = self.params
        x, y, th, q, l = self.Th
        dx, dy, dth, dq, dl = self.dTh
        
        dxD, thD = self.dxD, 0.0
        dx_max = 0.8

        x_lengthen = self.x_lengthen
        
        # if self.time_elapsed > 3:
        #     dxD, thD = 0.5, 0.0
        #     # dxD = -np.sign(x-0)*min([np.abs(3*(x-0)), 0.5])
        #     dx_max = dxD - 0.1
        #     # x_lengthen = 0.05 + 0.05
        
        # if self.time_elapsed > 20:
        #     dxD, thD = 0.3, 0.0
        #     dx_max = 0.5
        #     # x_lengthen = 0.05 + 0.1
        
        if np.abs(dxD) > 0.005:
           K1, K2, K3 = 0.3, 0.0, 0.0
        else:
           K1, K2, K3 = 0.2, 0.3, 0.4
        
        
        if self.status is None:
            self.qTD = self._fp_controller(dxD, thD, dx_max, K1, K2, K3)

        self._set_status()

        if self.status == rphase.BOTTOM:
            # self.la0, self.la_t0 = la, self.time_elapsed
            self.la_tar = x_lengthen + self.la_min
            
        if self.status == rphase.LO:
            # self.la0, self.la_t0 = la, self.time_elapsed
            self.la_tar = self.la_min
        
        # if (self.status == rphase.FAL) or (self.status == rphase.TOP):
        if (self.status == rphase.RIS) or (self.status == rphase.FAL) or (self.status == rphase.TOP):
            self.qTD = self._fp_controller(dxD, thD, dx_max, K1, K2, K3)
            Kp, Kv = 4500.0, 450.0
            tau[3] = -(Kp*(q - self.qTD) + Kv*(dq))

        if self.status == rphase.TD:
            self.thLO = -th

        if (self.status == rphase.TD)or(self.status == rphase.SQD) or (self.status == rphase.STU) or (self.status == rphase.BOTTOM):
            Kp, Kv = 3000, 325.0
            tau[3] = Kp*(th - self.thLO) + Kv*(dth) 

        lg = la + l
        self.params[5] = self._la_controller(k = 200)
        l = lg - self.params[5]
        self.set_state([x, y, th, q, l, dx, dy, dth, dq, dl ])
        
        return tau
    
    def _set_status(self):
        MB, MF, IB, IF, L, la, c, ks, kt, x0s = self.params
        x, y, th, q, l = self.Th
        dx, dy, dth, dq, dl = self.dTh
        cq, sq, cth, sth = np.cos(q), np.sin(q), np.cos(th), np.sin(th)
        cqth, sqth =  np.cos(q+th), np.sin(q+th)
        
        peef = np.array([
            c*sth+(la+l)*sqth+x,
            -c*cth-cqth*(+la+l)+y,
        ], dtype=np.float)
        
        if self.status == rphase.FAL:
            if peef[1] < 0:
                self.status = rphase.TD
        elif self.status == rphase.TD:
            self.status = rphase.SQD
        elif self.status == rphase.SQD:
            if dy > 0:
                self.status = rphase.BOTTOM
        elif self.status == rphase.BOTTOM:
            self.status = rphase.STU
        elif self.status == rphase.STU:
            if peef[1] > 0:
                self.status = rphase.LO
        elif self.status == rphase.LO:
            self.status = rphase.RIS
        elif self.status == rphase.RIS:
            if dy < 0:
                self.status = rphase.TOP
        elif self.status == rphase.TOP:
            self.status = rphase.FAL
        else:
            if peef[1] < 0:
                if dy < 0:
                    self.status = rphase.SQD
                else:
                    self.status = rphase.STU
            else:
                if dy < 0:
                    self.status = rphase.FAL
                else:
                    self.status = rphase.RIS
    
    def _la_controller(self, k):
        MB, MF, IB, IF, L, la, c, ks, kt, x0s = self.params
        if self.la_tar is None:
            return la 
        res_la = la + k*(self.la_tar - la)*self.dt
        # res_la = self.la0 + np.sign(self.la_tar - la) * k * (self.time_elapsed - self.la_t0)**2
        if res_la >= self.la_tar:
            return self.la_tar
        if res_la <= self.la_min:
            return self.la_min
        if res_la >= self.la_max:
            return self.la_max
        return res_la

    def _fp_controller(self, dxD, thD, dx_max, K1, K2, K3):
        MB, MF, IB, IF, L, la, c, ks, kt, x0s = self.params
        x, y, th, q, l = self.Th
        dx, dy, dth, dq, dl = self.dTh
        cq, sq, cth, sth = np.cos(q), np.sin(q), np.cos(th), np.sin(th)

        if np.abs(dxD) > 0.005:
            tST = np.pi * np.sqrt(MB/ks) 
            if dxD < dx - dx_max :
                xST = tST * (dx - dx_max)
            elif dxD > dx + dx_max : 
                xST = tST * (dx + dx_max)
            else:
                xST = tST * dxD
        else:
            xST = 0.0

        xERR = K1*(dx-dxD) - K2*(th-thD) - K3*(dth)
        xTD = (la+l)*(-c*sth*MB + (MB+MF)*xERR)/(L/2*MF+(la+l)*MB) + xST/2.0
        while np.abs(xTD) > (la+l):
            xERR = xERR*0.95
            xTD = (la+l)*(-c*sth*MB + (MB+MF)*xERR)/(L/2*MF+(la+l)*MB) + xST/2.0

        qTD = np.arcsin(xTD/(la+l)) - th
        return qTD

    @property
    def vis_body(self):
        MB, MF, IB, IF, L, la, c, ks, kt, x0s = self.params
        x, y, th, q, l = self.Th

        cb, cf, cj = self.vis_point

        pb = cb[0].reshape(2,1) + sRb(th) @ np.array([
            [0, -c/4],
            [3*c/8, -c/4],
            [3*c/8, c/4],
            [-3*c/8, c/4],
            [-3*c/8, -c/4],
            [0, -c/4],
            [0, -c],
        ]).T

        pf = cf[0].reshape(2,1) + sRb(q+th) @ np.array([
            [0.0, L/2],
            [0.0, -L/2],
        ]).T

        pa = cj[0].reshape(2,1) + sRb(q+th) @ np.array([
            [0.0, 0.0],
            [0.0, -la],
        ]).T

        return [
            (pb, 1.2, 'blue'),
            (pf, 1.2, 'blue'),
            (pa, 1.8, 'r')
        ]

    @property
    def vis_point(self):
        MB, MF, IB, IF, L, la, c, ks, kt, x0s = self.params
        x, y, th, q, l = self.Th
        cq, sq, cth, sth = np.cos(q), np.sin(q), np.cos(th), np.sin(th)
        cqth, sqth =  np.cos(q+th), np.sin(q+th)
        
        cb = np.array([ x , y ])
        cj = cb + np.array([c*sth, -c*cth])
        cf =  cj + np.array([(l+la-L/2)*sqth, -(l+la-L/2)*cqth])
        return [
            (cb, 2.5, 'indigo'),
            (cf, 2.5, 'indigo'),
            (cj, 2.5, 'green')
        ]
    
    @property
    def vis_lim(self):
        MB, MF, IB, IF, L, la, c, ks, kt, x0s = self.params
        cb, cf, cj = self.vis_point
        ymax, ymin = cb[0][1] + L, cb[0][1] - L
        xmax, xmin = cb[0][0] + L, cb[0][0] - L
        return xmax, xmin, ymax, ymin

    @property
    def vis_text(self):
        x, y, th, q, l = self.Th
        dx, dy, dth, dq, dl = self.dTh
        if self.status is None:
            status = 'None'
        else:
            status = self.status.name
        return [
            (self.time_elapsed, (0.02, 0.96), 'time: {0:.2f} sec', 9, 'b'),
            (th*180/np.pi, (0.02, 0.93), 'th: {0:.1f} deg', 9, 'dimgray'),
            (q*180/np.pi, (0.02, 0.90), 'q: {0:.1f} deg', 9, 'dimgray'),
            ((self.qTD)*180/np.pi, (0.02, 0.87), 'qTD: {0:.1f} deg', 9, 'dimgray'),
            ((q-self.qTD)*180/np.pi, (0.02, 0.84), 'ERR(q-qTD): {0:.1f} deg', 9, 'dimgray'),
            ((self.thLO)*180/np.pi, (0.02, 0.81), 'thLO: {0:.1f} deg', 9, 'dimgray'),
            ((th-self.thLO)*180/np.pi, (0.02, 0.78), 'ERR(th-thLO): {0:.1f} deg', 9, 'dimgray'),
            (status, (0.02, 0.75), 'status: {0}', 9, 'r'),
            (dx, (0.02, 0.72), 'dx: {0:.3f} m/s', 9, 'b'),
        ]
