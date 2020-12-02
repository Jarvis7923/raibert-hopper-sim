
import time
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons


class vis:
    def __init__(self, fig, rd):
        self._fig = fig
        self._xrange, self._yrange = 5, 5
        self._ax = self._fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-self._xrange/2, self._xrange/2), ylim=(-0.1, self._yrange-0.1))
        # self._ax.margins(x=0.0, y=0.25)
        plt.subplots_adjust(left=0.15, bottom=0.15)
        self._ax.fill_between([-100,100], -10, 0, facecolor='lightgray', interpolate=True)
        self._ax.plot([-100,100], [0, 0], color='silver')
        self._rd = rd
        self. _init_eles()

    def _init_eles(self):
        self.scats, self.lns, self.texts = [], [], [] 
        
        for ele in self._rd.vis_body:
            data, width, color = ele
            ln, = self._ax.plot([],[], lw=width, c=color) 
            self.lns.append(ln)

        for ele in self._rd.vis_point:
            data, size, color = ele
            self.scats.append(self._ax.scatter([], [], s=size**2, c=color))
        
        for ele in self._rd.vis_text:
            data, pos, expr, size, color = ele
            self.texts.append(self._ax.text(pos[0], pos[1], expr.format(data), transform=self._ax.transAxes, fontsize=size, color=color))
        
    def ani_init(self):
        # for l in self.lns:
        #     l.set_data([], [])
        # for s in self.scats:
        #     s.set_offsets([])
        # for t in self.texts:
        #     t.set_text('')
        for ln, ele in zip(self.lns, self._rd.vis_body):
            ln.set_data(*(ele[0][0,:], ele[0][1,:]))
        
        for s, ele in zip(self.scats, self._rd.vis_point):
            s.set_offsets(ele[0].flat)
        
        for t, ele in zip(self.texts, self._rd.vis_text):
            data, pos, expr, size, color = ele
            t.set_text(expr.format(data))

        return tuple(self.lns + self.scats + self.texts)

    def ani_update(self, i):
        for ln, ele in zip(self.lns, self._rd.vis_body):
            ln.set_data(*(ele[0][0,:], ele[0][1,:]))
        
        for s, ele in zip(self.scats, self._rd.vis_point):
            s.set_offsets(ele[0].flat)
        
        for t, ele in zip(self.texts, self._rd.vis_text):
            data, pos, expr, size, color = ele
            t.set_text(expr.format(data))
        
        rxmax, rxmin, rymax, rymin = self._rd.vis_lim
        xmin, xmax = self._ax.get_xlim()
        ymin, ymax = self._ax.get_ylim()

        if rxmax > xmax: 
            self._ax.set_xlim(xmax=rxmax, xmin=rxmax-self._xrange)
        if rymax > ymax: 
            self._ax.set_ylim(ymax=rymax, ymin=rymax-self._yrange)
        if rxmin < xmin: 
            self._ax.set_xlim(xmax=rxmin+self._xrange, xmin=rxmin)
        if rymin < ymin: 
            self._ax.set_ylim(ymax=rymin+self._yrange, ymin=rymin)
        
        return tuple(self.lns + self.scats + self.texts)

    def show(self, hz = 30):
        dt, t0 = 1000/hz, time.time()
        self.ani_update(0)
        t1 = time.time()
        interval = dt - (t1 - t0)
        
        axcolor = 'lightgoldenrodyellow'
    
        axv = plt.axes([0.4, 0.03, 0.3, 0.02], facecolor=axcolor)
        sv = Slider(axv, 'La', 0.0, 0.1, valinit=0.02)

        def update(val):
            self._rd.x_lengthen = sv.val
        sv.on_changed(update)

        axh = plt.axes([0.4, 0.07, 0.3, 0.02], facecolor=axcolor)

        sh = Slider(axh, 'vF', -0.5, 0.5, valinit=0)

        def update(val):
            self._rd.dxD = sh.val
        sh.on_changed(update)
    
        # rax = plt.axes([0.1, 0.7, 0.05, 0.05])
        # button = Button(rax, 'change')
        # def change(event):
        #     self._rd.test = 0.0
        # button.on_clicked(change)

        ani = animation.FuncAnimation(self._fig, self.ani_update, init_func=self.ani_init,
                        frames=300, interval=interval, blit=False)
        plt.show()

