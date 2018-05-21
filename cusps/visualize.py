# 
# visualize.py
##
#
import cusps
import os
import matplotlib.pyplot as plt


class plotter(object):
    def __init__(self, **kargs):

        xscale, yscale = 'linear','linear'
        if cusps.util.get_from_dict(kargs, 'xscale'):
            xscale = kargs.pop('xscale')
        if cusps.util.get_from_dict(kargs, 'yscale'):
            yscale = kargs.pop('yscale')

        self.fig = plt.figure(**kargs)
        self.ax  = self.fig.gca()

        self.ax.set_xscale(xscale)
        self.ax.set_yscale(yscale)

    def set_xlim(self, xlims):
        '''
            xlims: a list of form [xlow, xmax]
        '''
        self.ax.set_xlim(xlims)

    def set_ylim(self, ylims):
        '''
            ylims: a list of form [ylow, ymax]
        '''
        self.ax.set_ylim(ylims)

    def set_title(self, title, fontsize=18,  **kwargs):
        kwargs.update({'fontsize':fontsize})
        self.ax.set_title(title, **kwargs)

    def set_xlabel(self, label, fontsize=18, **kwargs):
        kwargs.update({'fontsize':fontsize})
        self.ax.set_xlabel(label, **kwargs)

    def set_ylabel(self, label, fontsize=18, **kwargs):
        kwargs.update({'fontsize':fontsize})
        self.ax.set_ylabel(label, **kwargs)

    def add_data(self, x,y, **kwargs):
        self.ax.plot(x,y,**kwargs)

    def add_err(self,x,y,yerr,ls='none',**kwargs):
        self.ax.errorbar(x,y,yerr=yerr,ls=ls,**kwargs)

    def show_legends(self, **kwargs):
        self.ax.legend(**kwargs)

    def tick_params(self, **kwargs):
        self.ax.tick_params(**kwargs)

    def show(self):
        self.fig.show()

    def clear(self):
        self.fig.clf()

    def vline(self, x=0, **kwargs):
        self.ax.axvline(**kwargs)

    def hline(self, y=0, **kwargs):
        self.ax.axhline(y=0, **kwargs)

    def save(self, output):
        print "saving %s" %output
        self.fig.savefig(output)
        self.clear()

