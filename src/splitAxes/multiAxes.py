import numpy as np
from matplotlib.axes._axes import Axes
from collections.abc import Iterable

class multiAxes:
    def __init__(self, axes=None, disableYticks = False):
        self.d = .015 # how big to make the diagonal lines in axes coordinates
        self._scaleType = 'linear'
        self.disableYticks = disableYticks
        self.axes = axes
        if axes:
            self._disable = False
        else:
            self._disable = True
        if not self._disable:
            self._setup()

    def _setup(self):
        assert not self._disable
        self._break_fmt = self.axes[0].spines['top'].get_edgecolor()

        if len(self.axes) > 1:
            self.axes[0].spines['right'].set_visible(False)
            self.axes[-1].spines['left'].set_visible(False)

            self.axes[0].tick_params(right=False, which='both')
            self.axes[-1].tick_params(left=False, which='both',labelleft=False)

            kwargs = dict(transform=self.axes[0].transAxes, color=self._break_fmt, clip_on=False)
            self.axes[0].plot((1-self.d,1+self.d), (-self.d,+self.d), **kwargs)
            self.axes[0].plot((1-self.d,1+self.d),(1-self.d,1+self.d), **kwargs)

            kwargs.update(transform=self.axes[-1].transAxes)  # switch to the bottom axes
            self.axes[-1].plot((-self.d,+self.d), (1-self.d,1+self.d), **kwargs)
            self.axes[-1].plot((-self.d,+self.d), (-self.d,+self.d), **kwargs)

            for ax in self.axes[1:-1]:
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.tick_params(left=False,right=False,which='both',labelleft=False)

                kwargs = dict(transform=ax.transAxes, color=self._break_fmt, clip_on=False)
                ax.plot((1-self.d,1+self.d), (-self.d,+self.d), **kwargs)
                ax.plot((1-self.d,1+self.d),(1-self.d,1+self.d), **kwargs)

                kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
                ax.plot((-self.d,+self.d), (1-self.d,1+self.d), **kwargs)
                ax.plot((-self.d,+self.d), (-self.d,+self.d), **kwargs)

        if self.disableYticks:
            self.axes[0].tick_params(labelleft=False)

    def _calc_linear_auto_limits(self, x, overshoot=True):
        limits = list()
        domainSize = max(x) - min(x)
        inset = -0.05*domainSize if overshoot else 0
        domainStart = min(x) - inset
        domainEnd = max(x) + inset
        axisLimitSize = (domainSize / len(self.axes)) + (2*inset)/len(self.axes)
        leftLimits = np.arange(domainStart, domainEnd, axisLimitSize)
        rightLimits = np.arange(domainStart + axisLimitSize, domainEnd + axisLimitSize, axisLimitSize)
        for left, right in zip(leftLimits, rightLimits):
            limits.append([left, right])
        return limits

    def _calc_log_auto_limits(self, x):
        ...

    def _calc_auto_x_limits(self, x, limType='linear', recursionDepth=None):
        limits = self._calc_linear_auto_limits(x)
        currentLimits = self.get_xlim()
        if currentLimits[0] <= limits[0][0]: limits = self._calc_linear_auto_limits([currentLimits[0], limits[-1][1]], overshoot=False)
        if currentLimits[1] >= limits[-1][1]: limits = self._calc_linear_auto_limits([limits[0][0], currentLimits[1]], overshoot=False)
        return limits

    def _auto_limits(self, x, limits, limType):
        if limits is not None:
            assert len(limits) == len(self.axes)
        else:
            limits = self._calc_auto_x_limits(x, limType=limType)
        return limits

    def _proto_ax_plot_call(self, fnName, x, y, limits=None, **kwargs):
        assert not self._disable
        limits = self._auto_limits(x, limits, self._scaleType)
        for ax, xlim in zip(self.axes, limits):
            fn = getattr(ax, fnName)
            fn(x, y, **kwargs)
            ax.set_xlim(xlim)

    def plot(self, x, y, **kwargs):
        self._proto_ax_plot_call('plot', x, y, **kwargs)

    def errorbar(self, x, y, **kwargs):
        self._proto_ax_plot_call('errorbar', x, y, **kwargs)

    def scatter(self, x, y, **kwargs):
        self._proto_ax_plot_call('scatter', x, y, **kwargs)

    def step(self, x, y, **kwargs):
        self._proto_ax_plot_call('step', x, y, **kwargs)

    def loglog(self, x, y, **kwargs):
        self._proto_ax_plot_call('loglog', x, y, **kwargs)

    def semilogx(self, x, y, **kwargs):
        self._proto_ax_plot_call('semilogx', x, y, **kwargs)

    def semilogy(self, x, y, **kwargs):
        self._proto_ax_plot_call('semilogy', x, y, **kwargs)

    def bar(self, x, height, **kwargs):
        self._proto_ax_plot_call('bar', x, height, **kwargs)

    def barh(self, y, width, **kwargs):
        self._proto_ax_plot_call('barh', y, width, **kwargs)

    def fill_between(self, x, y1, **kwargs):
        self._proto_ax_plot_call('fill_between', x, y1, **kwargs)

    def fill_betweenx(self, y, x1, **kwargs):
        self._proto_ax_plot_call('fill_betweenx', y, x1, **kwargs)

    def axhline(self, y=0, xmin=0, xmax=0, **kwargs):
        assert not self._disable
        for ax in self.axes:
            ax.axhline(y=y, xmin=xmin, xmax=xmax, **kwargs)

    def axvline(self, x=0, ymin=0, ymax=0, **kwargs):
        assert not self._disable
        for ax in self.axes:
            ax.avhline(x=x, ymin=ymin, ymax=ymax, **kwargs)


    def set_xlim(self, low, high):
        assert not self._disable
        if not isinstance(low, Iterable):
            tempDomainSize = high-low
            limits = self._calc_linear_auto_limits([low, high], overshoot=False)
            for ax, xlim in zip(self.axes, limits):
                ax.set_xlim(xlim)
        else:
            assert len(low) == len(high) == len(self.axes)
            for ax, l, h in zip(self.axes, low, high):
                self.set_xlim(l, h)

    def get_xlim(self):
        assert not self._disable
        lower = self.axes[0].get_xlim()[0]
        upper = self.axes[-1].get_xlim()[-1]
        return lower, upper

    def get_ylim(self):
        assert not self._disable
        return self.axes[0].get_ylim()

    def set_ylim(self, low, high):
        assert not self._disable
        for ax in self.axes:
            ax.set_ylim(low, high)

    def set_xlabel(self, xlabel, position="center", labelpos=0.5, labelpad=0, **kwargs):
        assert not self._disable

        assert position in ["center", "left", "right", "fLeft", "fRight", "manual"]

        fig = self.axes[0].get_figure()
        fig.canvas.draw()
        inv = fig.transFigure.inverted()


        leftPos = self.axes[0].get_position()
        rightPos = self.axes[-1].get_position()

        height = leftPos.y1 - leftPos.y0

        positioner = {"center": 1/2, "left": 1/3, "right": 2/3, "fLeft": 0.1, "fRight": 0.9, "manual": labelpos}

        xPos = ((rightPos.x1 - leftPos.x0)*positioner[position]) + leftPos.x0
        labelpad = inv.transform((0, labelpad))
        fig.text(xPos, leftPos.y0-0.1*height-labelpad[1], xlabel, horizontalalignment='center', **kwargs)

    def set_ylabel(self, ylabel, **kwargs):
        assert not self._disable
        self.axes[0].set_ylabel(ylabel, **kwargs)

    def __repr__(self):
        return f"<multiAxes: {len(self)} axes>"

    def __len__(self):
        if self._disable:
            return 0
        else:
            return len(self.axes)

    def __getitem__(self, keyID):
        assert not self._disable
        return self.axes[keyID]

    def append(self, item):
        assert isinstance(item, Axes)
        if self.axes is None:
            self.axes = list()
        self.axes.append(item)

    def commit(self):
        assert self.axes is not None
        if self._disable:
            self._disable = False
        self._setup()
