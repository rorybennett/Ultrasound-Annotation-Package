"""
Spline interaction class for dragging spline points around and updating the related recording object.
"""
import numpy as np
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from classes import Scan, Utils, ScanUtil as su
from classes.ErrorDialog import ErrorDialog


class Spline:
    """
    A polygon editor.

    Key-bindings
      'd' delete the vertex under point

      'i' insert a vertex at point.  You must be within epsilon of the
          line connecting two existing vertices

    """
    showVertices = True
    epsilon = 5  # max pixel distance to count as a vertex hit

    def __init__(self, ax, scan: Scan):
        self.spline_success = False
        try:
            self.background = None
            self.ax = ax

            self.scan = scan
            self.dd = self.scan.displayDimensions
            self.fd = self.scan.frames[self.scan.currentFrame - 1].shape
            # Points for polygon - vertices.
            points = self.scan.getPointsOnFrame()
            points = np.asfarray(Utils.organiseClockwise(points))
            points = np.append(points, [points[0, :]], axis=0)
            points = np.asfarray([su.pixelsToDisplay(point, self.dd, self.fd) for point in points])
            # Polygon created using points as vertices.
            self.poly = Polygon(np.column_stack([points[:, 0], points[:, 1]]), animated=True, visible=False)

            ax.add_patch(self.poly)

            x, y = zip(*self.poly.xy)

            # Used to show vertices along spline line
            self.line = Line2D(x, y, ls='--', linewidth=.5, marker='x', markeredgecolor=(0, 1, 0), animated=True)
            self.ax.add_line(self.line)

            xs, ys = self.poly.xy[:].T
            xs, ys = Utils.interpolate(xs, ys, 1000)
            self.spline = Line2D(xs, ys, animated=True, color=(0, 1, 0))
            self.ax.add_line(self.spline)

            self.cid_poly = self.poly.add_callback(self.poly_changed)
            # The active vertex
            self._ind = None

            # Canvas event binding.
            self.canvas = self.poly.figure.canvas
            self.cid_draw = self.canvas.mpl_connect('draw_event', self.draw_event)
            self.cid_button_press = self.canvas.mpl_connect('button_press_event', self.button_press_event)
            self.cid_button_release = self.canvas.mpl_connect('button_release_event', self.button_release_event)
            self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self.motion_notify_event)

            self.canvas.draw()

            self.spline_success = True
        except Exception as e:
            ErrorDialog(None, 'Error initialising spline, ensure enough points are available', e)

    def draw_event(self, _):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.ax.draw_artist(self.spline)

    def poly_changed(self, poly):
        """This method is called whenever the path-patch object is called."""
        # Only copy the artist props to the line (except visibility).
        vis = self.line.get_visible()
        Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # Don't use the poly visibility state.

    def index_under_event(self, event):
        """
        Return the index of the point closest to the event position or *None*
        if no point is within ``self.epsilon`` to the event position.
        """
        # display coordinates
        xy = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]

        d = np.hypot(xt - event.x, yt - event.y)
        ind_seq, = np.nonzero(d == d.min())
        ind = ind_seq[0]

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def button_press_event(self, event):
        """
        Callback for mouse button presses.
        """
        if not self.showVertices or event.inaxes is None:
            return
        elif event.button == 1:  # Drag points.
            self._ind = self.index_under_event(event)
        elif event.button == 2:  # Add points.
            xys = self.poly.get_transform().transform(self.poly.xy)
            p = event.x, event.y  # Display coordinates.
            for i in range(len(xys) - 1):
                s0 = xys[i]
                s1 = xys[i + 1]
                d = Utils.pointSegmentDistance(p, s0, s1)
                if d <= self.epsilon:
                    self.poly.xy = np.insert(self.poly.xy, i + 1, [event.xdata, event.ydata], axis=0)
                    self.line.set_data(zip(*self.poly.xy))
                    xs, ys = self.poly.xy[:].T
                    xs, ys = Utils.interpolate(xs, ys, 1000)
                    self.spline.set_data(xs, ys)
                    break
        elif event.button == 3:  # Delete points.
            ind = self.index_under_event(event)
            if ind is not None:
                self.poly.xy = np.delete(self.poly.xy,
                                         ind, axis=0)
                self.line.set_data(zip(*self.poly.xy))
                xs, ys = self.poly.xy[:].T
                xs, ys = Utils.interpolate(xs, ys, 1000)
                self.spline.set_data(xs, ys)
        if self.line.stale:
            self.canvas.draw_idle()

    def button_release_event(self, event):
        """
        Callback for mouse button releases.
        """
        if not self.showVertices or event.button != 1:
            return

        self._ind = None

    def motion_notify_event(self, event):
        """
        Callback for mouse movements.
        """
        if not self.showVertices or self._ind is None or event.inaxes is None or event.button != 1:
            return

        x, y = event.xdata, event.ydata

        self.poly.xy[self._ind] = x, y

        if self._ind == 0:
            self.poly.xy[-1] = x, y
        elif self._ind == len(self.poly.xy) - 1:
            self.poly.xy[0] = x, y

        self.line.set_data(zip(*self.poly.xy))

        xs, ys = self.poly.xy[:].T
        xs, ys = Utils.interpolate(xs, ys, 1000)
        self.spline.set_data(xs, ys)

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.ax.draw_artist(self.spline)
        self.canvas.blit(self.ax.bbox)

    def close_spline(self):
        # Remove callbacks.
        self.poly.remove_callback(self.cid_poly)
        self.canvas.mpl_disconnect(self.cid_draw)
        self.canvas.mpl_disconnect(self.cid_button_press)
        self.canvas.mpl_disconnect(self.cid_button_release)
        self.canvas.mpl_disconnect(self.cid_motion)
        # Extract endPoints from spline.
        xs, ys = self.poly.xy.T
        # Evenly space the endPoints (21 = 20 endPoints).
        xn, yn = Utils.interpolate(xs, ys, 21)
        endPoints = np.column_stack([xn, yn])[:-1]

        # Due to differences between canvas origin, axis origin, and poly origin, this is a special conversion.
        points_ratio = [[p[0] / self.dd[0], p[1] / self.dd[1]] for p in endPoints]

        endPoints = [[p[0] * self.d[1] - (self.d[1] / (self.d[1] * (self.ip / 100))), p[1] * self.d[0] + self.io]
                         for p in points_ratio]

        self.scan.postSplineInteraction(endPoints)
