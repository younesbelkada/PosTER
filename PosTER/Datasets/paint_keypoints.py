import numpy as np
import matplotlib
import matplotlib.animation
import matplotlib.collections
import matplotlib.patches

import argparse

COCO_PERSON_SKELETON = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]]

class Configurable:
    """Make a class configurable with CLI and by instance.
    .. warning::
        This is an experimental class.
        It is in limited use already but should not be expanded for now.
    To use this class, inherit from it in the class that you want to make
    configurable. There is nothing else to do if your class does not have
    an `__init__` method. If it does, you should take extra keyword arguments
    (`kwargs`) in the signature and pass them to the super constructor.
    Example:
    >>> class MyClass(openpifpaf.Configurable):
    ...     a = 0
    ...     def __init__(self, myclass_argument=None, **kwargs):
    ...         super().__init__(**kwargs)
    ...     def get_a(self):
    ...         return self.a
    >>> MyClass().get_a()
    0
    Instance configurability allows to overwrite a class configuration
    variable with an instance variable by passing that variable as a keyword
    into the class constructor:
    >>> MyClass(a=1).get_a()  # instance variable overwrites value locally
    1
    >>> MyClass().get_a()  # while the class variable is untouched
    0
    """
    def __init__(self, **kwargs):
        # use kwargs to set instance attributes to overwrite class attributes
        for key, value in kwargs.items():
            assert hasattr(self, key), key
            setattr(self, key, value)

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        """Extend an ArgumentParser with the configurable parameters."""

    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Configure the class from parsed command line arguments."""

class KeypointPainter(Configurable):
    """Paint poses.
    The constructor can take any class attribute as parameter and
    overwrite the global default for that instance.
    Example to create a KeypointPainter with thick lines:
    >>> kp = KeypointPainter(line_width=48)
    """

    show_box = False
    show_joint_confidences = False
    show_joint_scales = False
    show_decoding_order = False
    show_frontier_order = False
    show_only_decoded_connections = False

    textbox_alpha = 0.5
    text_color = 'white'
    monocolor_connections = False
    line_width = None
    marker_size = 1
    solid_threshold = 0.5
    font_size = 8

    def __init__(self, *, xy_scale=1.0, highlight=None, highlight_invisible=False, **kwargs):
        super().__init__(**kwargs)

        self.xy_scale = xy_scale
        self.highlight = highlight
        self.highlight_invisible = highlight_invisible

        # set defaults for line_width and marker_size depending on monocolor
        if self.line_width is None:
            self.line_width = 2 if self.monocolor_connections else 6
        if self.marker_size is None:
            if self.monocolor_connections:
                self.marker_size = max(self.line_width + 1, int(self.line_width * 3.0))
            else:
                self.marker_size = max(1, int(self.line_width * 0.5))

    def _draw_skeleton(self, ax, x, y, v, *, skeleton, masked_x=None, masked_y=None, mask_joints=False, skeleton_mask=None, color=None, alpha=1.0, **kwargs):
        if not np.any(v > 0):
            return

        if skeleton_mask is None:
            skeleton_mask = [True for _ in skeleton]
        assert len(skeleton) == len(skeleton_mask)

        # connections
        lines, line_colors, line_styles = [], [], []
        for ci, ((j1i, j2i), mask) in enumerate(zip(np.array(skeleton) - 1, skeleton_mask)):
            if not mask:
                continue
            c = color
            if not self.monocolor_connections:
                c = matplotlib.cm.get_cmap('tab20')((ci % 20 + 0.05) / 20)
            if v[j1i] > 0 and v[j2i] > 0:
                lines.append([(x[j1i], y[j1i]), (x[j2i], y[j2i])])
                line_colors.append(c)
                if v[j1i] > self.solid_threshold and v[j2i] > self.solid_threshold:
                    line_styles.append('solid')
                else:
                    line_styles.append('dashed')
        ax.add_collection(matplotlib.collections.LineCollection(
            lines, colors=line_colors,
            linewidths=kwargs.get('linewidth', self.line_width),
            linestyles=kwargs.get('linestyle', line_styles),
            capstyle='round',
            alpha=alpha,
        ))

        # joints
        ax.scatter(
            x[v > 0.0], y[v > 0.0], s=self.marker_size**2, marker='.',
            color=color if self.monocolor_connections else 'white',
            edgecolor='k' if self.highlight_invisible else None,
            zorder=2,
            alpha=alpha,
        )

        

        # highlight joints
        if self.highlight is not None:
            highlight_v = np.zeros_like(v)
            highlight_v[self.highlight] = 1
            highlight_v = np.logical_and(v, highlight_v)

            ax.scatter(
                x[highlight_v], y[highlight_v], s=self.marker_size**2, marker='.',
                color=color if self.monocolor_connections else 'white',
                edgecolor='k' if self.highlight_invisible else None,
                zorder=2,
                alpha=alpha,
            )

        # show masked joints
        if mask_joints:
            for i in np.arange(.1,1.01,.1):
                ax.scatter(x[masked_x == 0.0], y[masked_y == 0.0], s=(50*i*(1*.9+.1))**2, marker='.', color=(0,0,0,.5/i/10), zorder=3)
            #ax.scatter(
            #    x[masked_x == 0.0], y[masked_y == 0.0], s=(self.marker_size*22)**2, marker='.',
            #    color='black', alpha=1, zorder=3
            #)