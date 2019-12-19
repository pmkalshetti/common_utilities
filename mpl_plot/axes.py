def format_3d_axes(ax, axis_start, axis_range):
    """Formats 3d plot.

    Arguments
    ---------
    ax : matplotlib.Axes

    axis_start : tuple of 3 scalars
        start limit of axis

    axis_range : scalar
        defines common range for all axis.
    """
    # remove grid
    ax.grid(False)

    # set background pane color from gray to white
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # set pane borders from gray to black
    ax.xaxis.pane.set_edgecolor("black")
    ax.yaxis.pane.set_edgecolor("black")
    ax.zaxis.pane.set_edgecolor("black")

    # tick placement (keep clean outside boundary)
    ax.xaxis._axinfo['tick']['inward_factor'] = 0
    ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.yaxis._axinfo['tick']['inward_factor'] = 0
    ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.zaxis._axinfo['tick']['inward_factor'] = 0
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.4

    [t.set_va('center') for t in ax.get_yticklabels()]
    [t.set_ha('left') for t in ax.get_yticklabels()]
    [t.set_va('center') for t in ax.get_xticklabels()]
    [t.set_ha('right') for t in ax.get_xticklabels()]
    [t.set_va('center') for t in ax.get_zticklabels()]
    [t.set_ha('left') for t in ax.get_zticklabels()]

    # set labels
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    # aspect
    ax.set_aspect("equal")

    # set axis limits
    ax.set_xlim(axis_start[0], axis_start[0]+axis_range)
    ax.set_ylim(axis_start[1], axis_start[1]+axis_range)
    ax.set_zlim(axis_start[2], axis_start[2]+axis_range)

    # set ticks at limits
    ax.set_xticks((axis_start[0], axis_start[0]+axis_range))
    ax.set_yticks((axis_start[1], axis_start[1]+axis_range))
    ax.set_zticks((axis_start[2], axis_start[2]+axis_range))
