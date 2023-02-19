import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mtick
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import datetime
from . import save
import statsmodels.api as sm
from . import Reslib

plt.rc('font', family='Times New Roman')


def plot_2d_sample_space(raw_saver,
                         variable_id1,
                         variable_id2,
                         obj_path="trace2d.pdf",
                         figsize=(8, 8),
                         dpi=400,
                         title=" ",
                         lbl_fontsize=16,
                         marker=".",
                         markersize=50,
                         x_label="Variable1",
                         y_label="Variable2",
                         tick_fontsize=14,
                         tick_dir="out",
                         framewidth=1.2,
                         cmap="plasma",
                         single_c="b",
                         cbar_ttl="Iterations",
                         cbar_ttl_size=16,
                         cbar_lbl_size=12,
                         cbar_frac=0.05,
                         cbar_pad=0.05,
                         slim=True,
                         tight=True,
                         show=True
                         ):
    """
    This function is for the users to plot the 2D sample searching space (two variables can be considered).

    :argument
    raw_saver: caliboy.save.RawDataSaver objective
    variable_id1: the index of the first variable (x) -> int (column index, the sequence is same as what in lb and ub)
    variable_id2: the index of the second variable (y) -> int
    obj_path: target saving path -> str, default = "trace2d.pdf", acceptable format = ["jpg", "png", "pdf", ...]
    figsize: the x, y size of the figure -> tuple, default = (8, 8)
    dpi: the resolution of the figure -> int, default = 400
    title: the title of the figure -> str
    lbl_fontsize: fontsize of axis labels -> int, default = 16
    marker: type of the markers in the scatter -> str, default = "." (dot like)
    markersize: the size of markers -> int, default = 50
    x_label: the x axis label -> str, default ="Variable1"
    y_label: the y axis label -> str, default ="Variable2"
    tick_fontsize: fontsize of tick labels -> int or float, default = 14
    tick_dir: tick direction -> str ("in" or "out"), default = "out"
    framewidth: width of the frame line -> float, default = 1.2
    cmap: color map -> str, default = "plasma"
    single_c: color -> str, default = "b", this will be used if the optimization method is GLUE (no iteration)
    cbar_ttl: the title of color bar -> str, default = "Iterations", users can use " " to replace it
    cbar_ttl_size: the size of color bar title -> int or float, default = 16
    cbar_lbl_size: the size of color bar tick labels -> int or float, default = 12
    cbar_frac: the size of color bar -> float, default = 0.05
    cbar_pad: the gap between cbar and the main figure -> float, default = 0.05
    slim: the plotting range option -> bool, default = True, if True, the plotting range will be confined to the
          value ranges of the given data.
    tight: option for the white frame outside the plotting area -> bool, default = True, tight output (slim white frame)
    show: option for showing in the IDE -> bool, default = True

    Usage:
    from pycup import save, plot

    opt_res = save.RawDataSaver.load(r"RawResult.rst")
    plot.plot_2d_sample_space(opt_res,variable_id1=0,variable_id2=1,obj_path="sample_space.jpg")

    Note:
    selectable cmaps includes [('Perceptually Uniform Sequential',
                            ['viridis', 'inferno', 'plasma', 'magma']),
         ('Sequential',     ['Blues', 'BuGn', 'BuPu',
                             'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
                             'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
                             'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
         ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool',
                             'copper', 'gist_heat', 'gray', 'hot',
                             'pink', 'spring', 'summer', 'winter']),
         ('Diverging',      ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
                             'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
                             'seismic']),
         ('Qualitative',    ['Accent', 'Dark2', 'Paired', 'Pastel1',
                             'Pastel2', 'Set1', 'Set2', 'Set3']),
         ('Miscellaneous',  ['gist_earth', 'terrain', 'ocean', 'gist_stern',
                             'brg', 'CMRmap', 'cubehelix',
                             'gnuplot', 'gnuplot2', 'gist_ncar',
                             'nipy_spectral', 'jet', 'rainbow',
                             'gist_rainbow', 'hsv', 'flag', 'prism'])]
    """
    if not isinstance(raw_saver, save.RawDataSaver):
        raise TypeError("The input saver object should be pycup.save.RawDataSaver.")

    iterations, population = analyze_saver(raw_saver)
    if iterations > 1:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        samples = np.array(raw_saver.historical_samples, dtype=object)
        a = []
        c = np.linspace(0, iterations, iterations)
        for i, sample in enumerate(samples):
            ai = np.ones(samples[i].shape[0])
            ai = ai * c[i]
            a.append(ai)
        samples = np.concatenate(samples)
        a = np.concatenate(a)
        s = plt.scatter(samples[:, variable_id1], samples[:, variable_id2], c=a,
                        marker=marker, s=markersize, cmap=cmap)

        bar = plt.colorbar(fraction=cbar_frac, pad=cbar_pad)
        bar.set_label(cbar_ttl, fontsize=cbar_ttl_size)
        bar.ax.tick_params(labelsize=cbar_lbl_size)
    else:
        samples = np.array(raw_saver.historical_samples)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        s = plt.scatter(samples[:, variable_id1], samples[:, variable_id2], c=single_c,
                        marker=marker, s=markersize, cmap=cmap)
    if slim:
        xmin = np.min(samples[:, variable_id1], axis=0)
        ymin = np.min(samples[:, variable_id2], axis=0)
        xmax = np.max(samples[:, variable_id1], axis=0)
        ymax = np.max(samples[:, variable_id2], axis=0)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(framewidth)
    ax.spines['top'].set_linewidth(framewidth)
    ax.spines['right'].set_linewidth(framewidth)
    ax.spines['left'].set_linewidth(framewidth)
    ax.tick_params(
        labelsize=tick_fontsize,  # y轴字体大小设置
        direction=tick_dir  # y轴标签方向设置
    )

    plt.xlabel(x_label, fontsize=lbl_fontsize)
    plt.ylabel(y_label, fontsize=lbl_fontsize)
    plt.title(title)
    if tight:
        plt.savefig(obj_path, bbox_inches='tight')
    else:
        plt.savefig(obj_path)
    if show:
        plt.show()


def plot_3d_sample_space(raw_saver,
                         variable_id1,
                         variable_id2,
                         variable_id3,
                         obj_path="trace3d.pdf",
                         figsize=(8, 8),
                         dpi=400,
                         title=" ",
                         lbl_fontsize=16,
                         marker=".",
                         markersize=50,
                         x_label="Variable1",
                         y_label="Variable2",
                         z_label="Variable3",
                         tick_fontsize=12,
                         tick_dir="out",
                         cmap="plasma",
                         single_c="b",
                         cbar_ttl="Iterations",
                         cbar_ttl_size=16,
                         cbar_lbl_size=12,
                         cbar_frac=0.03,
                         cbar_pad=0.05,
                         view_init=(),
                         slim=True,
                         tight=True,
                         show=True):
    """
    This function is for the users to plot the 3D sample searching space (three variables can be considered).

    :argument
    raw_saver: caliboy.save.RawDataSaver objective
    variable_id1: the index of the first variable (x) -> int (column index, the sequence is same as what in lb and ub)
    variable_id2: the index of the second variable (y) -> int
    variable_id3: the index of the third variable (z) -> int
    obj_path: target saving path -> str, default = "trace3d.pdf", acceptable format = ["jpg", "png", "pdf", ...]
    figsize: the x, y size of the figure -> tuple, default = (8, 8)
    dpi: the resolution of the figure -> int, default = 400
    title: the title of the figure -> str
    lbl_fontsize: fontsize of axis labels -> int, default = 16
    marker: type of the markers in the scatter -> str, default = "." (dot like)
    markersize: the size of markers -> int or float, default = 50
    x_label: the x axis label -> str, default ="Variable1"
    y_label: the y axis label -> str, default ="Variable2"
    z_label: the z axis label -> str, default ="Variable3"
    tick_fontsize: fontsize of tick labels -> int or float, default = 14
    tick_dir: tick direction -> str ("in" or "out"), default = "out"
    framewidth: width of the frame line -> float, default = 1.2
    cmap: color map -> str, default = "plasma"
    single_c: color -> str, default = "b", this will be used if the optimization method is GLUE (no iteration)
    cbar_ttl: the title of color bar -> str, default = "Iterations", users can use " " to replace it
    cbar_ttl_size: the size of color bar title -> int or float, default = 16
    cbar_lbl_size: the size of color bar tick labels -> int or float, default = 12
    cbar_frac: the size of color bar -> float, default = 0.03
    cbar_pad: the gap between cbar and the main figure -> float, default = 0.05
    view_init: view angle -> tuple, (float, float)
    slim: the plotting range option -> bool, default = True, if True, the plotting range will be confined to the
          value ranges of the given data.
    tight: option for the white frame outside the plotting area -> bool, default = True, tight output (slim white frame)
    show: option for showing in the IDE -> bool, default = True

    Usage:
    from pycup import save, plot

    opt_res = save.RawDataSaver.load(r"RawResult.rst")
    plot.plot_3d_sample_space(opt_res,variable_id1=0,variable_id2=1,variable_id3=2,obj_path="sample_space.jpg")

    Note:
    selectable cmaps includes [('Perceptually Uniform Sequential',
                            ['viridis', 'inferno', 'plasma', 'magma']),
         ('Sequential',     ['Blues', 'BuGn', 'BuPu',
                             'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
                             'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
                             'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
         ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool',
                             'copper', 'gist_heat', 'gray', 'hot',
                             'pink', 'spring', 'summer', 'winter']),
         ('Diverging',      ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
                             'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
                             'seismic']),
         ('Qualitative',    ['Accent', 'Dark2', 'Paired', 'Pastel1',
                             'Pastel2', 'Set1', 'Set2', 'Set3']),
         ('Miscellaneous',  ['gist_earth', 'terrain', 'ocean', 'gist_stern',
                             'brg', 'CMRmap', 'cubehelix',
                             'gnuplot', 'gnuplot2', 'gist_ncar',
                             'nipy_spectral', 'jet', 'rainbow',
                             'gist_rainbow', 'hsv', 'flag', 'prism'])]
    """
    if not isinstance(raw_saver, save.RawDataSaver):
        raise TypeError("The input saver object should be pycup.save.RawDataSaver.")

    iterations, population = analyze_saver(raw_saver)
    if iterations > 1:
        samples = np.array(raw_saver.historical_samples, dtype=object)
        a = []
        c = np.linspace(0, iterations, iterations)
        for i, sample in enumerate(samples):
            ai = np.ones(samples[i].shape[0])
            ai = ai * c[i]
            a.append(ai)
        samples = np.concatenate(samples)
        a = np.concatenate(a)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        # The tested version should use the tolist() method to avoid the Invalid RGBA error
        s = ax.scatter(xs=samples[:, variable_id1].tolist(), ys=samples[:, variable_id2].tolist(),
                       zs=samples[:, variable_id3].tolist(), c=a, marker=marker, s=markersize, cmap=cmap)
        bar = plt.colorbar(s, fraction=cbar_frac, pad=cbar_pad)
        bar.set_label(cbar_ttl, fontsize=cbar_ttl_size)
        bar.ax.tick_params(labelsize=cbar_lbl_size)
    else:
        samples = np.array(raw_saver.historical_samples)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        s = ax.scatter(xs=samples[:, variable_id1], ys=samples[:, variable_id2], zs=samples[:, variable_id3],
                       c=single_c, marker=marker, s=markersize, cmap=cmap)
    ax.view_init(*view_init)
    if slim:
        xmin = np.min(samples[:, variable_id1], axis=0)
        ymin = np.min(samples[:, variable_id2], axis=0)
        zmin = np.min(samples[:, variable_id3], axis=0)
        xmax = np.max(samples[:, variable_id1], axis=0)
        ymax = np.max(samples[:, variable_id2], axis=0)
        zmax = np.max(samples[:, variable_id3], axis=0)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
    ax.set_xlabel(x_label, fontsize=lbl_fontsize)
    ax.set_ylabel(y_label, fontsize=lbl_fontsize)
    ax.set_zlabel(z_label, fontsize=lbl_fontsize)
    ax.tick_params(
        labelsize=tick_fontsize,
        direction=tick_dir
    )
    plt.title(title)

    if tight:
        plt.savefig(obj_path, bbox_inches='tight')
    else:
        plt.savefig(obj_path)
    if show:
        plt.show()


def plot_2d_fitness_space(raw_saver,
                          variable_id,
                          fitness_id=0,
                          obj_path="fitness2d.pdf",
                          figsize=(8, 8),
                          dpi=400,
                          title=" ",
                          lbl_fontsize=16,
                          marker=".",
                          markersize=50,
                          x_label="Variable1",
                          y_label="Fitness",
                          tick_fontsize=14,
                          tick_dir="out",
                          framewidth=1.2,
                          cmap="plasma",
                          single_c="b",
                          cbar_ttl="Iterations",
                          cbar_ttl_size=16,
                          cbar_lbl_size=12,
                          cbar_frac=0.05,
                          cbar_pad=0.05,
                          x_lim=(),
                          y_lim=(),
                          slim=True,
                          tight=True,
                          show=True
                          ):
    """
    This function is for the users to plot the 2D sample's fitness space (sample value versus fitness value,
    one variable can be considered).

    :argument
    raw_saver: caliboy.save.RawDataSaver objective
    variable_id: the index of the variable (x) -> int (column index, the sequence is same as what in lb and ub)
    obj_path: target saving path -> str, default = "fitness2d.pdf", acceptable format = ["jpg", "png", "pdf", ...]
    figsize: the x, y size of the figure -> tuple, default = (8, 8)
    dpi: the resolution of the figure -> int, default = 400
    title: the title of the figure -> str
    lbl_fontsize: fontsize of axis labels -> int, default = 16
    marker: type of the markers in the scatter -> str, default = "." (dot like)
    markersize: the size of markers -> int or float, default = 50
    x_label: the x axis label -> str, default ="Variable1"
    y_label: the y axis label -> str, default ="Fitness"
    tick_fontsize: fontsize of tick labels -> int or float, default = 14
    tick_dir: tick direction -> str ("in" or "out"), default = "out"
    framewidth: width of the frame line -> float, default = 1.2
    cmap: color map -> str, default = "plasma"
    single_c: color -> str, default = "b", this will be used if the optimization method is GLUE (no iteration)
    cbar_ttl: the title of color bar -> str, default = "Iterations", users can use " " to replace it
    cbar_ttl_size: the size of color bar title -> int or float, default = 16
    cbar_lbl_size: the size of color bar tick labels -> int or float, default = 12
    cbar_frac: the size of color bar -> float, default = 0.05
    cbar_pad: the gap between cbar and the main figure -> float, default = 0.05
    x_lim: x axis plotting range -> tuple
    y_lim: y axis plotting range -> tuple
    slim: the plotting range option -> bool, default = True, if True, the plotting range will be confined to the
          value ranges of the given data.
    tight: option for the white frame outside the plotting area -> bool, default = True, tight output (slim white frame)
    show: option for showing in the IDE -> bool, default = True

    Usage:
    from pycup import save, plot

    opt_res = save.RawDataSaver.load(r"RawResult.rst")
    plot.plot_2d_fitness_space(opt_res,variable_id=1,y_lim=(0,))

    Note:
    selectable cmaps includes [('Perceptually Uniform Sequential',
                            ['viridis', 'inferno', 'plasma', 'magma']),
         ('Sequential',     ['Blues', 'BuGn', 'BuPu',
                             'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
                             'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
                             'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
         ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool',
                             'copper', 'gist_heat', 'gray', 'hot',
                             'pink', 'spring', 'summer', 'winter']),
         ('Diverging',      ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
                             'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
                             'seismic']),
         ('Qualitative',    ['Accent', 'Dark2', 'Paired', 'Pastel1',
                             'Pastel2', 'Set1', 'Set2', 'Set3']),
         ('Miscellaneous',  ['gist_earth', 'terrain', 'ocean', 'gist_stern',
                             'brg', 'CMRmap', 'cubehelix',
                             'gnuplot', 'gnuplot2', 'gist_ncar',
                             'nipy_spectral', 'jet', 'rainbow',
                             'gist_rainbow', 'hsv', 'flag', 'prism'])]
    """
    if not isinstance(raw_saver, save.RawDataSaver):
        raise TypeError("The input saver object should be pycup.save.RawDataSaver.")

    iterations, population = analyze_saver(raw_saver)
    if iterations > 1:
        samples = np.array(raw_saver.historical_samples, dtype=object)
        fitness = np.array(raw_saver.historical_fitness, dtype=object)
        a = []
        c = np.linspace(0, iterations, iterations)
        for i, sample in enumerate(samples):
            ai = np.ones(samples[i].shape[0])
            ai = ai * c[i]
            a.append(ai)
        samples = np.concatenate(samples)
        fitness = np.concatenate(fitness)
        a = np.concatenate(a)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        s = plt.scatter(samples[:, variable_id], fitness[:, fitness_id], c=a,
                        marker=marker, s=markersize, cmap=cmap)

        bar = plt.colorbar(fraction=cbar_frac, pad=cbar_pad)
        bar.set_label(cbar_ttl, fontsize=cbar_ttl_size)
        bar.ax.tick_params(labelsize=cbar_lbl_size)
    else:
        samples = np.array(raw_saver.historical_samples)
        fitness = np.array(raw_saver.historical_fitness)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        s = plt.scatter(samples[:, variable_id], fitness[:, fitness_id], c=single_c,
                        marker=marker, s=markersize, cmap=cmap)
    if slim:
        xmin = np.min(samples[:, variable_id], axis=0)
        ymin = np.min(fitness[:, 0], axis=0)
        xmax = np.max(samples[:, variable_id], axis=0)
        ymax = np.max(fitness[:, 0], axis=0)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
    if x_lim:
        plt.xlim(*x_lim)
    if y_lim:
        plt.ylim(*y_lim)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(framewidth)
    ax.spines['top'].set_linewidth(framewidth)
    ax.spines['right'].set_linewidth(framewidth)
    ax.spines['left'].set_linewidth(framewidth)
    ax.tick_params(
        labelsize=tick_fontsize,
        direction=tick_dir
    )
    ax.ticklabel_format(style='sci', scilimits=(-3, 4), axis='y', useMathText=True)
    plt.xlabel(x_label, fontsize=lbl_fontsize)
    plt.ylabel(y_label, fontsize=lbl_fontsize)
    plt.title(title)
    if tight:
        plt.savefig(obj_path, bbox_inches='tight')
    else:
        plt.savefig(obj_path)
    if show:
        plt.show()


def plot_2d_behaviour_fitness(proc_saver,
                              variable_id,
                              fitness_id=0,
                              obj_path="fitness2d.pdf",
                              figsize=(8, 8),
                              dpi=400,
                              title=" ",
                              lbl_fontsize=16,
                              marker=".",
                              markersize=50,
                              x_label="Variable1",
                              y_label="Fitness",
                              tick_fontsize=14,
                              tick_dir="out",
                              framewidth=1.2,
                              c="b",
                              x_lim=(),
                              y_lim=(),
                              slim=True,
                              tight=True,
                              show=True
                              ):
    """
    This function is for the users to plot the 2D sample's fitness space (sample value versus fitness value,
    one variable can be considered).

    :argument
    raw_saver: caliboy.save.ProcResultSaver objective
    variable_id: the index of the variable (x) -> int (column index, the sequence is same as what in lb and ub)
    obj_path: target saving path -> str, default = "fitness2d.pdf", acceptable format = ["jpg", "png", "pdf", ...]
    figsize: the x, y size of the figure -> tuple, default = (8, 8)
    dpi: the resolution of the figure -> int, default = 400
    title: the title of the figure -> str
    lbl_fontsize: fontsize of axis labels -> int, default = 16
    marker: type of the markers in the scatter -> str, default = "." (dot like)
    markersize: the size of markers -> int or float, default = 50
    x_label: the x axis label -> str, default ="Variable1"
    y_label: the y axis label -> str, default ="Fitness"
    tick_fontsize: fontsize of tick labels -> int or float, default = 14
    tick_dir: tick direction -> str ("in" or "out"), default = "out"
    framewidth: width of the frame line -> float, default = 1.2
    c: color -> str, default = "b", this will be used if the optimization method is GLUE (no iteration)
    x_lim: x axis plotting range -> tuple
    y_lim: y axis plotting range -> tuple
    slim: the plotting range option -> bool, default = True, if True, the plotting range will be confined to the
          value ranges of the given data.
    tight: option for the white frame outside the plotting area -> bool, default = True, tight output (slim white frame)
    show: option for showing in the IDE -> bool, default = True

    Usage:
    from pycup import save, plot

    opt_res = save.RawDataSaver.load(r"RawResult.rst")
    plot.plot_2d_behaviour_fitness(opt_res,variable_id=1,y_lim=(0,))
    """
    if not isinstance(proc_saver, save.ProcResultSaver):
        raise TypeError("The input saver object should be pycup.save.ProcResultSaver.")
    samples = np.array(proc_saver.behaviour_results.behaviour_samples)
    fitness = np.array(proc_saver.behaviour_results.behaviour_fitness)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    s = plt.scatter(samples[:, variable_id], fitness[:, fitness_id], c=c,
                    marker=marker, s=markersize)
    if slim:
        xmin = np.min(samples[:, variable_id], axis=0)
        ymin = np.min(fitness[:, 0], axis=0)
        xmax = np.max(samples[:, variable_id], axis=0)
        ymax = np.max(fitness[:, 0], axis=0)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
    if x_lim:
        plt.xlim(*x_lim)
    if y_lim:
        plt.ylim(*y_lim)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(framewidth)
    ax.spines['top'].set_linewidth(framewidth)
    ax.spines['right'].set_linewidth(framewidth)
    ax.spines['left'].set_linewidth(framewidth)
    ax.tick_params(
        labelsize=tick_fontsize,
        direction=tick_dir
    )
    ax.ticklabel_format(style='sci', scilimits=(-3, 4), axis='y', useMathText=True)
    plt.xlabel(x_label, fontsize=lbl_fontsize)
    plt.ylabel(y_label, fontsize=lbl_fontsize)
    plt.title(title)
    if tight:
        plt.savefig(obj_path, bbox_inches='tight')
    else:
        plt.savefig(obj_path)
    if show:
        plt.show()


def plot_3d_fitness_space(raw_saver,
                          variable_id1,
                          variable_id2,
                          fitness_id=0,
                          obj_path="fitness3d.pdf",
                          figsize=(8, 8),
                          dpi=400,
                          title=" ",
                          lbl_fontsize=16,
                          marker=".",
                          markersize=50,
                          x_label="Variable1",
                          y_label="Variable2",
                          z_label="Fitness",
                          tick_fontsize=14,
                          tick_dir="out",
                          cmap="plasma",
                          single_c="b",
                          cbar_ttl="Iterations",
                          cbar_ttl_size=16,
                          cbar_lbl_size=12,
                          cbar_frac=0.03,
                          cbar_pad=0.05,
                          view_init=(),
                          x_lim=(),
                          y_lim=(),
                          z_lim=(),
                          slim=True,
                          tight=True,
                          show=True
                          ):
    """
    This function is for the users to plot the 3D sample's fitness space (sample value versus fitness value,
    two variables can be considered).

    :argument
    raw_saver: caliboy.save.RawDataSaver objective
    variable_id1: the index of the first variable (x1) -> int (column index, the sequence is same as what in lb and ub)
    variable_id2: the index of the first variable (x2) -> int
    obj_path: target saving path -> str, default = "fitness3d.pdf", acceptable format = ["jpg", "png", "pdf", ...]
    figsize: the x, y size of the figure -> tuple, default = (8, 8)
    dpi: the resolution of the figure -> int, default = 400
    title: the title of the figure -> str
    lbl_fontsize: fontsize of axis labels -> int or float, default = 16
    marker: type of the markers in the scatter -> str, default = "." (dot like)
    markersize: the size of markers -> int or float, default = 50
    x_label: the x axis label -> str, default ="Variable1"
    y_label: the y axis label -> str, default ="Variable2"
    z_label: the z axis label -> str, default ="Fitness"
    tick_fontsize: fontsize of tick labels -> int or float, default = 14
    tick_dir: tick direction -> str ("in" or "out"), default = "out"
    framewidth: width of the frame line -> float, default = 1.2
    cmap: color map -> str, default = "plasma"
    single_c: color -> str, default = "b", this will be used if the optimization method is GLUE (no iteration)
    cbar_ttl: the title of color bar -> str, default = "Iterations", users can use " " to replace it
    cbar_ttl_size: the size of color bar title -> int or float, default = 16
    cbar_lbl_size: the size of color bar tick labels -> int or float, default = 12
    cbar_frac: the size of color bar -> float, default = 0.05
    cbar_pad: the gap between cbar and the main figure -> float, default = 0.05
    view_init: view angle -> tuple, (float, float)
    x_lim: x axis plotting range -> tuple
    y_lim: y axis plotting range -> tuple
    z_lim: z axis plotting range -> tuple
    slim: the plotting range option -> bool, default = True, if True, the plotting range will be confined to the
          value ranges of the given data.
    tight: option for the white frame outside the plotting area -> bool, default = True, tight output (slim white frame)
    show: option for showing in the IDE -> bool, default = True

    Usage:
    from pycup import save, plot

    opt_res = save.RawDataSaver.load(r"RawResult.rst")
    plot.plot_3d_fitness_space(opt_res,variable_id1=1,variable_id2=2,obj_path="sample_space.jpg")

    Note:
    selectable cmaps includes [('Perceptually Uniform Sequential',
                            ['viridis', 'inferno', 'plasma', 'magma']),
         ('Sequential',     ['Blues', 'BuGn', 'BuPu',
                             'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
                             'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
                             'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
         ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool',
                             'copper', 'gist_heat', 'gray', 'hot',
                             'pink', 'spring', 'summer', 'winter']),
         ('Diverging',      ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
                             'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
                             'seismic']),
         ('Qualitative',    ['Accent', 'Dark2', 'Paired', 'Pastel1',
                             'Pastel2', 'Set1', 'Set2', 'Set3']),
         ('Miscellaneous',  ['gist_earth', 'terrain', 'ocean', 'gist_stern',
                             'brg', 'CMRmap', 'cubehelix',
                             'gnuplot', 'gnuplot2', 'gist_ncar',
                             'nipy_spectral', 'jet', 'rainbow',
                             'gist_rainbow', 'hsv', 'flag', 'prism'])]
    """
    if not isinstance(raw_saver, save.RawDataSaver):
        raise TypeError("The input saver object should be pycup.save.RawDataSaver.")

    iterations, population = analyze_saver(raw_saver)
    if iterations > 1:
        samples = np.array(raw_saver.historical_samples, dtype=object)
        fitness = np.array(raw_saver.historical_fitness, dtype=object)
        a = []
        c = np.linspace(0, iterations, iterations)
        for i, sample in enumerate(samples):
            ai = np.ones(samples[i].shape[0])
            ai = ai * c[i]
            a.append(ai)
        samples = np.concatenate(samples)
        fitness = np.concatenate(fitness)
        a = np.concatenate(a)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        s = ax.scatter(xs=samples[:, variable_id1].tolist(), ys=samples[:, variable_id2].tolist(),
                       zs=fitness[:, fitness_id].tolist(), c=a,
                       marker=marker, s=markersize, cmap=cmap)
        bar = plt.colorbar(s, fraction=cbar_frac, pad=cbar_pad)
        bar.set_label(cbar_ttl, fontsize=cbar_ttl_size)
        bar.ax.tick_params(labelsize=cbar_lbl_size)

    else:
        samples = np.array(raw_saver.historical_samples)
        fitness = np.array(raw_saver.historical_fitness)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        s = ax.scatter(xs=samples[:, variable_id1], ys=samples[:, variable_id2], zs=fitness[:, fitness_id], c=single_c,
                       marker=marker, s=markersize, cmap=cmap)
    ax.view_init(*view_init)
    if slim:
        xmin = np.min(samples[:, variable_id1], axis=0)
        ymin = np.min(samples[:, variable_id2], axis=0)
        zmin = np.min(fitness[:, 0], axis=0)
        xmax = np.max(samples[:, variable_id1], axis=0)
        ymax = np.max(samples[:, variable_id2], axis=0)
        zmax = np.max(fitness[:, 0], axis=0)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
    if x_lim:
        ax.set_xlim(*x_lim)
    if y_lim:
        ax.set_ylim(*y_lim)
    if z_lim:
        ax.set_zlim(*z_lim)
    ax = plt.gca()

    ax.tick_params(
        labelsize=tick_fontsize,
        direction=tick_dir
    )

    ax.set_xlabel(x_label, fontsize=lbl_fontsize)
    ax.set_ylabel(y_label, fontsize=lbl_fontsize)
    ax.set_zlabel(z_label, fontsize=lbl_fontsize)
    ax.ticklabel_format(style='sci', scilimits=(-3, 4), axis='z', useMathText=True)
    plt.title(title)
    if tight:
        plt.savefig(obj_path, bbox_inches='tight')
    else:
        plt.savefig(obj_path)
    if show:
        plt.show()


def plot_uncertainty_band(proc_saver,
                          obsx,
                          obsy,
                          station=None,
                          event=None,
                          obj_path="band.pdf",
                          dpi=400,
                          figsize=(8, 4),
                          framewidth=1.2,
                          title=" ",
                          ticklocs=None,
                          xticks=None,
                          lbl_fontsize=16,
                          tick_fontsize=14,
                          tick_dir="out",
                          xtick_rotation=0,
                          ytick_rotation=0,
                          bestlinewidth=1,
                          bestlinestyle="-",
                          medianlinestyle="-",
                          medianlinewidth=1,
                          bandlinewidth=1,
                          bandlinestyle="-",
                          marker=".",
                          markersize=50,
                          x_label="Step",
                          y_label="Value",
                          ppulabel="95PPU",
                          legend_on=True,
                          legendloc=0,
                          legend_fontsize=14,
                          pad=(0.1, 0.15, 0.9, 0.85),
                          idx=None,
                          ylim=None,
                          frameon=False,
                          slim=True,
                          show=True,
                          tight=False,
                          draw_best=True,
                          draw_median=True,
                          twin_x=False,
                          twin_data=None,
                          twin_ylim=None,
                          twin_ylabel="Value2"):
    """
    This function is for the users to plot the uncertainty band of time-series prediction.

    :argument
    proc_saver: pycup.save.ProcResultSaver object (uncertainty analysis result)
    obsx: the observation time stamp x data array -> np.array
    obsy: the observation y data array -> np.array
    obj_path: target saving path -> str, default = "band.pdf", acceptable format = ["jpg", "png", "pdf", ...]
    dpi: the resolution of the figure -> int, default = 400
    figsize: the x, y size of the figure -> tuple, default = (8, 8)
    framewidth: width of the frame line -> float, default = 1.2
    title: the title of the figure -> str
    ticklocs: tick locations -> array like, default = None, users can specify the time step that they want to be plotted
    xticks: tick labels -> array like, default = None, users can specify the tick label contents (e.g. date str)
    lbl_fontsize: fontsize of axis labels -> int or float, default = 16
    tick_fontsize: fontsize of tick labels -> int or float, default = 14
    tick_dir: tick direction -> str ("in" or "out"), default = "out"
    xtick_rotation: x-tick label direction angel -> float, default = 0
    ytick_rotation: y-tick label direction angel -> float, default = 0
    bestlinewidth: the width of the best simulation result line -> float, default = 1
    bestlinestyle: the style of the best simulation result line -> str, default = "-"
    bandlinewidth: the width of the upper and lower uncertainty band lines -> float, default = 1
    bandlinestyle: the style of the upper and lower uncertainty band lines -> str, default = "-"
    marker: the marker style of the observation points -> str, default = "."
    markersize: the size of markers -> int or float, default = 50
    x_label: the x axis label -> str, default ="Step"
    y_label: the y axis label -> str, default ="Value"
    ppulabel: the uncertainty band legend label -> str, default = "95PPU"
    legendloc: the location of the legend -> int (0-8), default = 0
    legend_fontsize: the fontsize of phrases in the legend -> int or float, default = 14
    pad: the extent of the plotting area -> tuple, default = (0.1,0.15,0.9,0.85) left bottom right top
    idx: a list/tuple of x indices to draw only a part of the whole series. e.g. idx = [0,100]
    ylim: the y axis plotting value range -> tuple, default = None
    frameon: the switch for opening the legend frame -> bool, default = False
    slim: the plotting range option -> bool, default = True, if True, the plotting range will be confined to the
          value ranges of the given data.
    tight: option for the white frame outside the plotting area -> bool, default = True, tight output (slim white frame)
    show: option for showing in the IDE -> bool, default = True
    draw_best: switch for drawing the best simulation result line -> bool, default = True

    Usage:
    from pycup import save, plot
    import numpy as np
    import random

    # load the result
    opt_res = save.ProcResultSaver.load(r"ProcResult.rst")

    # simulate an observation data array (in this example)
    t = np.arange(0,100)
    res = []
    for i in t:
        obs = np.sin(i) * (2.1 + np.random.randn()) + np.cos(i) * (5.2 + np.random.randn()) ** 3 + random.random()
        res.append(obs)

    # plot the uncertainty band
    plot.plot_uncertainty_band(opt_res, t, res, ylim=(-600, 600), ticklocs=np.arange(0, 100, 10),legendloc=1)
    """
    if isinstance(proc_saver, save.ProcResultSaver):

        if isinstance(proc_saver.uncertain_results.ppu_line_lower, Reslib.SimulationResult):
            if station and event:
                proc_saver.extract_station_event(station, event)
            else:
                raise ValueError(
                    "The Reslib has been used, therefore, the station name and event name should be given.")

        if not idx:
            ppu_lower = proc_saver.uncertain_results.ppu_line_lower.flatten()
            ppu_upper = proc_saver.uncertain_results.ppu_line_upper.flatten()
            if draw_best and isinstance(proc_saver.best_result.best_results, np.ndarray):
                best_result = proc_saver.best_result.best_results.flatten()
            else:
                best_result = None
        else:
            ppu_lower = proc_saver.uncertain_results.ppu_line_lower.flatten()
            ppu_upper = proc_saver.uncertain_results.ppu_line_upper.flatten()
            ppu_lower = ppu_lower[idx[0]:idx[1]]
            ppu_upper = ppu_upper[idx[0]:idx[1]]
            if draw_best and isinstance(proc_saver.best_result.best_results, np.ndarray):
                best_result = proc_saver.best_result.best_results.flatten()
                best_result = best_result[idx[0]:idx[1]]
            else:
                best_result = None
        if len(ppu_lower) < 2 or len(ppu_upper) < 2:
            raise ValueError(
                "The uncertainty band plotting function does not accept the calculation result with a length less than 2.")
    elif isinstance(proc_saver, save.ValidationProcSaver):

        if isinstance(proc_saver.ppu_lower, Reslib.SimulationResult):
            if station and event:
                proc_saver.extract_station_event(station, event)
            else:
                raise ValueError(
                    "The Reslib has been used, therefore, the station name and event name should be given.")

        if not idx:
            ppu_lower = proc_saver.ppu_lower.flatten()
            ppu_upper = proc_saver.ppu_upper.flatten()
            if draw_best and isinstance(proc_saver.best_result, np.ndarray):
                best_result = proc_saver.best_result.flatten()
            else:
                best_result = None
        else:
            ppu_lower = proc_saver.ppu_lower.flatten()
            ppu_upper = proc_saver.ppu_upper.flatten()
            ppu_lower = ppu_lower[idx[0]:idx[1]]
            ppu_upper = ppu_upper[idx[0]:idx[1]]
            if draw_best and isinstance(proc_saver.best_result, np.ndarray):
                best_result = proc_saver.best_result[idx[0]:idx[1]]
            else:
                best_result = None
        if len(ppu_lower) < 2 or len(ppu_upper) < 2:
            raise ValueError(
                "The uncertainty band plotting function does not accept the calculation result with a length less than 2.")
    elif isinstance(proc_saver, save.PredProcSaver):

        if isinstance(proc_saver.ppu_lower, Reslib.SimulationResult):
            if station and event:
                proc_saver.extract_station_event(station, event)
            else:
                raise ValueError(
                    "The Reslib has been used, therefore, the station name and event name should be given.")

        if not idx:
            ppu_lower = proc_saver.ppu_lower.flatten()
            ppu_upper = proc_saver.ppu_upper.flatten()
            draw_best = False
            best_result = None
        else:
            ppu_lower = proc_saver.ppu_lower.flatten()
            ppu_upper = proc_saver.ppu_upper.flatten()
            ppu_lower = ppu_lower[idx[0]:idx[1]]
            ppu_upper = ppu_upper[idx[0]:idx[1]]
            draw_best = False
            best_result = None
        if len(ppu_lower) < 2 or len(ppu_upper) < 2:
            raise ValueError(
                "The uncertainty band plotting function does not accept the calculation result with a length less than 2.")
    else:
        raise TypeError("The input saver should be ProcResultSaver,ValidationProcSaver or PredProcSaver.")

    x = np.arange(len(ppu_lower))
    fig = plt.figure(dpi=dpi, figsize=figsize)
    plt.plot(x, ppu_lower, color="dimgray", zorder=8, label=ppulabel, linewidth=bandlinewidth, linestyle=bandlinestyle)
    plt.plot(x, ppu_upper, color="dimgray", zorder=8, linewidth=bandlinewidth, linestyle=bandlinestyle)
    plt.fill_between(x, ppu_lower, ppu_upper, facecolor="lightgray", alpha=0.5)
    if draw_best and isinstance(best_result, np.ndarray):
        plt.plot(best_result, color="red", zorder=9, label="Best simulation", linewidth=bestlinewidth,
                 linestyle=bestlinestyle)
    if draw_median:
        median_prediction = proc_saver.median_prediction.flatten()
        if not idx:
            plt.plot(median_prediction, color="green", zorder=8, label="Median prediction", linewidth=medianlinewidth,
                     linestyle=medianlinestyle)
        else:
            median_pred = median_prediction[idx[0]: idx[1]]
            plt.plot(median_pred, color="green", zorder=8, label="Median prediction", linewidth=medianlinewidth,
                     linestyle=medianlinestyle)
    if obsy is not None and obsx is not None:
        plt.scatter(obsx, obsy, color="royalblue", marker=marker, s=markersize, zorder=10, label="Observations")
    if legend_on:
        plt.legend(loc=legendloc, frameon=frameon, prop={'size': legend_fontsize})
    plt.xlabel(x_label, fontsize=lbl_fontsize)
    plt.ylabel(y_label, fontsize=lbl_fontsize)
    plt.title = title

    plt.xticks(ticklocs, xticks, rotation=xtick_rotation)
    plt.yticks(rotation=ytick_rotation)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(framewidth)
    ax.spines['top'].set_linewidth(framewidth)
    ax.spines['right'].set_linewidth(framewidth)
    ax.spines['left'].set_linewidth(framewidth)
    ax.tick_params(
        labelsize=tick_fontsize,
        direction=tick_dir,
    )
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    # rain data
    if twin_x:
        supported_colors = ['r', 'g', 'b', 'y', "brown", "gray", "magenta", "cyan", "orange", "purple"]
        ax2 = ax.twinx()
        ax2.invert_yaxis()
        if isinstance(twin_data, list) or isinstance(twin_data, tuple):
            for i in twin_data:
                ax2.bar(np.arange(len(i)), i, color=supported_colors[i], alpha=0.3)
        else:
            ax2.bar(np.arange(len(twin_data)), twin_data, color='g', alpha=0.3)
        if twin_ylim:
            ax2.set_ylim(twin_ylim)
        ax2.set_ylabel(twin_ylabel, fontsize=lbl_fontsize)
        ax2.tick_params(
            labelsize=tick_fontsize,
            direction=tick_dir,
        )

    plt.subplots_adjust(left=pad[0], bottom=pad[1], right=pad[2], top=pad[3])

    if slim:
        plt.xlim(0, len(ppu_lower))
    if tight:
        plt.savefig(obj_path, bbox_inches='tight')
    else:
        plt.savefig(obj_path)
    if show:
        plt.show()


def plot_posterior_distribution(proc_saver,
                                variable_id,
                                obj_path="posterior.pdf",
                                figsize=(8, 8),
                                dpi=400,
                                subon=True,
                                subloc=0,
                                gap=0.15,
                                subgap=0.22,
                                subsizex=0.22,
                                subsizey=0.22,
                                framewidth=1.2,
                                bins=20,
                                subframewidth=1.0,
                                title=" ",
                                lbl_fontsize=18,
                                tick_fontsize=16,
                                tick_dir="out",
                                x_label="Variable",
                                y_label="Normalized Weight",
                                grid=True,
                                subgrid=False,
                                gridwidth=1.0,
                                subgridwidth=1.0,
                                color="darkcyan",
                                subcolor="tomato",
                                slim=True,
                                reverse=False,
                                subslim=True,
                                tight=True,
                                show=True):
    """
    This function is for the users to plot the posterior distribution of a variable. Both the probability of a sample
    and the cumulative probability can be plotted by this function.

    :argument
    proc_saver: pycup.save.ProcResultSaver object (uncertainty analysis result)
    variable_id: the index of the variable (x) -> int (column index, the sequence is same as what in lb and ub)
    obj_path: target saving path -> str, default = "posterior.pdf", acceptable format = ["jpg", "png", "pdf", ...]
    figsize: the x, y size of the figure -> tuple, default = (8, 8)
    dpi: the resolution of the figure -> int, default = 400
    subon: the switch for the subplot -> bool, default = True
    subloc: the location of the subplot -> int (0-8), default = 0
    gap: the width of the white frame -> float, default = 0.15
    subgap: the gap between the subplot and the main plot's margin -> float, default = 0.22
    subsizex: the width of the subplot -> float, default = 0.22
    subsizey: the height of the subplot -> float, default = 0.22
    framewidth: width of the frame line -> float, default = 1.2
    bins: the number of bins in the histogram -> float, default = 20
    subframewidth: width of the frame line of the subplot -> float, default = 1.0
    title: the title of the figure -> str
    lbl_fontsize: fontsize of axis labels -> int or float, default = 16
    tick_fontsize: fontsize of tick labels -> int or float, default = 14
    tick_dir: tick direction -> str ("in" or "out"), default = "out"
    x_label: the x axis label -> str, default ="Variable"
    y_label: the y axis label -> str, default ="Normalized Weight"
    grid: the switch for the grid in the main plot -> bool, default = True
    subgrid: the switch for the grid in the subplot -> bool, default = False
    gridwidth: the width of the grid lines in the main plot -> float, default = 1.0
    subgridwidth: the width of the grid lines in the subplot -> float, default = 1.0
    color: the color of the elements in the main plot -> "str", default = "darkcyan"
    subcolor: the color of the elements in the subplot -> "str", default = "tomato"
    slim: the plotting range option -> bool, default = True, if True, the plotting range will be confined to the
          value ranges of the given data.
    reverse: switch for plotting the main plot and subplot in reverse order
    subslim: the plotting range option for the subplot -> bool, default = True
    tight: option for the white frame outside the plotting area -> bool, default = True, tight output (slim white frame)
    show: option for showing in the IDE -> bool, default = True

    Usage:
    from pycup import save, plot

    opt_res = save.ProcResultSaver.load(r"ProcResult.rst")
    plot.plot_posterior_distribution(opt_res,variable_id=1,obj_path="dis.jpg")
    """
    if not isinstance(proc_saver, save.ProcResultSaver):
        raise TypeError("The input saver object should be pycup.save.ProcResultSaver.")
    sorted_sample_val = proc_saver.posterior_results.sorted_sample_val
    cum_sample = proc_saver.posterior_results.cum_sample
    fig = plt.figure(figsize=figsize, dpi=dpi)
    left, bottom, width, height = gap, gap, 1 - 2 * gap, 1 - 2 * gap
    ax1 = fig.add_axes([left, bottom, width, height])

    ax1.set_xlabel(x_label, fontsize=lbl_fontsize)
    ax1.set_ylabel(y_label, fontsize=lbl_fontsize)
    ax1.spines['bottom'].set_linewidth(framewidth)
    ax1.spines['top'].set_linewidth(framewidth)
    ax1.spines['right'].set_linewidth(framewidth)
    ax1.spines['left'].set_linewidth(framewidth)
    ax1.tick_params(
        labelsize=tick_fontsize,
        direction=tick_dir
    )
    if grid:
        ax1.grid(linestyle="--", color="silver", linewidth=gridwidth)
    if subloc == 0:
        left, bottom, width, height = subgap, 1 - subgap - subsizey, subsizex, subsizey
    elif subloc == 1:
        left, bottom, width, height = 0.5 - subsizex / 2, 1 - subgap - subsizey, subsizex, subsizey
    elif subloc == 2:
        left, bottom, width, height = 1 - subsizex - subgap, 1 - subgap - subsizey, subsizex, subsizey
    elif subloc == 3:
        left, bottom, width, height = subgap, 0.5 - 0.5 * subsizey, subsizex, subsizey
    elif subloc == 4:
        left, bottom, width, height = 0.5 - subsizex / 2, 0.5 - 0.5 * subsizey, subsizex, subsizey
    elif subloc == 5:
        left, bottom, width, height = 1 - subsizex - subgap, 0.5 - 0.5 * subsizey, subsizex, subsizey
    elif subloc == 6:
        left, bottom, width, height = subgap, subgap, subsizex, subsizey
    elif subloc == 7:
        left, bottom, width, height = 0.5 - subsizex / 2, subgap, subsizex, subsizey
    elif subloc == 8:
        left, bottom, width, height = 1 - subsizex - subgap, subgap, subsizex, subsizey
    barx = proc_saver.behaviour_results.behaviour_samples[:, variable_id].flatten()
    bary = proc_saver.behaviour_results.normalized_weight.flatten()

    ax2 = fig.add_axes([left, bottom, width, height])

    ax1.spines['bottom'].set_linewidth(subframewidth)
    ax1.spines['top'].set_linewidth(subframewidth)
    ax1.spines['right'].set_linewidth(subframewidth)
    ax1.spines['left'].set_linewidth(subframewidth)
    ax1.ticklabel_format(style='sci', scilimits=(-3, 4), axis='y', useMathText=True)
    ax2.ticklabel_format(style='sci', scilimits=(-3, 4), axis='y', useMathText=True)

    ax2.tick_params(
        labelsize=tick_fontsize,
        direction=tick_dir
    )
    if subgrid:
        ax2.grid(linestyle="--", color="silver", linewidth=subgridwidth)
    # ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    if reverse:
        ax2.plot(sorted_sample_val[:, variable_id], cum_sample[:, variable_id], color=subcolor)
        ax1.hist(barx, weights=bary, bins=bins, color=color)
    else:
        ax1.plot(sorted_sample_val[:, variable_id], cum_sample[:, variable_id], color=color)
        ax2.hist(barx, weights=bary, bins=bins, color=subcolor)

    plt.title(title)

    xmin = np.min(sorted_sample_val[:, variable_id], axis=0)
    xmax = np.max(sorted_sample_val[:, variable_id], axis=0)

    if slim:
        ax1.set_xlim(xmin, xmax)

    if subslim:
        ax2.set_xlim(xmin, xmax)

    if not subon:
        fig.delaxes(ax2)
    if tight:
        plt.savefig(obj_path, bbox_inches='tight')
    else:
        plt.savefig(obj_path)
    if show:
        plt.show()


def plot_posterior_distributions(l_proc_savers,
                                 variable_id,
                                 obj_path="posterior.pdf",
                                 figsize=(8, 8),
                                 dpi=400,
                                 gap=0.15,
                                 framewidth=1.2,
                                 linewidth=1,
                                 title=" ",
                                 lbl_fontsize=18,
                                 tick_fontsize=16,
                                 tick_dir="out",
                                 x_label="Variable",
                                 y_label="Normalized Weight",
                                 legend_labels=None,
                                 legend_on=True,
                                 legendloc=0,
                                 frameon=True,
                                 legend_fontsize=14,
                                 grid=True,
                                 gridwidth=1.0,
                                 slim=True,
                                 tight=True,
                                 show=True):
    """
    This function is for the users to plot the posterior distributions of a variable estimated by different algorithms.

    :argument
    l_proc_savers: a list of pycup.save.ProcResultSaver object (uncertainty analysis result)
    variable_id: the index of the variable (x) -> int (column index, the sequence is same as what in lb and ub)
    obj_path: target saving path -> str, default = "posterior.pdf", acceptable format = ["jpg", "png", "pdf", ...]
    figsize: the x, y size of the figure -> tuple, default = (8, 8)
    dpi: the resolution of the figure -> int, default = 400
    gap: the width of the white frame -> float, default = 0.15
    framewidth: width of the frame line -> float, default = 1.2
    title: the title of the figure -> str
    lbl_fontsize: fontsize of axis labels -> int or float, default = 16
    tick_fontsize: fontsize of tick labels -> int or float, default = 14
    tick_dir: tick direction -> str ("in" or "out"), default = "out"
    x_label: the x axis label -> str, default ="Variable"
    y_label: the y axis label -> str, default ="Normalized Weight"
    legend_labels: a list of labels of the legend, typically the algorithm names -> e.g. ["PSO","SSA","GLUE"]
    legendloc: the location of the legend -> int (0-8), default = 0
    legend_fontsize: the fontsize of phrases in the legend -> int or float, default = 14
    grid: the switch for the grid in the main plot -> bool, default = True
    subgrid: the switch for the grid in the subplot -> bool, default = False
    gridwidth: the width of the grid lines in the main plot -> float, default = 1.0
    slim: the plotting range option -> bool, default = True, if True, the plotting range will be confined to the
          value ranges of the given data.
    reverse: switch for plotting the main plot and subplot in reverse order
    tight: option for the white frame outside the plotting area -> bool, default = True, tight output (slim white frame)
    show: option for showing in the IDE -> bool, default = True

    Usage:
    from pycup import save, plot

    opt_res1 = save.ProcResultSaver.load(r"ProcResult1.rst")
    opt_res2 = save.ProcResultSaver.load(r"ProcResult2.rst")
    opt_res3 = save.ProcResultSaver.load(r"ProcResult3.rst")
    l_res = [opt_res1,opt_res1,opt_res1]
    plot.plot_posterior_distributions(l_res,variable_id=1,obj_path="dis.jpg")
    """
    if not isinstance(l_proc_savers, list) and not isinstance(l_proc_savers, tuple):
        raise TypeError("The argument l_proc_savers should be a list or a tuple.")
    for proc_saver in l_proc_savers:
        if not isinstance(proc_saver, save.ProcResultSaver):
            raise TypeError("The input saver object should be pycup.save.ProcResultSaver.")
    cs = ['r', 'g', 'b', 'y', "brown", "gray", "magenta", "cyan", "orange", "purple", "k", "pink"]
    fig = plt.figure(figsize=figsize, dpi=dpi)
    left, bottom, width, height = gap, gap, 1 - 2 * gap, 1 - 2 * gap
    ax1 = fig.add_axes([left, bottom, width, height])
    ax1.set_xlabel(x_label, fontsize=lbl_fontsize)
    ax1.set_ylabel(y_label, fontsize=lbl_fontsize)
    ax1.spines['bottom'].set_linewidth(framewidth)
    ax1.spines['top'].set_linewidth(framewidth)
    ax1.spines['right'].set_linewidth(framewidth)
    ax1.spines['left'].set_linewidth(framewidth)
    ax1.ticklabel_format(style='sci', scilimits=(-3, 4), axis='y', useMathText=True)
    ax1.tick_params(
        labelsize=tick_fontsize,
        direction=tick_dir
    )
    if grid:
        ax1.grid(linestyle="--", color="silver", linewidth=gridwidth)
    xmins = []
    xmaxs = []
    for i in range(len(l_proc_savers)):
        sorted_sample_val = l_proc_savers[i].posterior_results.sorted_sample_val
        xmin = np.min(sorted_sample_val[:, variable_id], axis=0)
        xmax = np.max(sorted_sample_val[:, variable_id], axis=0)
        xmins.append(xmin)
        xmaxs.append(xmax)
    xmin = np.min(xmins)
    xmax = np.max(xmaxs)
    for i in range(len(l_proc_savers)):
        sorted_sample_val = l_proc_savers[i].posterior_results.sorted_sample_val
        cum_sample = l_proc_savers[i].posterior_results.cum_sample
        x = sorted_sample_val[:, variable_id]
        y = cum_sample[:, variable_id]
        x = np.insert(x, 0, xmin)
        x = np.append(x, xmax)
        y = np.insert(y, 0, 0.0)
        y = np.append(y, 1.0)
        if legend_labels:
            lbl = legend_labels[i]
        else:
            lbl = "Algorithm{}".format(i + 1)
        ax1.plot(x, y, color=cs[i], label=lbl, linewidth=linewidth)

    plt.title(title)
    if legend_on:
        plt.legend(loc=legendloc, frameon=frameon, prop={'size': legend_fontsize})
    if slim:
        ax1.set_xlim(xmin, xmax)

        ax1.set_ylim(0, 1)

    if tight:
        plt.savefig(obj_path, bbox_inches='tight')
    else:
        plt.savefig(obj_path)
    if show:
        plt.show()


def plot_3d_posterior_hist(proc_results,
                           variable_id,
                           obj_path="posterior_hist3d.pdf",
                           figsize=(8, 8),
                           dpi=400,
                           bins=10,
                           alpha=0.7,
                           bar_width=0.6,
                           title=" ",
                           lbl_fontsize=16,
                           x_label="Value",
                           y_label="Algorithm",
                           z_label="Normalized Weight",
                           y_ticklabels=None,
                           tick_fontsize=14,
                           tick_dir="out",
                           view_init=(20, 30),
                           colors=None,
                           tight=True,
                           show=True
                           ):
    """
    This function is for the users to plot the posterior distributions of a variable obtained from different algorithms.
    Caused by the characteristic of the matplotlib, the 3D plot may not seems correctly at some angles, the problem cannot
    be fixed currently.

    :argument
    proc_results: a list containing pycup.save.ProcResultSaver object (uncertainty analysis result)
    variable_id: the index of the variable (x) -> int (column index, the sequence is same as what in lb and ub)
    obj_path: target saving path -> str, default = "posterior_bar3d.pdf", acceptable format = ["jpg", "png", "pdf", ...]
    figsize: the x, y size of the figure -> tuple, default = (8, 8)
    dpi: the resolution of the figure -> int, default = 400
    bins: the number of bins of each distribution data set -> int, default = 10
    bar_width: the width of each bar -> float (0.0~1.0), default = 0.7
    title: the title of the figure -> str
    lbl_fontsize: fontsize of axis labels -> int or float, default = 16
    tick_fontsize: fontsize of tick labels -> int or float, default = 14
    tick_dir: tick direction -> str ("in" or "out"), default = "out"
    x_label: the x axis label -> str, default ="Value"
    y_label: the y axis label -> str, default ="Algorithm"
    z_label: the z axis label -> str, default ="Normalized Weight"
    y_ticklabels: a list of y_ticklabels (this would be the name of your algorithms) -> array like
    colors: a list/tuple/array of the colors of the histograms -> array like
    tight: option for the white frame outside the plotting area -> bool, default = True, tight output (slim white frame)
    show: option for showing in the IDE -> bool, default = True

    Usage:
    from pycup import save, plot

    res1 = save.ProcResultSaver.load(r"ProcResult1.rst")
    res2 = save.ProcResultSaver.load(r"ProcResult2.rst")
    r_list = [res1, res2]
    plot.plot_3d_posterior_hist(r_list,variable_id=0,obj_path="hist3d.jpg",y_ticklabels=["SSA","GWO"])
    """
    if not isinstance(proc_results, list) and not isinstance(proc_results, tuple):
        raise TypeError("The argument l_proc_savers should be a list or a tuple.")
    for proc_saver in proc_results:
        if not isinstance(proc_saver, save.ProcResultSaver):
            raise TypeError("The input saver object should be pycup.save.ProcResultSaver.")
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(projection="3d")
    if colors is not None:
        cs = colors
    else:
        cs = ['r', 'g', 'b', 'y', "brown", "gray", "magenta", "cyan", "orange", "purple"]
    zs = np.arange(10)
    num_res = len(proc_results)
    idx = np.arange(len(proc_results))
    all_params = np.concatenate(
        [proc_results[id].behaviour_results.behaviour_samples[:, variable_id] for id in range(len(proc_results))])
    histrange = (np.min(all_params), np.max(all_params))
    for c, z, id in zip(cs[0:num_res], zs[0:num_res], idx):
        ys = proc_results[id].behaviour_results.normalized_weight.flatten()
        xs = proc_results[id].behaviour_results.behaviour_samples[:, variable_id].flatten()
        bar_c = c
        hist, xedges = np.histogram(xs, weights=ys, bins=bins, range=histrange)
        xpos = xedges[0:-1]
        ypos = np.ones(len(xpos)) * id
        zpos = np.zeros(len(xpos))
        dx = (xedges[1] - xedges[0]) * np.ones(len(zpos))
        dy = bar_width * np.ones(len(zpos))
        dz = hist
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=bar_c, alpha=alpha, edgecolor="black")

    ax.set_xlabel(x_label, fontsize=lbl_fontsize)
    ax.set_ylabel(y_label, fontsize=lbl_fontsize)
    ax.set_zlabel(z_label, fontsize=lbl_fontsize)
    ax.ticklabel_format(style='sci', scilimits=(-3, 4), axis='x', useMathText=True)
    ax.ticklabel_format(style='sci', scilimits=(-3, 4), axis='z', useMathText=True)
    ax.set_ylim(0, len(proc_results))
    ax.set_yticks(np.arange(len(proc_results)))
    if y_ticklabels:
        ax.set_yticklabels(y_ticklabels)
    else:
        ax.set_yticklabels(["Algorithm{}".format(i + 1) for i in range(len(proc_results))])
    ax.view_init(*view_init)
    ax.tick_params(
        labelsize=tick_fontsize,
        direction=tick_dir
    )
    plt.title(title)
    if tight:
        plt.savefig(obj_path, bbox_inches='tight')
    else:
        plt.savefig(obj_path)
    if show:
        plt.show()


def plot_3d_posterior_distributions(proc_results,
                                    variable_id,
                                    obj_path="posterior3d.pdf",
                                    figsize=(8, 8),
                                    dpi=400,
                                    alpha=0.3,
                                    title=" ",
                                    lbl_fontsize=16,
                                    x_label="Value",
                                    y_label="Algorithm",
                                    z_label="Normalized Weight",
                                    y_ticklabels=None,
                                    tick_fontsize=14,
                                    tick_dir="out",
                                    view_init=(20, 30),
                                    colors=None,
                                    slim=False,
                                    tight=True,
                                    show=True
                                    ):
    """
    This function is for the users to plot the posterior distributions of a variable obtained from different algorithms.

    :argument
    proc_results: a list containing pycup.save.ProcResultSaver object (uncertainty analysis result)
    variable_id: the index of the variable (x) -> int (column index, the sequence is same as what in lb and ub)
    obj_path: target saving path -> str, default = "posterior3d.pdf", acceptable format = ["jpg", "png", "pdf", ...]
    figsize: the x, y size of the figure -> tuple, default = (8, 8)
    dpi: the resolution of the figure -> int, default = 400
    title: the title of the figure -> str
    lbl_fontsize: fontsize of axis labels -> int or float, default = 16
    tick_fontsize: fontsize of tick labels -> int or float, default = 14
    tick_dir: tick direction -> str ("in" or "out"), default = "out"
    x_label: the x axis label -> str, default ="Value"
    y_label: the y axis label -> str, default ="Algorithm"
    z_label: the z axis label -> str, default ="Normalized Weight"
    y_ticklabels: a list of y_ticklabels (this would be the name of your algorithms) -> array like
    view_init: a tuple of the view angle -> tuple, default = (20,30)
    tight: option for the white frame outside the plotting area -> bool, default = True, tight output (slim white frame)
    show: option for showing in the IDE -> bool, default = True

    Usage:
    from pycup import save, plot

    res1 = save.ProcResultSaver.load(r"ProcResult1.rst")
    res2 = save.ProcResultSaver.load(r"ProcResult2.rst")
    r_list = [res1, res2]
    plot.plot_3d_posterior_distributions(r_list,variable_id=0,obj_path="dis3d.jpg",y_ticklabels=["SSA","GWO"])
    """
    if not isinstance(proc_results, list) and not isinstance(proc_results, tuple):
        raise TypeError("The argument l_proc_savers should be a list or a tuple.")
    for proc_saver in proc_results:
        if not isinstance(proc_saver, save.ProcResultSaver):
            raise TypeError("The input saver object should be pycup.save.ProcResultSaver.")
    if colors is not None:
        cs = colors
    else:
        cs = ['r', 'g', 'b', 'y', "brown", "gray", "magenta", "cyan", "orange", "purple"]
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(projection="3d")
    xs = []
    zs = []
    ys = []
    min_xs = []
    max_xs = []
    for i in range(len(proc_results)):
        z = proc_results[i].posterior_results.cum_sample[:, variable_id].flatten()
        x = proc_results[i].posterior_results.sorted_sample_val[:, variable_id].flatten()
        y = np.ones(len(z)) * i
        min_xs.append(np.min(x))
        max_xs.append(np.max(x))
        xs.append(x)
        zs.append(z)
        ys.append(y)
    min_x = np.min(min_xs)
    max_x = np.max(max_xs)
    for i in range(len(proc_results)):
        xs[i] = np.insert(xs[i], 0, min_x)
        ys[i] = np.insert(ys[i], 0, i)
        zs[i] = np.insert(zs[i], 0, 0.0)

        xs[i] = np.append(xs[i], max_x)
        ys[i] = np.append(ys[i], i)
        zs[i] = np.append(zs[i], 1.0)
    # max_x = np.max(xs)

    for i in range(len(proc_results)):
        verts = [(xs[i][j], ys[i][j], zs[i][j]) for j in range(len(xs[i]))] + [(xs[i].max(), i, 0), (xs[i].min(), i, 0)]
        ax.add_collection3d(Poly3DCollection([verts], color=cs[i], alpha=alpha, linewidths=0))
        ax.plot(xs[i], ys[i], zs[i], color=cs[i])
    ax.set_xlabel(x_label, fontsize=lbl_fontsize)
    ax.set_ylabel(y_label, fontsize=lbl_fontsize)
    ax.set_zlabel(z_label, fontsize=lbl_fontsize)
    ax.ticklabel_format(style='sci', scilimits=(-3, 4), axis='x', useMathText=True)
    ax.ticklabel_format(style='sci', scilimits=(-3, 4), axis='z', useMathText=True)
    if slim == True:
        ax.set_ylim(0, len(proc_results) - 1)
    else:
        ax.set_ylim(0, len(proc_results))
    ax.set_xlim(min_x, max_x)
    ax.set_yticks(np.arange(len(proc_results)))
    if y_ticklabels:
        ax.set_yticklabels(y_ticklabels)
    else:
        ax.set_yticklabels(["Algorithm{}".format(i + 1) for i in range(len(proc_results))])
    ax.view_init(*view_init)
    ax.tick_params(
        labelsize=tick_fontsize,
        direction=tick_dir
    )
    plt.title(title)
    if tight:
        plt.savefig(obj_path, bbox_inches='tight')
    else:
        plt.savefig(obj_path)
    if show:
        plt.show()


def plot_opt_curves(
        raw_saver,
        obj_path="opt_curves.pdf",
        figsize=(8, 8),
        dpi=400,
        framewidth=1.2,
        title=" ",
        lbl_fontsize=16,
        tick_fontsize=14,
        tick_dir="out",
        x_label="Iterations",
        y_label="Fitness",
        linewidth=2.0,
        linestyle=None,
        colors=None,
        gridwidth=1.0,
        legendon=True,
        legendloc=0,
        legendlabels=None,
        legend_fontsize=14,
        frameon=True,
        grid=True,
        slim=True,
        tight=True,
        show=True
):
    """
    This function is for the users to plot the optimization curve of an optimization process.

    :argument
    raw_saver: caliboy.save.RawDataSaver objective
    obj_path: target saving path -> str, default = "opt_curves.pdf", acceptable format = ["jpg", "png", "pdf", ...]
    figsize: the x, y size of the figure -> tuple, default = (8, 8)
    dpi: the resolution of the figure -> int, default = 400
    framewidth: width of the frame line -> float, default = 1.2
    title: the title of the figure -> str
    lbl_fontsize: fontsize of axis labels -> int or float, default = 16
    tick_fontsize: fontsize of tick labels -> int or float, default = 14
    tick_dir: tick direction -> str ("in" or "out"), default = "out"
    x_label: the x axis label -> str, default ="Iterations"
    y_label: the y axis label -> str, default ="Fitness"
    linewidth: line width of the optimization curves -> float, default = 2.0
    linestyle: line style of the optimization curves -> str, default = None (use the matplotlib default setting)
    colors: colors of the optimization curves -> list, default = None
    gridwidth: the width of the grid lines in the main plot -> float, default = 1.0
    legendon: switch for the legend -> bool, default = True
    legendloc: the location of the legend -> int (0-8), default = 0
    legendlabels: labels in the legend -> list (e.g. ["SSA","WOA"]
    legend_fontsize: fontsize of legend labels -> int or float, default = 14
    frameon: the switch for opening the legend frame -> bool, default = True
    grid: the switch for the grid in the main plot -> bool, default = True
    slim: the plotting range option -> bool, default = True, if True, the plotting range will be confined to the
          value ranges of the given data.
    tight: option for the white frame outside the plotting area -> bool, default = True, tight output (slim white frame)
    show: option for showing in the IDE -> bool, default = True

    Usage:
    from pycup import save, plot

    raw_res1 = save.RawDataSaver.load(r"RawResult1.rst")
    raw_res2 = save.RawDataSaver.load(r"RawResult2.rst")
    raw_res3 = save.RawDataSaver.load(r"RawResult3.rst")
    raw_res = [raw_res1,raw_res2,raw_res3]

    plot.plot_opt_curves(raw_res,slim=False,frameon=False,legendlabels=["SSA","GWO","WOA"],
                         linestyle=["--","-","-."],colors=["r","g","b"],obj_path="opt_curves.jpg")
    """
    curves = []
    if isinstance(raw_saver, tuple) or isinstance(raw_saver, list):
        for s in raw_saver:
            if s.opt_type == "GLUE" or s.opt_type == "MO-SWARM":
                raise AttributeError("GLUE or Multi-objective algorithms have not opt-curve.")
            curve = s.Curve
            curves.append(curve)
    else:
        if raw_saver.opt_type == "GLUE" or raw_saver.opt_type == "MO-SWARM":
            raise AttributeError("GLUE or Multi-objective algorithms have not opt-curve.")
        curve = raw_saver.Curve
        curves.append(curve)

    if legendlabels is None:
        legendlabels = ["Algorithm{}".format(i) for i in range(len(curves))]
    if linestyle is None:
        linestyle = [None for i in range(len(curves))]
    if colors is None:
        colors = [None for i in range(len(curves))]
    fig = plt.figure(figsize=figsize, dpi=dpi)
    for c in range(len(curves)):
        plt.plot(curves[c], linewidth=linewidth, label=legendlabels[c], linestyle=linestyle[c], c=colors[c])

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(framewidth)
    ax.spines['top'].set_linewidth(framewidth)
    ax.spines['right'].set_linewidth(framewidth)
    ax.spines['left'].set_linewidth(framewidth)
    ax.tick_params(
        labelsize=tick_fontsize,
        direction=tick_dir  #
    )
    ax.ticklabel_format(style='sci', scilimits=(-3, 4), axis='y', useMathText=True)
    plt.xlabel(x_label, fontsize=lbl_fontsize)
    plt.ylabel(y_label, fontsize=lbl_fontsize)
    plt.title(title)
    if grid:
        plt.grid(linestyle="--", color="silver", linewidth=gridwidth)
    if legendon:
        plt.legend(loc=legendloc, frameon=frameon, prop={'size': legend_fontsize})
    if slim:
        plt.ylim(np.min(curves), np.max(curves))
    if tight:
        plt.savefig(obj_path, bbox_inches='tight')
    else:
        plt.savefig(obj_path)
    if show:
        plt.show()


def get_time_labels(start, steps, showY=True, showdate=True, showH=True, showM=True, showS=True):
    """
    This function is for the users to generate the time labels for the uncertainty band plotting.

    :argument
    start: the start time label -> str, format = '%Y-%m-%d %H:%M:%S'
    steps: num. of steps that want to generate -> int (len(sequence))
    showY: switch for including the Year -> bool, default = True
    showdate: switch for including the date -> bool, default = True
    showH: switch for including the hour -> bool, default = True
    showM: switch for including the minute -> bool, default = True
    showS: switch for including the second -> bool, default = True

    :return:
    list_days: time labels array, array like
    """
    list_days = []
    start = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    fmt_date = []
    fmt_time = []
    if showY:
        fmt_date.append("%Y")
    if showdate:
        fmt_date.extend(['%m', '%d'])
    if showH:
        fmt_time.append('%H')
    if showM:
        fmt_time.append('%M')
    if showS:
        fmt_time.append('%S')
    date_str = "-".join(fmt_date)
    time_str = ":".join(fmt_time)
    fmt_str = date_str + " " + time_str
    list_days.append(datetime.datetime.strftime(start, fmt_str))
    for i in range(steps - 1):
        start += datetime.timedelta(minutes=5)
        list_days.append(datetime.datetime.strftime(start, fmt_str))
    return list_days


def plot_2d_pareto_front(
        raw_saver,
        objfunid1,
        objfunid2,
        obj_path="pareto2d.pdf",
        figsize=(8, 8),
        dpi=400,
        markersize=30,
        framewidth=1.2,
        title=" ",
        lbl_fontsize=16,
        tick_fontsize=14,
        tick_dir="out",
        x_label="Fitness 1",
        y_label="Fitness 2",
        color="red",
        gridwidth=1.0,
        frameon=True,
        grid=True,
        legendon=True,
        legendloc=0,
        legendlabel="Pareto non-dominated solutions",
        legend_fontsize=14,
        topsis_optimum=False,
        best_color="b",
        best_markersize=80,
        slim=True,
        tight=True,
        show=True):
    """
    This function is for the users to plot the 2D pareto front for multi-objective algorithms (MOPSO in the current version)
    Two objective functions can be considered in this function.

    :argument
    raw_saver: caliboy.save.RawDataSaver objective
    objfunid1: index of the objective function 1 (column id for extracting data from the fitness array) -> int
    objfunid2: index of the objective function 2 (column id for extracting data from the fitness array) -> int
    obj_path: target saving path -> str, default = "pareto2d.pdf", acceptable format = ["jpg", "png", "pdf", ...]
    figsize: the x, y size of the figure -> tuple, default = (8, 8)
    dpi: the resolution of the figure -> int, default = 400
    framewidth: width of the frame line -> float, default = 1.2
    title: the title of the figure -> str
    lbl_fontsize: fontsize of axis labels -> int or float, default = 16
    tick_fontsize: fontsize of tick labels -> int or float, default = 14
    tick_dir: tick direction -> str ("in" or "out"), default = "out"
    x_label: the x axis label -> str, default ="Fitness 1"
    y_label: the y axis label -> str, default ="Fitness 2"
    color: color of the pareto solution points -> str, default = "red"
    gridwidth: the width of the grid lines in the main plot -> float, default = 1.0
    frameon: the switch for opening the legend frame -> bool, default = True
    grid: the switch for the grid in the main plot -> bool, default = True
    legendon: switch for the legend -> bool, default = True
    legendloc: the location of the legend -> int (0-8), default = 0
    legendlabels: label in the legend -> str, default = "Pareto non-dominated solutions"
    legend_fontsize: fontsize of legend labels -> int or float, default = 14
    topsis_optimum: This can only be True when the raw saver has been processed by pycup.TOPSIS.TopsisAnalyzer. -> bool
    slim: the plotting range option -> bool, default = True, if True, the plotting range will be confined to the
          value ranges of the given data.
    tight: option for the white frame outside the plotting area -> bool, default = True, tight output (slim white frame)
    show: option for showing in the IDE -> bool, default = True

    Usage:
    import pycup as cp
    import numpy as np

    def mo_fun1(X):
        y1 = 1 - np.exp(-np.sum((X-1/np.sqrt(3))**2))
        y2 = 1 - np.exp(-np.sum((X+1/np.sqrt(3))**2))
        #y3 = 1 - np.exp(np.sum((X**2 + np.sqrt(2))))
        #fitness = np.array([y1,y2,y3]).reshape(1,-1)
        fitness = np.array([y1,y2]).reshape(1,-1)
        result = fitness
        return fitness,result

    lb = np.array([-2, -2, -2])
    ub = np.array([2, 2, 2])
    cp.MOPSO.run(pop=100, dim=3, lb=lb, ub = ub, MaxIter=20,n_obj=2,nAr=300,M=50,
                       fun=mo_fun1)

    from pycup import plot,save

    path = "RawResult.rst"
    saver = save.RawDataSaver.load(path)

    # Draw the Pareto front
    plot.plot_2d_pareto_front(saver,objfunid1=0,objfunid2=1,obj_path="pareto2d.pdf")
    """
    if not isinstance(raw_saver, save.RawDataSaver):
        raise TypeError("The input saver object should be pycup.save.RawDataSaver.")
    x = raw_saver.pareto_fitness[:, objfunid1]
    y = raw_saver.pareto_fitness[:, objfunid2]
    fig = plt.figure(figsize=figsize, dpi=dpi)

    ax = fig.add_subplot(111)
    ax.scatter(x, y, color=color, label=legendlabel, s=markersize)
    if topsis_optimum:
        if topsis_optimum:
            if hasattr(raw_saver, "TOPSISidx"):
                topsis_fitness = raw_saver.GbestScore
                ax.scatter(topsis_fitness[objfunid1], topsis_fitness[objfunid2], color=best_color, s=best_markersize,
                           label="TOPSIS optimum")
            else:
                raise AttributeError(
                    "The TOPSIS optimum can only be plotted using a pycup.TOPSIS.TopsisAnalyzer processed RawDataSaver.")
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(framewidth)
    ax.spines['top'].set_linewidth(framewidth)
    ax.spines['right'].set_linewidth(framewidth)
    ax.spines['left'].set_linewidth(framewidth)
    ax.tick_params(
        labelsize=tick_fontsize,
        direction=tick_dir
    )
    ax.ticklabel_format(style='sci', scilimits=(-3, 4), axis='y', useMathText=True)
    plt.xlabel(x_label, fontsize=lbl_fontsize)
    plt.ylabel(y_label, fontsize=lbl_fontsize)
    plt.title(title)

    if grid:
        plt.grid(linestyle="--", color="silver", linewidth=gridwidth)
    if legendon:
        plt.legend(loc=legendloc, frameon=frameon, prop={'size': legend_fontsize})
    if slim:
        plt.xlim(np.min(x), np.max(x))
        plt.ylim(np.min(y), np.max(y))
    if tight:
        plt.savefig(obj_path, bbox_inches='tight')
    else:
        plt.savefig(obj_path)
    if show:
        plt.show()


def plot_3d_pareto_front(raw_saver,
                         objfunid1,
                         objfunid2,
                         objfunid3,
                         obj_path="pareto3d.pdf",
                         figsize=(8, 8),
                         dpi=400,
                         title=" ",
                         lbl_fontsize=16,
                         marker=".",
                         markersize=50,
                         x_label="Fitness1",
                         y_label="Fitness2",
                         z_label="Fitness3",
                         tick_fontsize=14,
                         tick_dir="out",
                         color="r",
                         view_init=(),
                         x_lim=(),
                         y_lim=(),
                         z_lim=(),
                         topsis_optimum=False,
                         best_color="b",
                         best_markersize=120,
                         slim=True,
                         tight=True,
                         show=True):
    """
    This function is for the users to plot the 3D pareto front for multi-objective algorithms (MOPSO in the current version)
    Three objective functions can be considered in this function.

    :argument
    raw_saver: caliboy.save.RawDataSaver objective
    objfunid1: index of the objective function 1 (column id for extracting data from the fitness array) -> int
    objfunid2: index of the objective function 2 (column id for extracting data from the fitness array) -> int
    objfunid3: index of the objective function 2 (column id for extracting data from the fitness array) -> int
    obj_path: target saving path -> str, default = "pareto3d.pdf", acceptable format = ["jpg", "png", "pdf", ...]
    figsize: the x, y size of the figure -> tuple, default = (8, 8)
    dpi: the resolution of the figure -> int, default = 400
    title: the title of the figure -> str
    lbl_fontsize: fontsize of axis labels -> int or float, default = 16
    marker: the marker style of the observation points -> str, default = "."
    markersize: the size of markers -> int or float, default = 50
    x_label: the x axis label -> str, default ="Fitness 1"
    y_label: the y axis label -> str, default ="Fitness 2"
    z_label: the z axis label -> str, default ="Fitness 3"
    tick_fontsize: fontsize of tick labels -> int or float, default = 14
    tick_dir: tick direction -> str ("in" or "out"), default = "out"
    color: color of the pareto solution points -> str, default = "r"
    view_init: view angle -> tuple, (float, float)
    x_lim: x axis plotting range -> tuple
    y_lim: y axis plotting range -> tuple
    z_lim: z axis plotting range -> tuple
    topsis_optimum: This can only be True when the raw saver has been processed by pycup.TOPSIS.TopsisAnalyzer. -> bool
    slim: the plotting range option -> bool, default = True, if True, the plotting range will be confined to the
          value ranges of the given data.
    tight: option for the white frame outside the plotting area -> bool, default = True, tight output (slim white frame)
    show: option for showing in the IDE -> bool, default = True

    Usage:
    import pycup as cp
    import numpy as np
    from pycup import plot,save

    def mo_fun1(X):
        y1 = 1 - np.exp(-np.sum((X-1/np.sqrt(3))**2))
        y2 = 1 - np.exp(-np.sum((X+1/np.sqrt(3))**2))
        y3 = 1 - np.exp(np.sum((X**2 + np.sqrt(2))))
        fitness = np.array([y1,y2,y3]).reshape(1,-1)
        result = fitness
        return fitness,result

    lb = np.array([-2, -2, -2])
    ub = np.array([2, 2, 2])
    cp.MOPSO.run(pop=100, dim=3, lb=lb, ub = ub, MaxIter=20,n_obj=3,nAr=300,M=50,
                       fun=mo_fun1)

    path = "RawResult.rst"
    saver = save.RawDataSaver.load(path)
    plot.plot_3d_pareto_front(saver,objfunid1=0,objfunid2=1,objfunid3=2,obj_path="pareto3d.pdf")
    """
    if not isinstance(raw_saver, save.RawDataSaver):
        raise TypeError("The input saver object should be pycup.save.RawDataSaver.")
    x = raw_saver.pareto_fitness[:, objfunid1]
    y = raw_saver.pareto_fitness[:, objfunid2]
    z = raw_saver.pareto_fitness[:, objfunid3]
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    s = ax.scatter(xs=x, ys=y, zs=z, c=color,
                   marker=marker, s=markersize)
    if topsis_optimum:
        if hasattr(raw_saver, "TOPSISidx"):
            topsis_fitness = raw_saver.GbestScore
            ax.scatter(xs=topsis_fitness[objfunid1], ys=topsis_fitness[objfunid2], zs=topsis_fitness[objfunid3],
                       color=best_color, s=best_markersize, label="TOPSIS optimum")
        else:
            raise AttributeError(
                "The TOPSIS optimum can only be plotted using a pycup.TOPSIS.TopsisAnalyzer processed RawDataSaver.")
    ax.view_init(*view_init)
    if slim:
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(np.min(y), np.max(y))
        ax.set_zlim(np.min(z), np.max(z))
    if x_lim:
        ax.set_xlim(*x_lim)
    if y_lim:
        ax.set_ylim(*y_lim)
    if z_lim:
        ax.set_zlim(*z_lim)
    ax = plt.gca()

    ax.tick_params(
        labelsize=tick_fontsize,
        direction=tick_dir
    )

    ax.set_xlabel(x_label, fontsize=lbl_fontsize)
    ax.set_ylabel(y_label, fontsize=lbl_fontsize)
    ax.set_zlabel(z_label, fontsize=lbl_fontsize)
    ax.ticklabel_format(style='sci', scilimits=(-3, 4), axis='z', useMathText=True)
    plt.title(title)
    if tight:
        plt.savefig(obj_path, bbox_inches='tight')
    else:
        plt.savefig(obj_path)
    if show:
        plt.show()


def plot_2d_MO_fitness_space(raw_saver,
                             objfunid1,
                             objfunid2,
                             obj_path="MOfitness2d.pdf",
                             figsize=(8, 8),
                             dpi=400,
                             title=" ",
                             lbl_fontsize=16,
                             marker=".",
                             markersize=50,
                             x_label="Fitness1",
                             y_label="Fitness2",
                             tick_fontsize=14,
                             tick_dir="out",
                             framewidth=1.2,
                             cmap="plasma",
                             single_c="b",
                             cbar_ttl="Iterations",
                             cbar_ttl_size=16,
                             cbar_lbl_size=12,
                             cbar_frac=0.05,
                             cbar_pad=0.05,
                             x_lim=(),
                             y_lim=(),
                             slim=True,
                             tight=True,
                             show=True
                             ):
    """
    This function is for the users to plot the samples' 2D fitness history for multi-objective algorithms. Two objective
    functions can be considered in this function.

    :argument
    raw_saver: caliboy.save.RawDataSaver objective
    objfunid1: index of the objective function 1 (column id for extracting data from the fitness array) -> int
    objfunid2: index of the objective function 2 (column id for extracting data from the fitness array) -> int
    obj_path: target saving path -> str, default = "MOfitness2d.pdf", acceptable format = ["jpg", "png", "pdf", ...]
    figsize: the x, y size of the figure -> tuple, default = (8, 8)
    dpi: the resolution of the figure -> int, default = 400
    title: the title of the figure -> str
    lbl_fontsize: fontsize of axis labels -> int, default = 16
    marker: type of the markers in the scatter -> str, default = "." (dot like)
    markersize: the size of markers -> int or float, default = 50
    x_label: the x axis label -> str, default ="Fitness1"
    y_label: the y axis label -> str, default ="Fitness2"
    tick_fontsize: fontsize of tick labels -> int or float, default = 14
    tick_dir: tick direction -> str ("in" or "out"), default = "out"
    framewidth: width of the frame line -> float, default = 1.2
    cmap: color map -> str, default = "plasma"
    single_c: color -> str, default = "b", this will be used if the optimization method is GLUE (no iteration)
    cbar_ttl: the title of color bar -> str, default = "Iterations", users can use " " to replace it
    cbar_ttl_size: the size of color bar title -> int or float, default = 16
    cbar_lbl_size: the size of color bar tick labels -> int or float, default = 12
    cbar_frac: the size of color bar -> float, default = 0.05
    cbar_pad: the gap between cbar and the main figure -> float, default = 0.05
    x_lim: x axis plotting range -> tuple
    y_lim: y axis plotting range -> tuple
    slim: the plotting range option -> bool, default = True, if True, the plotting range will be confined to the
          value ranges of the given data.
    tight: option for the white frame outside the plotting area -> bool, default = True, tight output (slim white frame)
    show: option for showing in the IDE -> bool, default = True

    Usage:
    import pycup as cp
    import numpy as np
    from pycup import plot,save

    def mo_fun1(X):
        y1 = 1 - np.exp(-np.sum((X-1/np.sqrt(3))**2))
        y2 = 1 - np.exp(-np.sum((X+1/np.sqrt(3))**2))
        fitness = np.array([y1,y2]).reshape(1,-1)
        result = fitness
        return fitness,result

    lb = np.array([-2, -2, -2])
    ub = np.array([2, 2, 2])
    cp.MOPSO.run(pop=100, dim=3, lb=lb, ub = ub, MaxIter=20,n_obj=2,nAr=300,M=50,
                       fun=mo_fun1)


    path = "RawResult.rst"
    saver = save.RawDataSaver.load(path)
    plot.plot_2d_MO_fitness_space(saver,objfunid1=0,objfunid2=1)

    Note:
    selectable cmaps includes [('Perceptually Uniform Sequential',
                            ['viridis', 'inferno', 'plasma', 'magma']),
         ('Sequential',     ['Blues', 'BuGn', 'BuPu',
                             'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
                             'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
                             'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
         ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool',
                             'copper', 'gist_heat', 'gray', 'hot',
                             'pink', 'spring', 'summer', 'winter']),
         ('Diverging',      ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
                             'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
                             'seismic']),
         ('Qualitative',    ['Accent', 'Dark2', 'Paired', 'Pastel1',
                             'Pastel2', 'Set1', 'Set2', 'Set3']),
         ('Miscellaneous',  ['gist_earth', 'terrain', 'ocean', 'gist_stern',
                             'brg', 'CMRmap', 'cubehelix',
                             'gnuplot', 'gnuplot2', 'gist_ncar',
                             'nipy_spectral', 'jet', 'rainbow',
                             'gist_rainbow', 'hsv', 'flag', 'prism'])]
    """
    if not isinstance(raw_saver, save.RawDataSaver):
        raise TypeError("The input saver object should be pycup.save.RawDataSaver.")
    fitness = np.array(raw_saver.historical_fitness)
    iterations, population = analyze_saver(raw_saver)
    if iterations > 1:
        a = np.ones((fitness.shape[0], fitness.shape[1]))
        c = np.linspace(0, iterations, iterations)
        for i in range(fitness.shape[0]):
            a[i] = a[i] * c[i]
        fig = plt.figure(figsize=figsize, dpi=dpi)
        s = plt.scatter(fitness[:, :, objfunid1], fitness[:, :, objfunid2], c=a,
                        marker=marker, s=markersize, cmap=cmap)

        bar = plt.colorbar(fraction=cbar_frac, pad=cbar_pad)
        bar.set_label(cbar_ttl, fontsize=cbar_ttl_size)
        bar.ax.tick_params(labelsize=cbar_lbl_size)
    else:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        s = plt.scatter(fitness[:, objfunid1], fitness[:, objfunid2], c=single_c,
                        marker=marker, s=markersize, cmap=cmap)
    if slim:
        if iterations > 1:
            fitness = np.concatenate(fitness)
        xmin = np.min(fitness[:, objfunid1], axis=0)
        ymin = np.min(fitness[:, objfunid2], axis=0)
        xmax = np.max(fitness[:, objfunid1], axis=0)
        ymax = np.max(fitness[:, objfunid2], axis=0)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
    if x_lim:
        plt.xlim(*x_lim)
    if y_lim:
        plt.ylim(*y_lim)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(framewidth)
    ax.spines['top'].set_linewidth(framewidth)
    ax.spines['right'].set_linewidth(framewidth)
    ax.spines['left'].set_linewidth(framewidth)
    ax.tick_params(
        labelsize=tick_fontsize,  # y轴字体大小设置
        direction=tick_dir  # y轴标签方向设置
    )
    ax.ticklabel_format(style='sci', scilimits=(-3, 4), axis='y', useMathText=True)
    plt.xlabel(x_label, fontsize=lbl_fontsize)
    plt.ylabel(y_label, fontsize=lbl_fontsize)
    plt.title(title)
    if tight:
        plt.savefig(obj_path, bbox_inches='tight')
    else:
        plt.savefig(obj_path)
    if show:
        plt.show()


def plot_3d_MO_fitness_space(raw_saver,
                             objfunid1,
                             objfunid2,
                             objfunid3,
                             obj_path="MOfitness3d.pdf",
                             figsize=(8, 8),
                             dpi=400,
                             title=" ",
                             lbl_fontsize=16,
                             marker=".",
                             markersize=50,
                             x_label="Fitness1",
                             y_label="Fitness2",
                             z_label="Fitness3",
                             tick_fontsize=14,
                             tick_dir="out",
                             cmap="plasma",
                             single_c="b",
                             cbar_ttl="Iterations",
                             cbar_ttl_size=16,
                             cbar_lbl_size=12,
                             cbar_frac=0.03,
                             cbar_pad=0.05,
                             view_init=(),
                             x_lim=(),
                             y_lim=(),
                             z_lim=(),
                             slim=True,
                             tight=True,
                             show=True
                             ):
    """
    This function is for the users to plot the samples' 3D fitness history for multi-objective algorithms. Three objective
    functions can be considered in this function.

    :argument
    raw_saver: caliboy.save.RawDataSaver objective
    objfunid1: index of the objective function 1 (column id for extracting data from the fitness array) -> int
    objfunid2: index of the objective function 2 (column id for extracting data from the fitness array) -> int
    objfunid3: index of the objective function 2 (column id for extracting data from the fitness array) -> int
    obj_path: target saving path -> str, default = "MOfitness3d.pdf", acceptable format = ["jpg", "png", "pdf", ...]
    figsize: the x, y size of the figure -> tuple, default = (8, 8)
    dpi: the resolution of the figure -> int, default = 400
    title: the title of the figure -> str
    lbl_fontsize: fontsize of axis labels -> int or float, default = 16
    marker: type of the markers in the scatter -> str, default = "." (dot like)
    markersize: the size of markers -> int or float, default = 50
    x_label: the x axis label -> str, default ="Fitness1"
    y_label: the y axis label -> str, default ="Fitness2"
    z_label: the z axis label -> str, default ="Fitness3"
    tick_fontsize: fontsize of tick labels -> int or float, default = 14
    tick_dir: tick direction -> str ("in" or "out"), default = "out"
    framewidth: width of the frame line -> float, default = 1.2
    cmap: color map -> str, default = "plasma"
    single_c: color -> str, default = "b", this will be used if the optimization method is GLUE (no iteration)
    cbar_ttl: the title of color bar -> str, default = "Iterations", users can use " " to replace it
    cbar_ttl_size: the size of color bar title -> int or float, default = 16
    cbar_lbl_size: the size of color bar tick labels -> int or float, default = 12
    cbar_frac: the size of color bar -> float, default = 0.05
    cbar_pad: the gap between cbar and the main figure -> float, default = 0.05
    view_init: view angle -> tuple, (float, float)
    x_lim: x axis plotting range -> tuple
    y_lim: y axis plotting range -> tuple
    z_lim: z axis plotting range -> tuple
    slim: the plotting range option -> bool, default = True, if True, the plotting range will be confined to the
          value ranges of the given data.
    tight: option for the white frame outside the plotting area -> bool, default = True, tight output (slim white frame)
    show: option for showing in the IDE -> bool, default = True

    Usage:
    import pycup as cp
    import numpy as np

    def mo_fun1(X):
        y1 = 1 - np.exp(-np.sum((X-1/np.sqrt(3))**2))
        y2 = 1 - np.exp(-np.sum((X+1/np.sqrt(3))**2))
        y3 = 1 - np.exp(np.sum((X**2 + np.sqrt(2))))
        fitness = np.array([y1,y2,y3]).reshape(1,-1)
        #fitness = np.array([y1,y2]).reshape(1,-1)
        result = fitness
        return fitness,result

    lb = np.array([-2, -2, -2])
    ub = np.array([2, 2, 2])
    cp.MOPSO.run(pop=100, dim=3, lb=lb, ub = ub, MaxIter=20,n_obj=3,nAr=300,M=50,
                       fun=mo_fun1)

    from pycup import plot,save

    path = "RawResult.rst"
    saver = save.RawDataSaver.load(path)
    plot.plot_3d_MO_fitness_space(saver,objfunid1=0,objfunid2=1,objfunid3=2)


    Note:
    selectable cmaps includes [('Perceptually Uniform Sequential',
                            ['viridis', 'inferno', 'plasma', 'magma']),
         ('Sequential',     ['Blues', 'BuGn', 'BuPu',
                             'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
                             'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
                             'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
         ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool',
                             'copper', 'gist_heat', 'gray', 'hot',
                             'pink', 'spring', 'summer', 'winter']),
         ('Diverging',      ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
                             'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
                             'seismic']),
         ('Qualitative',    ['Accent', 'Dark2', 'Paired', 'Pastel1',
                             'Pastel2', 'Set1', 'Set2', 'Set3']),
         ('Miscellaneous',  ['gist_earth', 'terrain', 'ocean', 'gist_stern',
                             'brg', 'CMRmap', 'cubehelix',
                             'gnuplot', 'gnuplot2', 'gist_ncar',
                             'nipy_spectral', 'jet', 'rainbow',
                             'gist_rainbow', 'hsv', 'flag', 'prism'])]
    """
    if not isinstance(raw_saver, save.RawDataSaver):
        raise TypeError("The input saver object should be pycup.save.RawDataSaver.")
    fitness = np.array(raw_saver.historical_fitness)
    iterations, population = analyze_saver(raw_saver)
    if iterations > 1:
        a = np.ones((fitness.shape[0], fitness.shape[1]))
        c = np.linspace(0, iterations, iterations)
        for i in range(fitness.shape[0]):
            a[i] = a[i] * c[i]
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        s = ax.scatter(xs=fitness[:, :, objfunid1], ys=fitness[:, :, objfunid2], zs=fitness[:, :, objfunid3], c=a,
                       marker=marker, s=markersize, cmap=cmap)
        bar = plt.colorbar(s, fraction=cbar_frac, pad=cbar_pad)
        bar.set_label(cbar_ttl, fontsize=cbar_ttl_size)
        bar.ax.tick_params(labelsize=cbar_lbl_size)

    else:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        s = ax.scatter(xs=fitness[:, objfunid1], ys=fitness[:, objfunid2], zs=fitness[:, objfunid3], c=single_c,
                       marker=marker, s=markersize, cmap=cmap)
    ax.view_init(*view_init)
    if slim:
        if iterations > 1:
            fitness = np.concatenate(fitness)
        xmin = np.min(fitness[:, objfunid1], axis=0)
        ymin = np.min(fitness[:, objfunid2], axis=0)
        zmin = np.min(fitness[:, objfunid3], axis=0)
        xmax = np.max(fitness[:, objfunid1], axis=0)
        ymax = np.max(fitness[:, objfunid2], axis=0)
        zmax = np.max(fitness[:, objfunid3], axis=0)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
    if x_lim:
        ax.set_xlim(*x_lim)
    if y_lim:
        ax.set_ylim(*y_lim)
    if z_lim:
        ax.set_zlim(*z_lim)
    ax = plt.gca()

    ax.tick_params(
        labelsize=tick_fontsize,
        direction=tick_dir
    )

    ax.set_xlabel(x_label, fontsize=lbl_fontsize)
    ax.set_ylabel(y_label, fontsize=lbl_fontsize)
    ax.set_zlabel(z_label, fontsize=lbl_fontsize)
    ax.ticklabel_format(style='sci', scilimits=(-3, 4), axis='z', useMathText=True)
    plt.title(title)
    if tight:
        plt.savefig(obj_path, bbox_inches='tight')
    else:
        plt.savefig(obj_path)
    if show:
        plt.show()


def plot_3d_sample_fitness_space(raw_saver,
                                 objfunid1,
                                 objfunid2,
                                 variable_id,
                                 obj_path="MOfitness3d.pdf",
                                 figsize=(8, 8),
                                 dpi=400,
                                 title=" ",
                                 lbl_fontsize=16,
                                 marker=".",
                                 markersize=50,
                                 x_label="Fitness1",
                                 y_label="Fitness2",
                                 z_label="Sample Value",
                                 tick_fontsize=14,
                                 tick_dir="out",
                                 cmap="plasma",
                                 single_c="b",
                                 cbar_ttl="Iterations",
                                 cbar_ttl_size=16,
                                 cbar_lbl_size=12,
                                 cbar_frac=0.03,
                                 cbar_pad=0.05,
                                 view_init=(),
                                 x_lim=(),
                                 y_lim=(),
                                 slim=True,
                                 tight=True,
                                 show=True
                                 ):
    """
    This function is for the users to plot the sample values versus fitness history for multi-objective algorithms. Three objective
    functions can be considered in this function.

    :argument
    raw_saver: caliboy.save.RawDataSaver objective
    objfunid1: index of the objective function 1 (column id for extracting data from the fitness array) -> int
    objfunid2: index of the objective function 2 (column id for extracting data from the fitness array) -> int
    variable_id: index of the variable -> int
    obj_path: target saving path -> str, default = "MOfitness3d.pdf", acceptable format = ["jpg", "png", "pdf", ...]
    figsize: the x, y size of the figure -> tuple, default = (8, 8)
    dpi: the resolution of the figure -> int, default = 400
    title: the title of the figure -> str
    lbl_fontsize: fontsize of axis labels -> int or float, default = 16
    marker: type of the markers in the scatter -> str, default = "." (dot like)
    markersize: the size of markers -> int or float, default = 50
    x_label: the x axis label -> str, default ="Fitness1"
    y_label: the y axis label -> str, default ="Fitness2"
    z_label: the z axis label -> str, default ="Fitness3"
    tick_fontsize: fontsize of tick labels -> int or float, default = 14
    tick_dir: tick direction -> str ("in" or "out"), default = "out"
    framewidth: width of the frame line -> float, default = 1.2
    cmap: color map -> str, default = "plasma"
    single_c: color -> str, default = "b", this will be used if the optimization method is GLUE (no iteration)
    cbar_ttl: the title of color bar -> str, default = "Iterations", users can use " " to replace it
    cbar_ttl_size: the size of color bar title -> int or float, default = 16
    cbar_lbl_size: the size of color bar tick labels -> int or float, default = 12
    cbar_frac: the size of color bar -> float, default = 0.05
    cbar_pad: the gap between cbar and the main figure -> float, default = 0.05
    view_init: view angle -> tuple, (float, float)
    x_lim: x axis plotting range -> tuple
    y_lim: y axis plotting range -> tuple
    z_lim: z axis plotting range -> tuple
    slim: the plotting range option -> bool, default = True, if True, the plotting range will be confined to the
          value ranges of the given data.
    tight: option for the white frame outside the plotting area -> bool, default = True, tight output (slim white frame)
    show: option for showing in the IDE -> bool, default = True

    Usage:
    import pycup as cp
    import numpy as np

    def mo_fun1(X):
        y1 = 1 - np.exp(-np.sum((X-1/np.sqrt(3))**2))
        y2 = 1 - np.exp(-np.sum((X+1/np.sqrt(3))**2))
        y3 = 1 - np.exp(np.sum((X**2 + np.sqrt(2))))
        fitness = np.array([y1,y2,y3]).reshape(1,-1)
        #fitness = np.array([y1,y2]).reshape(1,-1)
        result = fitness
        return fitness,result

    lb = np.array([-2, -2, -2])
    ub = np.array([2, 2, 2])
    cp.MOPSO.run(pop=100, dim=3, lb=lb, ub = ub, MaxIter=20,n_obj=3,nAr=300,M=50,
                       fun=mo_fun1)

    from pycup import plot,save

    path = "RawResult.rst"
    saver = save.RawDataSaver.load(path)
    plot.plot_3d_MO_fitness_space(saver,objfunid1=0,objfunid2=1,variable_id=2)


    Note:
    selectable cmaps includes [('Perceptually Uniform Sequential',
                            ['viridis', 'inferno', 'plasma', 'magma']),
         ('Sequential',     ['Blues', 'BuGn', 'BuPu',
                             'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
                             'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
                             'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
         ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool',
                             'copper', 'gist_heat', 'gray', 'hot',
                             'pink', 'spring', 'summer', 'winter']),
         ('Diverging',      ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
                             'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
                             'seismic']),
         ('Qualitative',    ['Accent', 'Dark2', 'Paired', 'Pastel1',
                             'Pastel2', 'Set1', 'Set2', 'Set3']),
         ('Miscellaneous',  ['gist_earth', 'terrain', 'ocean', 'gist_stern',
                             'brg', 'CMRmap', 'cubehelix',
                             'gnuplot', 'gnuplot2', 'gist_ncar',
                             'nipy_spectral', 'jet', 'rainbow',
                             'gist_rainbow', 'hsv', 'flag', 'prism'])]
    """
    if not isinstance(raw_saver, save.RawDataSaver):
        raise TypeError("The input saver object should be pycup.save.RawDataSaver.")
    fitness = np.array(raw_saver.historical_fitness)
    samples = np.array(raw_saver.historical_samples)
    iterations, population = analyze_saver(raw_saver)
    if iterations > 1:
        a = np.ones((fitness.shape[0], fitness.shape[1]))
        c = np.linspace(0, iterations, iterations)
        for i in range(fitness.shape[0]):
            a[i] = a[i] * c[i]
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        s = ax.scatter(xs=fitness[:, :, objfunid1], ys=fitness[:, :, objfunid2], zs=samples[:, :, variable_id], c=a,
                       marker=marker, s=markersize, cmap=cmap)
        bar = plt.colorbar(s, fraction=cbar_frac, pad=cbar_pad)
        bar.set_label(cbar_ttl, fontsize=cbar_ttl_size)
        bar.ax.tick_params(labelsize=cbar_lbl_size)

    else:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        s = ax.scatter(xs=fitness[:, objfunid1], ys=fitness[:, objfunid2], zs=samples[:, variable_id], c=single_c,
                       marker=marker, s=markersize, cmap=cmap)
    ax.view_init(*view_init)
    if slim:
        if iterations > 1:
            fitness = np.concatenate(fitness)
        xmin = np.min(fitness[:, objfunid1], axis=0)
        ymin = np.min(fitness[:, objfunid2], axis=0)
        xmax = np.max(fitness[:, objfunid1], axis=0)
        ymax = np.max(fitness[:, objfunid2], axis=0)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    if x_lim:
        ax.set_xlim(*x_lim)
    if y_lim:
        ax.set_ylim(*y_lim)
    ax = plt.gca()

    ax.tick_params(
        labelsize=tick_fontsize,
        direction=tick_dir
    )

    ax.set_xlabel(x_label, fontsize=lbl_fontsize)
    ax.set_ylabel(y_label, fontsize=lbl_fontsize)
    ax.set_zlabel(z_label, fontsize=lbl_fontsize)
    ax.ticklabel_format(style='sci', scilimits=(-3, 4), axis='z', useMathText=True)
    plt.title(title)
    if tight:
        plt.savefig(obj_path, bbox_inches='tight')
    else:
        plt.savefig(obj_path)
    if show:
        plt.show()


def plot_radar_sensitivity(raw_saver, objfun_id, plot, obj_path="radar.pdf", dpi=400, figsize=(8, 8), linewidth=2,
                           fill=False, alpha_fill=0.3,
                           parameter_labels=None, title=" ", legend_on=True, legend_loc=0, legend_labels=None,
                           frameon=True, show=True):
    """
    This is a plotting function for sensitivity analysis result based on the raw data saver. Users can plot both the absolute values of
    the t-stats and the p-values.
    :param raw_saver: caliboy.save.RawDataSaver objective
    :param objfun_id: the column index of the objective function, for a single-objective optimization it should be 0
    :param plot: 0 or "t" -> plot the t-stats, 1 or "p" -> plot the p-values
    :param obj_path:  target saving path -> str, default = "radar.pdf", acceptable format = ["jpg", "png", "pdf", ...]
    :param dpi: the resolution of the figure -> int, default = 400
    :param figsize: the x, y size of the figure -> tuple, default = (8, 8)
    :param linewidth: the line width of the plot -> default = 3
    :param fill: whether fill the polygons or not -> default = False
    :param alpha_fill: the transparency of the filled polygons, not valid if fill = False -> default = 0.3
    :param parameter_labels: a list of parameter names that you are analyzing
    :param title: the title of the figure -> str
    :param legend_on: a switch for the legend ->  default = True
    :param legend_loc: the location of you legend -> int
    :param legend_labels: a list of labels in your legend
    :param frameon: a switch for the frame of the legend  ->  default = True
    :param show: option for showing in the IDE -> bool, default = True
    :return:

    usage1: plot several s-stats results in a figure
    from pycup import save
    import pycup as cp
    r1 = cp.save.RawDataSaver.load("RawResultGLUE.rst")
    r2 = cp.save.RawDataSaver.load("RawResultPSO.rst")
    r3 = cp.save.RawDataSaver.load("RawResultSSA.rst")
    r4 = cp.save.RawDataSaver.load("RawResultGWO.rst")
    r5 = cp.save.RawDataSaver.load("RawResultNSGA2.rst")
    cp.plot.plot_radar_sensitivity([r1,r2,r3,r4,r5],0,0,linewidth=3,frameon=False,parameter_labels=["a","b","c","d","e","f","g","h","i","j","k","l"],
                            legend_labels = ["method1","method2","method3","method4","method5"])

    usage2: plot the t-stats result of 1 saver in a figure
    from pycup import save
    import pycup as cp
    r1 = cp.save.RawDataSaver.load("RawResultGLUE.rst")
    cp.plot.plot_radar_sensitivity(r1,0,0,linewidth=3,parameter_labels=["a","b","c","d","e","f","g","h","i","j","k","l"])
    """
    if not isinstance(raw_saver, list) and not isinstance(raw_saver, tuple):
        if not (isinstance(raw_saver, save.RawDataSaver)):
            raise ValueError("The given saver object is not a save.RawDataSaver.")

        if raw_saver.opt_type == "GLUE":
            hf = raw_saver.historical_fitness
            hs = raw_saver.historical_samples
        elif raw_saver.opt_type == "SWARM":
            hf = raw_saver.historical_fitness
            hs = raw_saver.historical_samples
            hs = np.concatenate(hs)
            hf = np.concatenate(hf)
        else:
            hf = raw_saver.historical_fitness
            hs = raw_saver.historical_samples
            hs = np.concatenate(hs)
            hf = np.concatenate(hf)
        hf = hf[:, objfun_id]
        mlr_mdl = sm.OLS(hf, sm.add_constant(hs))
        res = mlr_mdl.fit()
        t_stats = np.abs(res.tvalues)[1:]
        p_values = np.array(res.pvalues)[1:]
        dataLenth = len(t_stats)
        if plot == "t" or plot == 0:
            data = t_stats
        elif plot == "p" or plot == 1:
            data = p_values
        else:
            raise ValueError("The argument plot should be 'p' or 1 for p-values or 't' or 0 for t-stats.")
        angles = np.linspace(0, 2 * np.pi, dataLenth, endpoint=False)
        if parameter_labels == None:
            labels = np.array(["x{}".format(i) for i in range(hs.shape[1])])
        else:
            labels = parameter_labels
        data = np.append(data, data[0])
        angles = np.append(angles, angles[0])
        labels = np.append(labels, labels[0])
        fig = plt.figure(facecolor="white", dpi=dpi, figsize=figsize)
        plt.subplot(111, polar=True)
        plt.plot(angles, data, 'bo-', color='g', linewidth=linewidth)
        if fill:
            plt.fill(angles, data, facecolor="g", alpha=alpha_fill)
        plt.thetagrids(angles * 180 / np.pi, labels, fontsize=30)
        plt.yticks(fontsize=20)
        plt.grid(True)
        plt.title(title)
        plt.savefig(obj_path)
        if show:
            plt.show()
    else:
        if legend_labels == None:
            legend_labels = ["Algorith{}".format(i) for i in range(len(raw_saver))]
        fig = plt.figure(facecolor="white", dpi=dpi, figsize=figsize)
        plt.subplot(111, polar=True)
        cs = ['r', 'g', 'b', 'y', "brown", "gray", "magenta", "cyan", "orange", "purple", "k", "pink"]
        for i, rs in enumerate(raw_saver):
            if not (isinstance(rs, save.RawDataSaver)):
                raise ValueError("The given saver object is not a save.RawDataSaver.")

            if rs.opt_type == "GLUE":
                hf = rs.historical_fitness
                hs = rs.historical_samples
            elif rs.opt_type == "SWARM":
                hf = rs.historical_fitness
                hs = rs.historical_samples
                hs = np.concatenate(hs)
                hf = np.concatenate(hf)
            else:
                hf = rs.historical_fitness
                hs = rs.historical_samples
                hs = np.concatenate(hs)
                hf = np.concatenate(hf)
            hf = hf[:, objfun_id]
            mlr_mdl = sm.OLS(hf, sm.add_constant(hs))
            res = mlr_mdl.fit()
            t_stats = np.abs(res.tvalues)[1:]
            p_values = np.array(res.pvalues)[1:]
            dataLenth = len(t_stats)
            if plot == "t" or plot == 0:
                data = t_stats
            elif plot == "p" or plot == 1:
                data = p_values
            else:
                raise ValueError("The argument plot should be 'p' or 1 for p-values or 't' or 0 for t-stats.")
            angles = np.linspace(0, 2 * np.pi, dataLenth, endpoint=False)
            if parameter_labels == None:
                labels = np.array(["x{}".format(i) for i in range(hs.shape[1])])
            else:
                labels = parameter_labels
            data = np.append(data, data[0])
            angles = np.append(angles, angles[0])
            labels = np.append(labels, labels[0])
            plt.plot(angles, data, 'bo-', color=cs[i], linewidth=linewidth, label=legend_labels[i])
            if fill:
                plt.fill(angles, data, facecolor=cs[i], alpha=alpha_fill)
        plt.thetagrids(angles * 180 / np.pi, labels, fontsize=30)
        plt.yticks(fontsize=20)
        plt.grid(True)
        plt.title(title)
        if legend_on:
            plt.legend(loc=legend_loc, fontsize=16, frameon=frameon)
        plt.savefig(obj_path)
        if show:
            plt.show()


def plot_behaviour_violins(proc_savers, variable_id, obj_path="violin.jpg", figsize=(6, 4), dpi=600,
                           x_ticklabels=None, xlabel=None, ylabel=None,
                           lbl_fontsize=16, tick_fontsize=16, show=True,
                           bottom_adjust=0.14, left_adjust=0.15):
    """
    :param proc_savers: a list or a tuple of pycup.save.ProcResultSaver -> list
    :param variable_id: the column index the variable that you want to plot -> int
    :param obj_path: the saving path of your figure -> str, default = "violin.jpg"
    :param figsize: the x, y size of the figure -> tuple, default = (8, 8)
    :param dpi: the resolution of the figure -> int, default = 400
    :param x_ticklabels: a list of tick labels. -> list
    :param xlabel: x axis label -> str
    :param ylabel: y axis label -> str
    :param lbl_fontsize: the fontsize of x and y label -> int, default = 16
    :param tick_fontsize: the fontsize of x and y tick labels -> int, default = 16
    :param show: option for showing in the IDE -> bool, default = True
    :param bottom_adjust: the bottom position adjust value to adjust the width of the gap -> float
    :param left_adjust: the left edge position adjust value to adjust the width of the gap -> float
    :return:

    usage:
    from pycup import plot,save
    saver1 = save.ProcResultSaver.load("ProcResult1.rst")
    saver2 = save.ProcResultSaver.load("ProcResult2.rst")
    plot.plot_behaviour_violins([saver1,saver2],0)
    """
    if not isinstance(proc_savers, tuple) and not isinstance(proc_savers, list):
        raise TypeError("The input data should be a list or a tuple of pycup.save.ProcResultSaver.")
    for i in proc_savers:
        if not isinstance(i, save.ProcResultSaver):
            raise TypeError("The input savers should be pycup.save.ProcResultSaver, please check.")
    dataset = [i.behaviour_results.behaviour_samples[:, variable_id].flatten() for i in proc_savers]
    plt.figure(figsize=figsize, dpi=dpi)
    violin = plt.violinplot(dataset=dataset, showextrema=False)
    plt.grid(c="lightgray", linestyle="--", zorder=0)
    for patch in violin['bodies']:
        patch.set_facecolor('#D43F3A')
        patch.set_edgecolor('black')
        patch.set_alpha(1)
    for i, d in enumerate(dataset):
        min_value, quantile1, median, quantile3, max_value = np.percentile(d, [0, 25, 50, 75, 100])
        mean = np.mean(d)
        plt.scatter(i + 1, median, color='white', zorder=4)
        plt.scatter(i + 1, mean, color='royalblue', zorder=5)
        plt.vlines(i + 1, quantile1, quantile3, lw=8, zorder=3)
        plt.vlines(i + 1, min_value, max_value, zorder=2)
    if x_ticklabels:
        plt.xticks(ticks=np.arange(1, len(dataset) + 1), labels=x_ticklabels, fontsize=tick_fontsize)
    else:
        plt.xticks(ticks=np.arange(1, len(dataset) + 1),
                   labels=["Algorithm{}".format(i + 1) for i in range(len(dataset))], fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    if xlabel:
        plt.xlabel(xlabel, fontsize=lbl_fontsize)
    else:
        plt.xlabel("Algorithm", fontsize=lbl_fontsize)
    if ylabel:
        plt.ylabel(ylabel, fontsize=lbl_fontsize)
    else:
        plt.ylabel("Variable", fontsize=lbl_fontsize)
    plt.subplots_adjust(bottom=bottom_adjust, left=left_adjust)
    plt.savefig(obj_path)
    if show:
        plt.show()


def analyze_saver(raw_saver):
    historical_sample_val = np.array(raw_saver.historical_samples, dtype=object)
    if len(historical_sample_val[0].shape) > 1:
        iterations = len(historical_sample_val)
        population = np.array([i.shape[0] for i in historical_sample_val])
    else:
        iterations = 1
        population = len(historical_sample_val)

    return iterations, population