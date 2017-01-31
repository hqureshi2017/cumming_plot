import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def _jitter(data, jit=None):
    """
    Calculate y-axis jitter for two series of data.

    Parameters
    ----------
    data : DataFrame
        Two series of data to which y-axis jitter wants to be added.
    jit : int, default None

    Returns
    -------
    jitter_a: np.array
        Y-axis jitter values for data[0]
    jitter_b: np.array
        Y-axis jitter values for data[1]

    Usage
    --------
    >>> a = [1, 2, 3, 3, 1]
    >>> b = [1, 4, 3, 2, 3]
    >>> data = [a, b]
    >>> a_jit, b_jit = jitter(data)
    >>> a_jit
    [0, 0, 0, 0.005, 0.005]
    >>> b_jit
    [0, 0, 0, 0, 0.005]

    """

    if jit is None:
        jit = 0.005

    a = data[0]
    b = data[1]

    duplicate_a = []
    duplicate_b = []
    jitter_a = [''] * len(a)
    jitter_b = [''] * len(b)
    jitter_a[0] = 0
    jitter_b[0] = 0

    for i in np.arange(1, len(a), 1):
        a_val = a.values[i]
        b_val = b.values[i]
        if a_val in a.values[0:i]:
            duplicate_a.append(a_val)
            val = jit * duplicate_a.count(a_val)
            jitter_a[i] = val
        else:
            jitter_a[i] = 0

        if b_val in b.values[0:i]:
            duplicate_b.append(b_val)
            val = jit * duplicate_b.count(b_val)
            jitter_b[i] = val
        else:
            jitter_b[i] = 0
    return jitter_a, jitter_b


def paired(data, ax, ab_errors='95%CI', yticks='default',
          dif_jit=None, ab_jit=None, style=None, ylabel=None,
          xlabel=None, zero_line=False, y2label=None, y2ticks=False,
          likert=False):

    """
    Parameters
    ----------
    data : list
        Paired data to be plotted [data_a, data_b]. Length of data_a and data_b
    must be equal.
    ab_errors : str, default '95%CI'
        Type of error bar to be plotted for data_a and data_b. Set to 'SD'
        to plot standard deviations.
    yticks : list, default None
        When not specified, y-range and y-ticks will default to those selected by matplotlib. However, these can be be
        specified by provide a list containing [ymin, ymax, step_size]
    dif_jit : int, default None
        Size of the y-axis jitter for the raw difference data. Defaults to 0.001.
    ab_jit : int, default None
        Size of the y-axis jitter for the raw a and b data. Defaults to 0.005.
    style : dict, default None
        Set marker color and shape. Defaults to {'a': ['wo', 'k'], 'b': ['ko', 'k'],'diff': ['k^', 'k']}.
    ylabel : str, default None
        Set y-axis label. Default is to not have a label.
    xlabel : list, default None
        Set x-axis labels. Default is to not have labels. Provide a list of 3 str [<a label>, <b label>, <diff label>]
    zero_line : bool, Default False
        Plot dashed line at zero across data a and b. This is useful if plotting difference of differences data.
    y2label : str, default None
        Set y-axis label. Default is to not have a label.
    y2ticks : bool, default False
        Set to add numerical values to right y-axis. Default is to not have any values.
    likert : bool, default False
        Set to indicate that plotted data come from a 7-point Likert scale. This will insert appropriate y-tick labels.

    Usage
    -----
    > # Generate fake data
    > import cumming_plot
    > from random import randint
    > start = [randint(1,4) for i in range(30)]
    > end = [randint(2,7) for i in range(30)]
    > data = [start, end]
    >
    > # Simple plot
    > import matplotlib.pyplot as plt
    > fig = plt.figure()
    > ax = fig.add_subplot(111)
    > cumming_plot.paired(data, ax)
    > plt.show()
    >
    > # Complex plot
    > ab_errors = 'SD'
    > yticks = [-10, 30, 10]
    > style = {'a': ['r*', 'k'], 'b':['yo', 'g'], 'diff': ['w^', 'm']}
    > ylabel = 'y label'
    > xlabel = ['start', 'end', 'dif']
    > zero_line = True
    > y2label = 'y2 difference axis'
    > y2ticks = True
    > cumming_plot.paired(data, ax, ab_errors=ab_errors, yticks=yticks,
                        style=style, ylabel=ylabel, xlabel=xlabel,
                        zero_line=zero_line, y2label=y2label,
                        y2ticks=y2ticks)
    >
    > # Likert-scale data plot
    > start = [randint(1,4) for i in range(30)]
    > end = [randint(2,7) for i in range(30)]
    > data = [start, end]
    > fig = plt.figure()
    > ax = fig.add_subplot(1,1,1)
    > cumming_plot.paired(data, ax, y2ticks=True, likert=True)
    > plt.show()

    """
    # Set matplotlib options
    mpl.rcParams['font.size'] = 16
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 16

    data[0] = pd.Series(data[0])
    data[1] = pd.Series(data[1])

    if ab_jit is None:
        ab_jit = 0.005
    if dif_jit is None:
        dif_jit = 0.001
    if style is None:
        style = {'a': ['wo', 'k'],
                 'b': ['ko', 'k'],
                 'diff': ['k^', 'k']}

    # x-axis spacing [a, raw a, space, raw b, b]
    x_spacing = [0.05, 0.1, 0.3, 0.35, 0.45]
    jitter_a, jitter_b = _jitter(data, ab_jit)

    if zero_line:
        x_val = [0, x_spacing[3] + (x_spacing[3] - x_spacing[2]) / 2]
        ax.plot(x_val, [0, 0], '--k')

    # Plot lines connecting paired points
    for a, b, j_a, j_b in zip(data[0], data[1], jitter_a, jitter_b):
        x_val = [x_spacing[1] + j_a, x_spacing[2] - j_b]
        ax.plot(x_val, [a, b], '-', color='0.75')

    # Plot raw data points for a
    ones = np.ones(len(data[0]))
    x_val_a = ones * x_spacing[1] + jitter_a
    ax.plot(x_val_a, data[0], style['a'][0],
            markeredgecolor=style['a'][1], markersize=5)

    # Plot raw data points for b
    x_val_b = ones * x_spacing[2] - jitter_b
    ax.plot(x_val_b, data[1], style['b'][0],
            markeredgecolor=style['b'][1], markersize=5)

    # Calculate mean [error_bar] for data a and b
    a_mean = data[0].mean()
    b_mean = data[1].mean()
    if ab_errors == '95%CI':
        a_error = data[0].sem() * 1.96
        b_error = data[1].sem() * 1.96
    elif ab_errors == 'SD':
        a_error = data[0].std()
        b_error = data[1].std()
    a_error_min = a_mean - a_error
    a_error_max = a_mean + a_error
    b_error_min = b_mean - b_error
    b_error_max = b_mean + b_error

    # Plot mean [error_bar] for data a and b
    ax.plot([x_spacing[0], x_spacing[0]], [a_error_min, a_error_max],
            '-' + style['a'][1], linewidth=2)
    ax.plot(x_spacing[0], a_mean, style['a'][0],
            markeredgecolor=style['a'][1], markersize=10)

    ax.plot([x_spacing[3], x_spacing[3]], [b_error_min, b_error_max],
            '-' + style['b'][1], linewidth=2)
    ax.plot(x_spacing[3], b_mean, style['b'][0],
            markeredgecolor=style['b'][1], markersize=10)

    # Create second y-axis
    ax2 = ax.twinx()
    # Calculate differences [b-a]
    BA_dif = data[1] - data[0]
    # Plot difference values
    jit = 0
    for val in BA_dif:
        ax2.plot(x_spacing[4] + jit, a_mean + val, style['diff'][0],
                 markeredgecolor=style['diff'][1], markersize=5)
        jit += dif_jit
    # Calculate x-value where to plot mean [95% CI]
    dif_x = x_spacing[4] + jit + (x_spacing[3] - x_spacing[2])
    # Calculate and plot mean [95% CI] for difference
    dif_mean = BA_dif.mean()
    dif_95 = BA_dif.sem() * 1.96
    y1 = dif_mean - dif_95
    y2 = dif_mean + dif_95
    ax2.plot([dif_x, dif_x], [a_mean + y1, a_mean + y2],
             '-' + style['diff'][1], linewidth=2)
    ax2.plot(dif_x, a_mean + dif_mean, style['diff'][0],
             markeredgecolor=style['diff'][1], markersize=10)
    # Plot dashed line at zero for difference (in line with mean a)
    ax2.plot([x_spacing[4] - ((x_spacing[4] - x_spacing[3]) / 2), dif_x + x_spacing[0]], [a_mean, a_mean],
             '--' + style['diff'][1])

    # Hide unwanted axes and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom='off',
                   top='off', labelbottom='off')
    ax2.tick_params(axis='x', which='both', bottom='off',
                    top='off', labelbottom='off')

    # Set axes colors
    ax.spines['right'].set_color(style['diff'][1])
    ax2.spines['right'].set_color(style['diff'][1])
    ax2.yaxis.label.set_color(style['diff'][1])
    ax2.tick_params(axis='y', colors=style['diff'][1])

    # Set x-axis limits
    ax.set_xlim([0, dif_x + x_spacing[0]])
    ax2.set_xlim([0, dif_x + x_spacing[0]])

    # Set axes labels
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if y2label is not None:
        ax2.set_ylabel(y2label)

    # Find min and max plotted value; used to set y-axis limits/ticks
    a = np.array(data[0])
    b = np.array(data[1])
    BAdif = np.array(BA_dif)
    min_val = min([a.min(), b.min(), BAdif.min() + a_mean])
    max_val = max([a.max(), b.max(), BAdif.max() + a_mean])

    # Set y-axis tick range and spacing if set by user
    if likert:
        ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
        ax.set_yticklabels(['(1) st. disagree', '(2) disagree',
                            '(3) som. disagree', '(4) neutral',
                            '(5) som. agree', '(6) agree', '(7) st. agree'])
    else:
        if yticks != 'default':
            ytick_vals = np.arange(yticks[0], yticks[1] + 1, yticks[2])
            str_ytick = str(yticks[2])
            if str_ytick[str_ytick.find('.') + 1] == '0':
                ytick_vals_verified = [int(val) for val in ytick_vals]
            else:
                ytick_vals_verified = ytick_vals
            ax.set_yticks(ytick_vals)
            ax.set_yticklabels(ytick_vals_verified)

    # Set tick labels for y2-axis
    if likert:
        y2tick_vals = np.arange(-14, 7, 1)
        tick_loc = y2tick_vals + a_mean
        ax2.set_yticks(tick_loc)
        ax2.set_yticklabels(y2tick_vals)
    else:
        if y2ticks:
            ticks = ax.get_yticks()
            y_tick_sp = ticks[1] - ticks[0]
            start = 0 - y_tick_sp * len(ticks)
            end = 0 + y_tick_sp * len(ticks)
            y2tick_vals = np.arange(start, end, y_tick_sp)
            y2tick_vals_verified = []
            str_ytick = str(y_tick_sp)
            if str_ytick[str_ytick.find('.') + 1] == '0':
                y2tick_vals_verified = [int(val) for val in y2tick_vals]
            else:
                y2tick_vals_verified = y2tick_vals
            tick_loc = y2tick_vals + a_mean
            ax2.set_yticks(tick_loc)
            ax2.set_yticklabels(y2tick_vals_verified)
        else:
            ax2.tick_params(axis='y', labelright='off')

            # Set limits for y-axis and y2-axis
    if likert:
        ax.set_ylim([min_val - 1, max_val + 1])
        ax2.set_ylim([min_val - 1, max_val + 1])
    else:
        if yticks != 'default':
            ax.set_ylim([yticks[0], yticks[1]])
            ax2.set_ylim([yticks[0], yticks[1]])
        else:
            ax.set_ylim([min_val - 1, max_val + 1])
            ax2.set_ylim([min_val - 1, max_val + 1])

    if xlabel is not None:
        if yticks != 'default':
            ticks = ax.get_yticks()
            y_val = ticks[0] - ((ticks[1] - ticks[0]) / 8)
        else:
            y_val = min_val - 2
        ax.text(x_spacing[0] - 0.01, y_val, xlabel[0], fontsize=14, va='top')
        ax.text(x_spacing[2] - 0.01, y_val, xlabel[1], fontsize=14, va='top')
        ax.text(x_spacing[4] + 0.02, y_val, xlabel[2], fontsize=14, va='top')


def _usage():
    """Print usage on command line."""

    print('\nPlot paired data and their difference based on the approach '
          'promoted by Cumming and Calin-Jageman (2017). \n\n'
          'From the command line:\n'
          '\t Plot data from a file:\n'
          '\t$ python cumming_plot.py <data> <param>\n'
          '\t\t<data> -> 2 columns of paired data to plot\n'
          '\t\tparam -> optional, specify parameters to plot function; in '
          'quotation marks\n\n'
          '\t$ python cumming_plot.py data.txt "zero_line=True,'
          'y2ticks=True"\n\n'
          '\tPlot series of examples: \n'
          '\t$ python cumming_plot.py examples\n'
          '\n\nFrom Python program:\n'
          '\timport cumming_plot\n'
          '\tfrom random import randint\n'
          '\tstart = [randint(1, 4) for i in range(30)]\n'
          '\tend = [randint(2, 7) for i in range(30)]\n'
          '\tdata = [start, end]\n'
          '\t# Simple plot\n'
          '\timport matplotlib.pyplot as plt\n'
          '\tfig = plt.figure()\n'
          '\tax = fig.add_subplot(111)\n'
          '\tcumming_plot.paired(data, ax)\n'
          '\tplt.show()\n\n'
    'reference:\nCumming G & Calin-Jageman R (2017). Introduction the New '
        'Statistics: \nEstimation, Open Science & Beyond. Routledge, '
        'East Sussex.\n')


def _examples_paired():
    from random import randint
    from random import choice

    ####################################
    # BASIC EXAMPLES WITH NUMERICAL DATA
    ####################################
    # Create fake data for plotting
    a = [randint(10, 21) for i in range(30)]
    b = [randint(0, 15) for i in range(30)]
    ab = [a, b]

    # SIMPLE PLOT
    fig = plt.figure()
    ax = fig.add_subplot(111)
    paired(ab, ax)
    plt.suptitle('Simple example, close to see next...')
    plt.show()

    # COMPLEX EXAMPLE
    ab_errors='SD'
    yticks = [-10, 30, 10]
    style = {'a': ['r*', 'k'], 'b': ['yo', 'g'], 'diff': ['w^', 'm']}
    ylabel = 'y label'
    xlabel = ['a', 'b', 'diff']
    zero_line = True
    y2label = 'y2 difference axis'
    y2ticks = True
    fig = plt.figure()
    ax = fig.add_subplot(111)
    paired(ab, ax, ab_errors=ab_errors, yticks=yticks,
                style=style, ylabel=ylabel, xlabel=xlabel,
                zero_line=zero_line, y2label=y2label,
                y2ticks=y2ticks)
    plt.suptitle('More complex example, close to see next...')
    plt.show()

    #######################
    # SIMPLE LIKERT EXAMPLE
    #######################
    # Generate fake Likert-scale data
    from random import randint
    a = [randint(1, 4) for i in range(30)]
    b = [randint(2, 7) for i in range(30)]
    ab = [a, b]

     # Generate simple plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    paired(ab, ax, y2ticks='True', likert=True)
    plt.suptitle('Example with Likert scale, close to see next...')
    plt.show()

    #############################
    # CRAZY EXAMPLE WITH SUBPLOTS
    #############################
    style1 = {'a': ['bo', 'b'], 'b': ['ro', 'r'], 'diff': ['g^', 'g']}
    style2 = {'a': ['b8', 'm'], 'b': ['yd', 'r'], 'diff': ['cp', 'k']}
    style3 = {'a': ['g*', 'k'], 'b': ['y*', 'b'], 'diff': ['cv', 'm']}
    style4 = {'a': ['rh', 'b'], 'b': ['mp', 'b'], 'diff': ['sk', 'y']}
    style5 = {'a': ['ko', 'b'], 'b': ['bo', 'r'], 'diff': ['gs', 'b']}
    style6 = {'a': ['y+', 'g'], 'b': ['kd', 'k'], 'diff': ['mx', 'b']}
    style7 = {'a': ['g^', 'r'], 'b': ['m*', 'r'], 'diff': ['r<', 'k']}
    style8 = {'a': ['mo', 'c'], 'b': ['cp', 'b'], 'diff': ['y>', 'm']}

    styles = [style1, style2, style3, style4, style5, style6, style7, style8]

    fig = plt.figure()
    subplots = range(1,9)

    for i, subplot in enumerate(subplots):
        # Set subplot parameters
        ab_errors = choice(['SD', '95%CI'])
        yticks = choice([[-10, 30, 10], [-10, 40, 5], 'default'])
        style = styles[i]
        ylabel = choice(['y label', 'amplitude (cm)', 'volume (L)', None])
        xlabel = choice([['START', 'END', 'diff'], ['ME', 'ME2', '2'], None])
        zero_line = choice([True, False])
        y2label = choice(['y2 difference axis', 'difference', None])
        y2ticks = choice([True, False])

        # Generate fake data
        n = randint(5, 50)
        a = [randint(10, 21) for i in range(n)]
        b = [randint(0, 15) for i in range(n)]
        ab = [a, b]

        # Generate subplot
        ax = fig.add_subplot(4, 2, subplot)
        paired(ab, ax, ab_errors=ab_errors, yticks=yticks,
                style=style, ylabel=ylabel, xlabel=xlabel,
                zero_line=zero_line, y2label=y2label,
                y2ticks=y2ticks)
    #plt.tight_layout()
    plt.suptitle('Crazy example, close to finish.')
    plt.show()


def _plot_data(argv):
    """
    Plot data in provided file.

    Parameters
    ----------
    argv : list
     Filename containing data to be plotted argv[0] and, if provided,
     parameters for the call to paired() argv[1].

    """
    with open(argv[0]) as f:
        dat = f.readlines()
    a = []
    b = []
    for row in dat:
        row_dat = row.split(',')
        a.append(float(row_dat[0]))
        b.append(float(row_dat[1]))
        ab = [a, b]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if len(argv) == 2:
        eval('paired(ab,ax,' + argv[1] + ')')
    else:
        paired(ab, ax)
    plt.show()

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        _usage()
    elif sys.argv[1] == 'examples':
        _examples_paired()
    else:
        _plot_data(sys.argv[1:])


