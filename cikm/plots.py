import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import set_matplotlib_formats
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
import os

set_matplotlib_formats('pdf', 'png')


def to_percent(y, position=None):
    """Format tick labels of a given axis to percentages

    - The argument should be a normalized value such that -1 <= y <= 1
    - Ignore the passed in position; it scales the default tick locations.
    """
    assert -1 <= y <= 1
    s = int(float(str(100 * y)))

    # The percent symbol needs escaping in latex
    return '${}\%$'.format(s) if mpl.rcParams['text.usetex'] else '{}%'.format(s)


def inches_to_points(inches):
    inches_per_pt = 1.0 / 72.27  # to convert pt to inches
    return inches / inches_per_pt


def points_to_inches(points):
    inches_per_pt = 1.0 / 72.27  # to convert pt to inches
    return points * inches_per_pt


def latexify(column_width_pt=243.91125, text_width_pt=505.89, scale=2, fontsize_pt=11, usetex=True):

    # sorted([f.name for f in mpl.matplotlib.font_manager.fontManager.ttflist])

    fig_width_pt = column_width_pt
    inches_per_pt = 1.0 / 72.27  # to convert pt to inches
    golden_mean = 0.61803398875  # (math.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
    fig_proportion = golden_mean
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * fig_proportion  # height in inches
    fig_size = [scale * fig_width, scale * fig_height]

    # Legend
    plt.rcParams['legend.fontsize'] = 14  # in pts (e.g. "x-small")

    # Lines
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['lines.linewidth'] = 2.0

    # Ticks
    # plt.rcParams['xtick.labelsize'] = 'x-small'
    # plt.rcParams['ytick.labelsize'] = 'x-small'
    # plt.rcParams['xtick.major.pad'] = 1
    # plt.rcParams['ytick.major.pad'] = 1

    # Axes
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 18
    # plt.rcParams['axes.labelpad'] = 0

    # LaTeX
    plt.rcParams['text.usetex'] = usetex
    plt.rcParams['text.latex.unicode'] = True
    plt.rcParams['text.latex.preview'] = False

    # use utf8 fonts becasue your computer can handle it :)
    # plots will be generated using this preamble
    plt.rcParams['text.latex.preamble'] = [
        r'\usepackage[utf8x]{inputenc}',
        r'\usepackage[T1]{fontenc}',
        r'\usepackage{amssymb}',
        r'\usepackage{amsmath}',
        r'\usepackage{wasysym}',
        r'\usepackage{stmaryrd}',
        r'\usepackage{subdepth}',
        r'\usepackage{type1cm}'
    ]

    # Fonts
    plt.rcParams['font.size'] = fontsize_pt  # font size in pts (good size 16)
    plt.rcParams['font.family'] = 'sans-serif'  # , 'Merriweather Sans'  # ['DejaVu Sans Display', "serif"]
    # plt.rcParams['font.serif'] = ['Merriweather',
    #                               'cm']  # blank entries should cause plots to inherit fonts from the document
    # plt.rcParams['font.sans-serif'] = 'Merriweather Sans'
    # plt.rcParams['font.monospace'] = 'Operator Mono'

    # Figure
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 75
    plt.rcParams['savefig.pad_inches'] = 0.01
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['figure.autolayout'] = False
    plt.rcParams['figure.figsize'] = fig_size
    plt.rcParams['image.interpolation'] = 'none'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42


def plot_pros_and_cons(df=None,
                       ax=None,
                       bw_min=0,
                       bw_max=0.9,
                       rank_col=None,
                       frameon=False,
                       lw=0.5,
                       fname_suffix='',
                       ylabel='\# Features',
                       xlabel=None,
                       markersize=5,
                       pros_min=-6,
                       pros_max=6,
                       show_xlabel=True,
                       title=None,
                       is_compelling=False,
                       title_fontsize='small',
                       rotation=0,
                       fname_prefix='',
                       markeredgewidth=0.5,
                       tick_length=1.0,
                       tick_width=0.5,
                       out_dir=None,
                       show_xticklabels=True,
                       minor_ylabel: bool = True,
                       major_ylabel: bool = True):
    """

    :param df:
    :param bw_min:
    :param bw_max:
    :param rank_col:
    :param frameon:
    :param lw:
    :param fname_suffix:
    :param ylabel:
    :param xlabel:
    :param markersize:
    :param pros_min:
    :param pros_max:
    :param title:
    :param is_compelling:
    :param title_fontsize:
    :param rotation:
    :param fname_prefix:
    :param markeredgewidth:
    :param out_dir:
    :param minor_ylabel: Whether or not the minor ylabel should be shown
    :param major_ylabel: Whether or not the major ylabel should be shown
    :return:
    """
    if is_compelling:
        cols = ['n_pros_comp', 'n_cons_comp', 'better_avg_comp', 'worse_avg_comp']
    else:
        cols = ['n_pros', 'n_cons', 'better_avg', 'worse_avg']

    aggs = {col: np.mean for col in cols}
    rank_col = rank_col if rank_col else 'rank_target_item_average_rating'
    df[rank_col] = df[rank_col].astype(int)

    grouped_ranks = df.groupby(rank_col).agg(aggs)

    params = dict(edgecolor='black', legend=False, lw=lw)
    if is_compelling:
        grouped_ranks.n_cons_comp = -grouped_ranks.n_cons_comp
        ax = grouped_ranks[['n_pros_comp']].plot(kind='bar', ax=ax, color='whitesmoke', rot=rotation, **params)
        grouped_ranks[['n_cons_comp']].plot(kind='bar', ax=ax, color='dimgray', **params)
        bw_cols = ['better_avg_comp', 'worse_avg_comp']
    else:
        grouped_ranks.n_cons = -grouped_ranks.n_cons
        ax = grouped_ranks[['n_pros']].plot(kind='bar', ax=ax, color='whitesmoke', rot=rotation, **params)
        grouped_ranks[['n_cons']].plot(kind='bar', ax=ax, color='dimgray', rot=rotation, **params)
        bw_cols = ['better_avg', 'worse_avg']

    grouped_ranks[bw_cols[1]] = grouped_ranks[bw_cols[1]] * -1

    ax.set_ylim(ymin=pros_min, ymax=pros_max)
    ax2 = ax.twinx()
    ax2.set_ylim(ymin=bw_min, ymax=bw_max)
    ax2.grid(False)

    if minor_ylabel:
        formatter = FuncFormatter(to_percent)
        ax2.yaxis.set_major_formatter(formatter)
    else:
        ax2.yaxis.label.set_visible(False)
        ax2.set_yticklabels([])

    ax2.set_ylim(-1, 1)

    grouped_ranks = grouped_ranks.reset_index()
    grouped_ranks[rank_col] = grouped_ranks[rank_col] - 1
    grouped_ranks = grouped_ranks.set_index(rank_col)

    params = dict(markeredgecolor='black', markeredgewidth=markeredgewidth, legend=False, color='black',
                  ms=markersize, zorder=10, alpha=1, linewidth=1, grid=False)
    grouped_ranks[bw_cols[0]].plot(ax=ax2, markerfacecolor='whitesmoke', label='Better', marker='o', **params)
    grouped_ranks[bw_cols[1]].plot(ax=ax2, markerfacecolor='black', label='Worse', marker='s', **params)

    if minor_ylabel:
        ax2.set_ylabel('\% Better/Worse')
    else:
        print('Removing minor ylabels')
        n_labels = len(ax2.get_yticklabels())
        ax2.set_ylabel([1]*n_labels)
        # ax2.yaxis.label.set_visible(False)

    # create patches for legend
    cons_patch = mpatches.Patch(facecolor='dimgray', linewidth=0.5, edgecolor='black', label='C')
    pros_patch = mpatches.Patch(facecolor='whitesmoke', linewidth=0.5, edgecolor='black', label='P')

    ax2_handles, ax2_labels = ax2.get_legend_handles_labels()

    legend = ax.legend(handles=[cons_patch, pros_patch, ax2_handles[1], ax2_handles[0]],
                       scatterpoints=1, ncol=4, numpoints=1, loc='lower left', frameon=frameon,
                       columnspacing=0.5, fontsize='x-small',
                       # borderpad=0.0,
                       )
    legend.get_frame().set_linewidth(lw)

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation)
    if major_ylabel:
        ax.set_ylabel(ylabel)

    xlabel = xlabel if xlabel else rank_col.replace('_', '-')
    ax.axhline(y=0, lw=1.35, color='k')
    ax.set_xlabel(xlabel)
    ax.set_xlim(-0.5, 9.5)

    if not show_xticklabels:
        ax.set_xticklabels([])
        ax2.set_xticklabels([])

    ax.tick_params('both', length=tick_length, width=tick_width, which='both')
    ax2.tick_params('both', length=tick_length, width=tick_width, which='both')

    if title:
        plt.title(title, fontsize=title_fontsize)

    file_suffix = 'com' if is_compelling else 'bas'

    plt.tight_layout()
    if out_dir:
        fname_suffix = fname_suffix if fname_suffix else rank_col.replace('_', '-')
        fname_tmpl = '{fname_prefix}pcbw-per-rank-{file_suffix}{fname_suffix}'
        path = os.path.join(out_dir, fname_tmpl.format(fname_prefix=fname_prefix,
                                                       file_suffix=file_suffix,
                                                       fname_suffix=fname_suffix))
        #         plt.savefig('{}.png'.format(path), bbox_inches='tight')
        plt.savefig('{}.pdf'.format(path), bbox_inches='tight')
        plt.savefig('{}.png'.format(path), bbox_inches='tight')
    return
