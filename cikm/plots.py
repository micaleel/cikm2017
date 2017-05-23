import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import set_matplotlib_formats
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
import os
from itertools import cycle
import matplotlib.pyplot as plt
import pandas as pd
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


def plot_avg_strengths_for_weight(weight, column_width_pt, explanations,
                                  figures_dir,
                                  title_fontsize=11, xlabel_fontsize=10,
                                  legend_kwds=None, wspace=0.0,
                                  markersize=5, dataset_names=None, lw=0.75):
    """Plot average strength per rank."""
    assert weight in ('uw', 'iw', 'nw')
    legend_kwds = legend_kwds or dict(fontsize='x-small', borderpad=0.35)
    dataset_names = dataset_names or {'ta': 'TripAdvisor', 'yp': 'Yelp',
                                      'ba': 'BeerAdvocate'}
    latexify(usetex=True, scale=1, column_width_pt=column_width_pt * 2)
    figsize = (points_to_inches(column_width_pt * 2), 2.5)
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True,
                             figsize=figsize)

    axes = cycle(axes.flatten())
    weight_markers = {'CS': '^', 'EX': 'o', 'MF': 's', 'AR': 'D'}
    col_aliases = {'avg-strength-rank_related_items_sims_np': 'CS',
                   'avg-strength-rank_strength': 'EX',
                   'avg-strength-rank_biasedmf': 'MF',
                   'avg-strength-rank_target_item_average_rating': 'AR'}

    def calc_avg_strength_per_rank_iter(df=None):
        for rank_col in rank_cols:
            c_aliases = {rank_col: 'rank-pos',
                         'strength': 'avg-strength-{}'.format(rank_col)}
            df_g = (df.groupby(rank_col).strength.mean().reset_index()
                    .rename(columns=c_aliases)
                    .set_index('rank-pos'))
            yield df_g

    legend = True
    for dataset in ('ta', 'yp', 'ba'):
        dataset_id = '{d}_bw_{w}'.format(d=dataset, w=weight)
        df = explanations[dataset_id]
        ax = next(axes)
        rank_cols = [c for c in df.columns if c.startswith('rank_')]

        df_avg_strength_per_rank = pd.concat(
            calc_avg_strength_per_rank_iter(df), axis=1)

        df_avg_strength_per_rank.rename(columns=col_aliases, inplace=True)

        for col in df_avg_strength_per_rank.columns:
            df_avg_strength_per_rank[col].plot(kind='line', ms=markersize,
                                               marker=weight_markers[col],
                                               lw=lw, alpha=0.85,
                                               markeredgewidth=lw,
                                               ax=ax, legend=False)
        # if legend:
        #             ax.legend(scatterpoints=1, ncol=4, numpoints=1, columnspacing=0.5,
        #                       loc='lower right', **legend_kwds)
        #             legend=False

        ax.set_xlabel('Rank $k$', fontsize=xlabel_fontsize)
        ax.set_ylabel('Average Strength', fontsize=xlabel_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=xlabel_fontsize * 0.9)

        ax.axhline(y=0, c='k', ls='-', lw=0.5)
        title = '{}-{}'.format(dataset_names[dataset], weight.upper())
        ax.set_title(title, fontsize=title_fontsize)

        fmt = 'avg-strength-per-rank-{dataset}-{weight}.csv'
        save_path = fmt.format(dataset=dataset, weight=weight)
        df_avg_strength_per_rank.reset_index().to_csv(save_path, index=False)

    # # plt.suptitle('Average Strength per Rank', fontsize='large')
    plt.minorticks_off()

    plt.legend(scatterpoints=1, ncol=4, numpoints=1, columnspacing=0.5,
               loc='upper right', **legend_kwds)
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', thight=True, nbins=8)
    plt.subplots_adjust(
        #         top=0.92,
        wspace=wspace, hspace=0)
    plt.tight_layout()

    # Save
    pad_inches = 0.055
    pdf_path = os.path.join(figures_dir,
                            'avg-strength-per-rank-{w}.pdf'.format(w=weight))
    png_path = os.path.join(figures_dir,
                            'avg-strength-per-rank-{w}.png'.format(w=weight))
    plt.savefig(pdf_path, bbox_inches='tight', pad_inches=pad_inches)
    plt.savefig(png_path, bbox_inches='tight', dpi=700, pad_inches=pad_inches)


def plot_pros_and_cons(df=None,
                       ax=None,
                       bw_min=0,
                       bw_max=0.9,
                       rank_col=None,
                       frameon=False,
                       lw=0.5,
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
                       file_prefix='',
                       markeredgewidth=0.5,
                       tick_length=1.0,
                       tick_width=0.5,
                       out_dir=None,
                       show_xticklabels=True,
                       minor_ylabel: bool = True,
                       major_ylabel: bool = True, legend_kwds=None, savefig_kwds=None):
    if is_compelling:
        cols = ['n_pros_comp', 'n_cons_comp', 'better_avg_comp', 'worse_avg_comp']
    else:
        cols = ['n_pros', 'n_cons', 'better_avg', 'worse_avg']

    rank_col = rank_col or 'rank_target_item_average_rating'
    df[rank_col] = df[rank_col].astype(int)

    grouped_ranks = df.groupby(rank_col).agg({col: np.mean for col in cols})

    params = dict(edgecolor='black', legend=False, lw=lw)
    if is_compelling:
        grouped_ranks.n_cons_comp = -grouped_ranks.n_cons_comp
        ax = grouped_ranks[['n_pros_comp']].plot(kind='bar', ax=ax, color='whitesmoke', rot=rotation, **params)
        # grouped_ranks[['n_pros_comp']].plot(kind='bar', ax=ax, color='whitesmoke', rot=rotation, **params)
        grouped_ranks[['n_cons_comp']].plot(kind='bar', ax=ax, color='dimgray', **params)
        bw_cols = ['better_avg_comp', 'worse_avg_comp']
    else:
        grouped_ranks.n_cons = -grouped_ranks.n_cons
        ax = grouped_ranks[['n_pros']].plot(kind='bar', ax=ax, color='whitesmoke', rot=rotation, **params)
        # grouped_ranks[['n_pros']].plot(kind='bar', ax=ax, color='whitesmoke', rot=rotation, **params)
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
        n_labels = len(ax2.get_yticklabels())
        ax2.set_ylabel([1] * n_labels)
        # ax2.yaxis.label.set_visible(False)

    # create patches for legend
    cons_patch = mpatches.Patch(facecolor='dimgray', linewidth=0.5, edgecolor='black', label='C')
    pros_patch = mpatches.Patch(facecolor='whitesmoke', linewidth=0.5, edgecolor='black', label='P')

    ax2_handles, ax2_labels = ax2.get_legend_handles_labels()

    legend_kwds = legend_kwds or dict(fontsize='x-small', borderpad=0)
    legend = ax.legend(handles=[cons_patch, pros_patch, ax2_handles[1], ax2_handles[0]],
                       scatterpoints=1, ncol=4, numpoints=1, loc='lower left', frameon=frameon,
                       columnspacing=0.5, **legend_kwds
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
        fmt = '{file_prefix}-pros-cons-{file_suffix}'
        filename = fmt.format(file_prefix=file_prefix, file_suffix=file_suffix)
        path = os.path.join(out_dir, filename).lower()
        savefig_kwds = savefig_kwds or dict(bbox_inches='tight', transparent=False)
        plt.savefig('{}.pdf'.format(path), **savefig_kwds)
    return


def plot_dataset_ndcg(df_ndcgs_all, dataset, markersize=5.5,
                      title_fontsize=11, xlabel_fontsize=10,
                      ylim=None, ylabel=None,
                      xlabel=None, ax=None, out_dir=None, title=None,
                      legend_kwds=None, wspace=0.08, rank_cols=None):
    """Plot all NDCGs of a given dataset.

    Args:
        df_ndcgs_all: DataFrame of all NDCGs
        dataset: Short name of the dataset to plot
        out_dir: where to save the figure.
    """
    # Define columns that'll be included in plot.
    if rank_cols is None:
        rank_cols = {'rank_related_items_sims_np': 'CS',
                     'rank_strength': 'EX',
                     'rank_target_item_average_rating': 'AR',
                     'rank_biasedmf': 'MF'}

    plot_cols = ('rank_related_items_sims_np', 'rank_strength',
                 'rank_biasedmf')
    markers = cycle(['D', '^', 'o', 'v', '+', 'v'])
    linestyles = cycle([':', '--', '-', '-.'])
    markerfacecolors = cycle(['r', 'b', 'g', 'y'])

    if not ax:
        fig, ax = plt.subplots()

    for rank_col in plot_cols:
        marker = next(markers)
        linestyle = next(linestyles)
        markerfacecolor = next(markerfacecolors)

        d = dataset.lower()
        assert len(d) > 0
        df = df_ndcgs_all.query('col == @rank_col and dataset == @d').copy()
        assert len(df) > 0

        df.set_index('k').rename(columns={'ndcg': rank_cols[rank_col]}).plot(
            kind='line',
            marker=marker,
            markersize=markersize,
            ax=ax,
            linewidth=1,
            alpha=0.85,
            label=rank_col,
            color='black',
            linestyle=linestyle,
            markerfacecolor=markerfacecolor,
            markeredgewidth=0.5,
            legend=False
        )

    ax.set_xticks([x + 1 for x in range(len(df))])
    ax.locator_params(axis='y', nticks=10)
    ax.tick_params(axis='both', which='major', labelsize=xlabel_fontsize * 0.9)

    if title:
        ax.set_title(title, fontsize=title_fontsize)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
        ax.set_xlabel(xlabel)

    if ylabel:
        ax.set_ylabel(ylabel, labelpad=0, fontsize=xlabel_fontsize)
    else:
        ax.set_ylabel(None)

    if ylim:
        ax.set_ylim(ylim)
    legend_kwds = legend_kwds or dict(fontsize='x-small', borderpad=0)
    plt.legend(scatterpoints=1,
               ncol=4,
               numpoints=1,
               columnspacing=0.5,
               loc='best',
               **legend_kwds)
    return


def plot_ndcgs_all(df_ndcgs_all, col_width_pts, figures_dir,
                   dataset_lookup=None, figsize=None, wspace=0.08):
    """Plot NDCGs for all datasets."""
    if dataset_lookup is None:
        dataset_lookup = {'ba': 'BeerAdvocate',
                          'ta': 'TripAdvisor',
                          'yp': 'Yelp'}

    latexify(usetex=True, scale=1, column_width_pt=col_width_pts)
    figsize = figsize or (points_to_inches(col_width_pts), 3)

    _, axes = plt.subplots(ncols=3, sharey=True, figsize=figsize)
    axes = axes.flatten()

    for dataset, ax in zip(['ta_nw', 'yp_nw', 'ba_nw'], axes):
        legend_kwds = dict(fontsize=10.5, frameon=True, borderpad=0.3)
        plot_dataset_ndcg(df_ndcgs_all=df_ndcgs_all, dataset=dataset,
                          ylim=(0.65, 1),
                          title_fontsize=14,
                          ax=ax,
                          out_dir=figures_dir,
                          xlabel='Rank $k$',
                          ylabel='$nDCG_k$',
                          legend_kwds=legend_kwds,
                          title=dataset_lookup[dataset.split('_')[0]])
        continue
    # plt.set_ylabel('$nDCG_k$')
    plt.tight_layout()
    plt.locator_params(axis='y', nbins=8)
    plt.subplots_adjust(top=0.95, wspace=wspace, hspace=0)

    # Save
    pad_inches = 0.055
    pdf_path = os.path.join(figures_dir, 'ndcgs.pdf')
    png_path = os.path.join(figures_dir, 'ndcgs.png')
    plt.savefig(pdf_path, bbox_inches='tight', pad_inches=pad_inches)
    plt.savefig(png_path, bbox_inches='tight', dpi=300, pad_inches=pad_inches)
    #     break

#     break
