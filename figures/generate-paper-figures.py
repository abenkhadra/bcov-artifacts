import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import sem, t
from scipy import mean
import matplotlib
import seaborn as sns
from math import sqrt
import os

"""
This script generates the figures depicted in our paper: "Efficient Binary-Level Coverage Analysis".
It loads the data from the dataset directory and outputs all figures to the
current directory
"""
data_dir = os.environ['PWD'] + "/../dataset/"
output_directory = os.environ['PWD'] + "/"


def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert (columns in [1, 2])

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': [r'\usepackage{newtxsf} '
                                      r'\usepackage{gensymb} '],
              'axes.labelsize': 9,  # fontsize for x and y labels (was 10)
              'axes.titlesize': 9,
              # 'text.fontsize': 8,  # was 10
              'legend.fontsize': 9,  # was 10
              'xtick.labelsize': 9,
              'ytick.labelsize': 9,
              'text.usetex': True,
              'figure.figsize': [fig_width, fig_height],
              'font.family': 'sans-serif'
              }

    matplotlib.rcParams.update(params)


# adopted from https://gist.github.com/dengemann/6030453
# License: BSD (3-clause)

def ci_within(df, indexvar, withinvars, measvar, confint=0.95,
              copy=True):
    """ Compute CI / SEM correction factor
    Morey 2008, Cousinaueu 2005, Loftus & Masson, 1994
    Also see R-cookbook http://goo.gl/QdwJl
    Note. This functions helps to generate appropriate confidence
    intervals for repeated measure designs.
    Standard confidence intervals are are computed on normalized data
    and a correction factor is applied that prevents insanely small values.
    df : instance of pandas.DataFrame
        The data frame objetct.
    indexvar : str
        The column name of of the identifier variable that representing
        subjects or repeated measures
    withinvars : str | list of str
        The column names of the categorial data identifying random effects
    measvar : str
        The column name of the response measure
    confint : float
        The confidence interval
    copy : bool
        Whether to copy the data frame or not.
    """
    if copy:
        df = df.copy()

    # Apply Cousinaueu's method:
    # compute grand mean
    mean_ = df[measvar].mean()

    # compute subject means
    subj_means = df.groupby(indexvar)[measvar].mean().values
    for subj, smean_ in zip(df[indexvar].unique(), subj_means):
        # center
        df[measvar][df[indexvar] == subj] -= smean_
        # add grand average
        df[measvar][df[indexvar] == subj] += mean_

    def sem(x):
        return x.std() / np.sqrt(len(x))

    def ci(x):
        se = sem(x)
        return se * scipy.stats.t.interval(confint, len(x) - 1)[1]

    aggfuncs = [np.mean, np.std, sem, ci]
    out = df.groupby(withinvars)[measvar].agg(aggfuncs)

    # compute & apply correction factor
    n_within = np.prod([len(df[k].unique()) for k in withinvars],
                       dtype=df[measvar].dtype)
    cf = np.sqrt(n_within / (n_within - 1))
    for k in ['sem', 'std', 'ci']:
        out[k] *= cf

    return out


def setup_jumptab_figure():
    plt.xlim(0.8)
    plt.xticks(np.arange(0.8, 1.01, 0.05))
    ax = plt.gca()
    ax.legend(loc=3, facecolor='0.98', framealpha=1, frameon=True, prop={'size': 9})
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(which='major', axis='x', linestyle='-.', linewidth=0.7, color='black')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()


def viz_jumptab_compiler(jumptab_df, compiler, output_dir):
    df = jumptab_df[jumptab_df['compiler'].str.contains(compiler)]
    ci_bcov_df = ci_within(df, 'build', ['binary'], 'bcov')
    ci_bcov_df.rename(columns={'mean': 'bcov', 'std': 'std-bcov'}, inplace=True)
    ci_bcov_df.drop(ci_bcov_df.columns[[2, 3]], axis=1, inplace=True)
    ci_ida_df = ci_within(df, 'build', ['binary'], 'ida')
    ci_ida_df.rename(columns={'mean': 'ida', 'std': 'std-ida'}, inplace=True)
    ci_ida_df.drop(ci_ida_df.columns[[2, 3]], axis=1, inplace=True)
    merged_df = pd.merge(ci_bcov_df, ci_ida_df, on='binary', how='inner')
    merged_df.reset_index(inplace=True)
    merged_df.sort_values(['binary'], inplace=True)

    merged_df[['bcov', 'ida']].plot(kind='barh', stacked=False, xerr=merged_df[['std-bcov', 'std-ida']].values.T,
                                    color=custom_colors, capsize=2, width=0.7,
                                    error_kw=dict(elinewidth=0.8, ecolor='black', capthick=0.8))
    labels = merged_df['binary'].values.tolist()
    labels = [w.replace('image-magick', 'magick') for w in labels]
    plt.yticks(np.arange(len(labels)), labels, rotation='horizontal')
    setup_jumptab_figure()
    plt.savefig(output_dir + "/jumptab-" + compiler + ".pdf")


def viz_jumptab_build(jumptab_df, compiler, output_dir):
    df = jumptab_df
    ci_bcov_df = ci_within(df, 'binary', ['build'], 'bcov')
    ci_bcov_df.rename(columns={'mean': 'bcov', 'std': 'std-bcov'}, inplace=True)
    ci_bcov_df.drop(ci_bcov_df.columns[[2, 3]], axis=1, inplace=True)
    ci_ida_df = ci_within(df, 'binary', ['build'], 'ida')
    ci_ida_df.rename(columns={'mean': 'ida', 'std': 'std-ida'}, inplace=True)
    ci_ida_df.drop(ci_ida_df.columns[[2, 3]], axis=1, inplace=True)
    merged_df = pd.merge(ci_bcov_df, ci_ida_df, on='build', how='inner')
    merged_df.reset_index(inplace=True)

    merged_df[['bcov', 'ida']].plot(kind='barh', stacked=False, xerr=merged_df[['std-bcov', 'std-ida']].values.T,
                                    color=custom_colors, capsize=1.5,
                                    error_kw=dict(elinewidth=0.7, ecolor='black', capthick=0.7))
    labels = merged_df['build'].values.tolist()
    plt.yticks(np.arange(len(labels)), labels, rotation='horizontal')
    setup_jumptab_figure()
    plt.savefig(output_dir + "/jumptab-build.pdf")


def viz_jumptab_comp(input_file, output_dir):
    jumptab_df = pd.read_csv(input_file)
    jumptab_df = jumptab_df.assign(actual_bcov=jumptab_df['total'] - jumptab_df['ida'])
    jumptab_df = jumptab_df.assign(actual_ida=jumptab_df['total'] - jumptab_df['bcov'])
    jumptab_df = jumptab_df.assign(bcov=jumptab_df['actual_bcov'] / jumptab_df['total'])
    jumptab_df = jumptab_df.assign(ida=jumptab_df['actual_ida'] / jumptab_df['total'])
    viz_jumptab_compiler(jumptab_df, 'clang', output_dir)
    viz_jumptab_compiler(jumptab_df, 'gcc', output_dir)
    viz_jumptab_build(jumptab_df, 'gcc', output_dir)


def viz_objdump_size_comp(input_file, output_dir):
    # compare size of subject binaries to objdump
    size_comp_df = pd.read_csv(input_file)
    size_comp_df.sort_values(by=['ratio'], inplace=True)
    size_comp_df = size_comp_df[['binary', 'ratio']]
    size_comp_df.plot(y='ratio', x='binary', kind='barh', width=0.6, color=custom_colors[0])
    ax = plt.gca()
    ax.legend().remove()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('')
    plt.ylabel('')
    ax.grid(which='major', axis='x', linestyle='-.', linewidth=0.6, color='black')
    custom_xticks = np.arange(0, 90, 10)
    custom_xticks[0] = 1
    custom_xlabels = [str(x) + "x" for x in custom_xticks]
    plt.xticks(custom_xticks)
    ax.set_xticklabels(custom_xlabels)
    ax.tick_params(axis='both', which='major', labelsize=9)
    plt.tight_layout()
    plt.savefig(output_dir + "/objdump-size-comp.pdf")


def viz_perf_overhead(input_file, output_dir):
    # performance overhead of bcov
    df = pd.read_csv(input_file)
    df.drop(df.columns[[4, 6, 7, 8]], axis=1, inplace=True)
    tag_list = ['orig', 'all', 'any']
    df = df[df.apply(lambda x: x['tag'] in tag_list, axis=1)]
    df = df.groupby(df.columns[0:4].tolist(), as_index=False, sort=False).mean()
    df = pd.pivot_table(df, values='elapsed-time', index=['package', 'compiler', 'build'],
                        columns=['tag']).reset_index()
    df = df.assign(any_ratio=df['any'] / df['orig'])
    df = df.assign(all_ratio=df['all'] / df['orig'])
    df = df.assign(comp_build=df['compiler'].astype(str) + '-' + df['build'])
    df.drop(df.columns[[2, 3]], axis=1, inplace=True)

    print(str(df))
    ci_df = ci_within(df, 'comp_build', ['package'], 'any_ratio')
    ci_df.reset_index(inplace=True)
    ci_df.drop(ci_df.columns[[3, 4]], axis=1, inplace=True)
    binary_list = ['ffmpeg', 'magick', 'gas', 'python', 'llc', 'opencv', 'perl', 'xerces']
    ci_df.replace(ci_df['package'].values, binary_list, inplace=True)
    ci_df.sort_values(['package'], inplace=True)
    ci_df[['mean']].plot(kind='barh', stacked=False, width=0.6,
                         xerr=ci_df[['std']].values.T,
                         color=custom_colors[0], capsize=2,
                         error_kw=dict(elinewidth=0.8, ecolor='black', capthick=0.8))
    labels = ci_df['package'].tolist()
    plt.yticks(np.arange(len(labels)), labels, rotation='horizontal')
    ax = plt.gca()
    ax.get_legend().remove()
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(which='major', axis='x', linestyle='-.', linewidth=0.6, color='black')
    plt.xlim(0.99)
    custom_xticks = np.arange(1, 1.5, 0.1)
    custom_xlabels = [("%.1f" % num) + "x" for num in custom_xticks]
    plt.xticks(custom_xticks)
    ax.set_xticklabels(custom_xlabels)
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(output_dir + "/perf-overhead.pdf")
    print(ci_df.to_string())
    print("any-mean=%f" % df[['any_ratio']].mean())
    print("all-mean=%f" % df[['all_ratio']].mean())


def viz_size_overhead(input_file, output_dir):
    # size overhead of bcov
    overhead_df = pd.read_csv(input_file)
    overhead_df = overhead_df.assign(all_image_overhead=overhead_df['all-image-size'] / overhead_df['orig-image-size'])
    overhead_df = overhead_df.assign(any_image_overhead=overhead_df['any-image-size'] / overhead_df['orig-image-size'])
    overhead_df = overhead_df.assign(all_file_overhead=overhead_df['all-file-size'] / overhead_df['orig-file-size'])
    overhead_df = overhead_df.assign(any_file_overhead=overhead_df['any-file-size'] / overhead_df['orig-file-size'])
    overhead_df = overhead_df.assign(compiler_build=overhead_df['compiler'].str.cat(overhead_df['build'], sep="-"))
    ci_df_arr = []
    ci_df_cols = ['all_image_overhead', 'any_image_overhead', 'all_file_overhead', 'any_file_overhead']
    for col in ci_df_cols:
        ci_df_arr.append(ci_within(overhead_df[['binary', 'compiler_build', col]], 'compiler_build', ['binary'], col))
        df = ci_df_arr[-1]
        df.drop(ci_df_arr[-1].columns[[2, 3]], axis=1, inplace=True)
        prefix = col.replace('_overhead', '')
        df.rename(columns={'mean': prefix + '_mean', 'std': prefix + '_std'}, inplace=True)

    any_df = pd.merge(ci_df_arr[1], ci_df_arr[3], on='binary', how='inner')
    any_df.rename(columns={'any_image_mean': 'mem', 'any_file_mean': 'file'}, inplace=True)
    # avg_image_overhead = any_df['mem'].mean()
    # avg_file_overhead = any_df['file'].mean()
    # print('any-test, avg-image=' + str(avg_image_overhead) + ",avg-file=" + str(avg_file_overhead))
    any_df[['mem', 'file']].plot(kind='barh', stacked=False, width=0.7,
                                 xerr=any_df[['any_image_std', 'any_file_std']].values.T,
                                 color=custom_colors, capsize=2,
                                 error_kw=dict(elinewidth=0.8, ecolor='black', capthick=0.8))
    any_df.reset_index(inplace=True)
    any_df.sort_values(['binary'], inplace=True)
    labels = any_df['binary'].values.tolist()
    plt.yticks(np.arange(len(labels)), labels, rotation='horizontal')
    ax = plt.gca()
    ax.legend(loc=3, facecolor='0.98', framealpha=1, frameon=True, prop={'size': 8})
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(which='major', axis='x', linestyle='-.', linewidth=0.5, color='black')
    plt.xlim(0.99)
    custom_xticks = np.arange(1, 1.5, 0.1)
    custom_xlabels = [("%.1f" % num) + "x" for num in custom_xticks]
    plt.xticks(custom_xticks)
    ax.set_xticklabels(custom_xlabels)
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(output_dir + "/any-size-overhead.pdf")


def viz_size_overhead_dist(input_file, output_dir):
    # distribution of size overhead
    df = pd.read_csv(input_file)
    cols = list(df.filter(regex='(package|mode|compiler|build|.*-size)'))
    df = df[cols]
    df = df.assign(total=df['patch-code-size'] + df['patch-data-size'])
    df = df.assign(relocation_code_size=df['patch-code-size'] - df['coverage-update-size'])
    df = df.assign(relocation_ratio=df['relocation_code_size'] / df['total'])
    df = df.assign(coverage_ratio=df['coverage-update-size'] / df['total'])
    df = df.assign(data_ratio=df['patch-data-size'] / df['total'])
    df = df.assign(compiler_binary=df['compiler'].str.cat(df['package'], sep="-"))
    df = df[df['mode'].str.contains('any')]
    df.drop(df.columns[2:9], axis=1, inplace=True)
    df.drop(df.columns[0], axis=1, inplace=True)
    ci_dfs_cols = ['relocation_ratio', 'coverage_ratio', 'data_ratio']
    merge_df = None
    for col in ci_dfs_cols:
        ci_df = ci_within(df[['build', 'compiler_binary', col]], 'compiler_binary', ['build'], col)
        ci_df.drop(ci_df.columns[[2, 3]], axis=1, inplace=True)
        prefix = col.replace('_ratio', '')
        ci_df.rename(columns={'mean': prefix, 'std': prefix + '_std'}, inplace=True)
        if merge_df is None:
            merge_df = ci_df
        else:
            merge_df = pd.merge(merge_df, ci_df, on='build', how='inner')

    merge_df.reset_index(inplace=True)

    merge_df[['relocation', 'coverage', 'data']].plot(kind='barh', stacked=True, width=0.3,
                                                      xerr=merge_df[
                                                          ['relocation_std', 'coverage_std', 'data_std']].values.T,
                                                      color=custom_colors, capsize=2,
                                                      error_kw=dict(elinewidth=0.8, ecolor='black', capthick=0.8))
    print(merge_df.to_string())
    print("rel-mean:" + str(merge_df['relocation'].mean()))
    print("cov-mean:" + str(merge_df['coverage'].mean()))
    print("data-mean:" + str(merge_df['data'].mean()))
    labels = merge_df['build'].values.tolist()
    plt.yticks(np.arange(len(labels)), labels, rotation='horizontal')
    plt.xlim(0.0)
    plt.xticks(np.arange(0.0, 1.2, 0.2))
    ax = plt.gca()
    ax.legend(['rel-code', 'cov-code', 'data'], facecolor='0.98', framealpha=1, frameon=True, prop={'size': 9})
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(which='major', axis='x', linestyle='-.', linewidth=0.7, color='black')
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(output_dir + "/size-overhead-dist.pdf")


def viz_dbi_perf_overhead_results(dbi_file, bcov_file, output_dir):
    # comparison to performance overhead of dbi tools
    bcov_df = pd.read_csv(bcov_file)
    bcov_df = bcov_df[(bcov_df['build'] == 'release') & (bcov_df['tag'] == 'any')]
    bcov_df.drop(bcov_df.columns[6:9], axis=1, inplace=True)
    bcov_df['tag'].replace('any', 'bcov', inplace=True)
    print(bcov_df.head())
    df = pd.read_csv(dbi_file)
    df = pd.concat([df, bcov_df], axis=0)
    df['timestamp'].apply(str)
    df = df[(df['timestamp'].apply(str)).str.contains('202002')]
    df = df[~(df['package'].str.match(pat='(cpython)|(perl)'))]
    df = df[df['build'] == 'release']
    df.drop(df.columns[4], axis=1, inplace=True)
    df = df.groupby(df.columns[0:4].tolist(), as_index=False, sort=False).mean()
    sort_keys = df.columns[0:4].tolist()
    sort_keys.reverse()

    df['package'].replace('FFmpeg', 'ffmpeg', inplace=True)
    df['package'].replace('ImageMagick', 'magick', inplace=True)
    df['package'].replace('binutils', 'gas', inplace=True)
    df['package'].replace('llvm', 'llc', inplace=True)
    df.sort_values(sort_keys, inplace=True)
    df = df.assign(overhead=df['elapsed-time'])
    base = (df[df['tag'] == 'orig'])['overhead'].tolist()
    for tag in df['tag'].unique():
        actual = df.loc[df['tag'] == tag, 'overhead'] / base
        df.loc[df['tag'] == tag, 'overhead'] = actual

    print("mean overhead per tag:" + str(df.groupby(['tag'], as_index=False, sort=False).mean()))
    df = df[['tag', 'package', 'overhead']][~ (df['tag'].str.match("(orig)"))]
    df = df.groupby(['package', 'tag'], as_index=False, sort=False).mean()
    df = df.pivot_table('overhead', ['package', 'tag'])

    df.unstack(level=1).plot(kind='barh', stacked=False, width=0.8, subplots=False,
                             color=custom_colors, capsize=1.5, figsize=[5.5, 3],
                             error_kw=dict(elinewidth=0.7, ecolor='black', capthick=0.7))
    print(str(df))
    # labels = df['package'].unique().tolist()
    # print(str(labels))

    labels = list(df.index.get_level_values(0).unique())
    plt.yticks(np.arange(len(labels)), labels, rotation='horizontal')
    # plt.xscale("log", basex=10)
    plt.xlim(0.5, 35)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(which='major', axis='x', linestyle='-.', linewidth=0.7, color='black')
    custom_xticks = np.arange(0, 35, 5)
    custom_xticks[0] = 1
    custom_xlabels = [str(x) + "x" for x in custom_xticks]
    ax.set_xticks(custom_xticks)
    ax.set_xticklabels(custom_xlabels)

    # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xlabel('')
    plt.ylabel('')
    plt.legend(['bcov', 'DR+drcov', 'DR', 'Pin'], prop={'size': 8}, facecolor='0.98', framealpha=1, frameon=True)
    plt.text(35, 2.05, '*', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir + "/dbi-overhead.pdf")


def setup_cfi_compl_figure(ax):
    plt.xlabel('.text coverage')
    plt.ylabel('binary count')
    plt.yticks(np.arange(0, 3000, 500))
    ax.grid(False)
    ax.grid(which='major', axis='y', linestyle='-.', linewidth=0.5, color='gray')


def viz_cfi_definition_completeness(input_file, output_dir):
    df = pd.read_csv(input_file)
    df2 = df[~ df['path'].str.match("/usr/lib/syslinux|/usr/lib/klibc|/usr/lib/debug/")]
    res = df2['code_sz'].apply(lambda x: int(x, 16))
    df2 = df2[res > 0x1000]
    ax = plt.gca()
    df.hist(column='ratio', bins=20, ax=ax)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.get_figure().gca().set_title("")
    setup_cfi_compl_figure(ax)
    plt.tight_layout()
    plt.savefig(output_dir + "/cfi-hist-all.pdf")
    plt.cla()
    df2.hist(column='ratio', bins=20, ax=ax)
    ax.get_figure().gca().set_title("")
    setup_cfi_compl_figure(ax)
    plt.tight_layout()
    plt.savefig(output_dir + "/cfi-hist-filter.pdf")
    print("total:" + str(len(df)) + ",filter:" + str(len(df2)))


def viz_cfg_based_function_identification(input_file, output_dir):
    df = pd.read_csv(input_file)
    df[['original', 'spedi-full', 'ida-full']].plot(kind='bar', width=0.5, color=custom_colors, fontsize=10,
                                                    figsize=[4.5, 3])
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(which='major', axis='y', linestyle='-.', linewidth=0.6, color='gray')
    ax.get_figure().gca().set_title("")
    plt.legend(['original', 'spedi', 'ida'], prop={'size': 9}, facecolor='0.98', framealpha=1, frameon=True,
               bbox_to_anchor=(0.9, 1))
    ax.set_xticklabels(df['benchmark'].to_list())
    plt.xticks(rotation=60)
    plt.ylabel('Function count')
    plt.tight_layout()
    plt.savefig(output_dir + "/cfg-based-funct.pdf")


def viz_jump_tab_type_dist(df, output_dir):
    df_jumptab_types = df.iloc[:, -8:-1].sum(axis=0)
    ax = plt.gca()
    df_jumptab_types.plot(kind='bar', width=0.5, color=custom_colors[0], figsize=[4, 2.3], legend=None, fontsize=10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.grid(which='major', axis='y', linestyle='-.', linewidth=0.6, color='gray')
    ax.get_figure().gca().set_title("")
    plt.xticks(rotation='horizontal')
    for p in ax.patches:
        ax.annotate(np.round(p.get_height(), decimals=2),
                    (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',
                    xytext=(0, 4), textcoords='offset points', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir + "/jumptab-type-dist.pdf")


def viz_jump_tab_bench_dist(df, output_dir):
    df_bench_dist = df[['binary', 'sum']].groupby(['binary']).sum()
    df_bench_dist.plot(kind='bar', width=0.5, color=custom_colors[0], figsize=[4, 2.3], legend=None, fontsize=10)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.grid(which='major', axis='y', linestyle='-.', linewidth=0.6, color='gray')
    ax.get_figure().gca().set_title("")
    plt.xticks(rotation=60)
    ax.xaxis.set_label_text("")
    for p in ax.patches:
        ax.annotate(np.round(p.get_height(), decimals=2),
                    (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',
                    xytext=(0, 4), textcoords='offset points', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir + "/jumptab-bench-dist.pdf")


def viz_jump_tab_comp_dist(df, output_dir):
    df_comp_dist = df[['compiler', 'sum']].groupby(['compiler']).sum()
    df_comp_dist.plot(kind='bar', width=0.3, color=custom_colors[0], figsize=[3.2, 2], legend=None, fontsize=10)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.grid(which='major', axis='y', linestyle='-.', linewidth=0.6, color='gray')
    ax.get_figure().gca().set_title("")
    ax.xaxis.set_label_text("")
    plt.xticks(rotation='horizontal')
    for p in ax.patches:
        ax.annotate(np.round(p.get_height(), decimals=2),
                    (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',
                    xytext=(0, 4), textcoords='offset points', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir + "/jumptab-comp-dist.pdf")


def viz_jump_tab_build_dist(df, output_dir):
    df_build_dist = df[['build', 'sum']].groupby(['build']).sum()
    df_build_dist.plot(kind='bar', width=0.2, color=custom_colors[0], figsize=[3.2, 2], legend=None, fontsize=10)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.get_figure().gca().set_title("")
    ax.xaxis.set_label_text("")
    plt.xticks(rotation='horizontal')
    for p in ax.patches:
        ax.annotate(np.round(p.get_height(), decimals=2),
                    (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',
                    xytext=(0, 4), textcoords='offset points', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir + "/jumptab-build-dist.pdf")


def viz_jump_table_distributions(input_file, output_dir):
    df = pd.read_csv(input_file)
    binary_name_map = {
        "as-new": "gas",
        "libMagickCore-7.Q16HDRI.so.6.0.0": "magick",
        "libxerces-c-3.2.so": "xerces",
        "libopencv_core.so.4.0.1": "opencv"
    }
    comp_name_map = {
        "clang-8": "clang-8.0",
        "gcc-7": "gcc-7.3"
    }

    df.replace({'binary': binary_name_map}, inplace=True)
    df.replace({'compiler': comp_name_map}, inplace=True)
    viz_jump_tab_type_dist(df, output_dir)
    viz_jump_tab_build_dist(df, output_dir)
    viz_jump_tab_comp_dist(df, output_dir)
    viz_jump_tab_bench_dist(df, output_dir)
    print(df.head())


def viz_probe_patch_package_dist(df, output_dir):
    package_name_map = {
        "binutils": "gas",
        "ImageMagick": "magick",
        "llvm": "llc",
        "cpython": "python",
        "FFmpeg": "ffmpeg"
    }

    df = df.drop(df.columns[[0, 1]], axis=1)
    df_mean = df.groupby(['package'], as_index=False, sort=False).mean()
    df_mean.replace({'package': package_name_map}, inplace=True)
    df_mean_ratios = df_mean.iloc[:, 1:].div(df_mean.sum(axis=1), axis=0).mul(100, axis=0)
    df_mean_ratios.plot(kind='bar', width=0.8, color=custom_colors, figsize=[7.5, 4.2], fontsize=11)
    print(df_mean_ratios)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.get_figure().gca().set_title("")
    ax.xaxis.set_label_text("")
    ax.legend(ncol=5, bbox_to_anchor=(1, 1.1), fontsize=9)
    custom_xlabels = df_mean['package'].tolist()
    ax.set_xticklabels(custom_xlabels)
    ax.grid(which='major', axis='y', linestyle='-.', linewidth=0.6, color='gray')
    plt.xticks(rotation='horizontal')
    plt.tight_layout()
    plt.savefig(output_dir + "/probe-patch-package-dist.pdf")


def viz_probe_patch_comp_dist(df, output_dir):
    comp_name_map = {
        "clang-8": "clang-8.0",
        "gcc-7": "gcc-7.3",
    }
    df = df.drop(df.columns[[1, 2]], axis=1)
    df_mean = df.groupby(['compiler'], as_index=False, sort=False).mean()
    df_mean.replace({'compiler': comp_name_map}, inplace=True)
    df_mean_ratios = df_mean.iloc[:, 1:].div(df_mean.sum(axis=1), axis=0).mul(100, axis=0)
    df_mean_ratios.plot(kind='bar', width=0.7, color=custom_colors, figsize=[6, 2.5], fontsize=10)
    print(df_mean_ratios)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.get_figure().gca().set_title("")
    ax.xaxis.set_label_text("")
    ax.legend(ncol=1, bbox_to_anchor=(1, 1), fontsize=9)
    custom_xlabels = df_mean['compiler'].tolist()
    ax.set_xticklabels(custom_xlabels)
    ax.grid(which='major', axis='y', linestyle='-.', linewidth=0.6, color='gray')
    plt.xticks(rotation='horizontal')
    plt.tight_layout()
    plt.savefig(output_dir + "/probe-patch-comp-dist.pdf")


def viz_probe_patch_build_dist(df, output_dir):
    df = df.drop(df.columns[[0, 2]], axis=1)
    df_mean = df.groupby(['build'], as_index=False, sort=False).mean()
    df_mean_ratios = df_mean.iloc[:, 1:].div(df_mean.sum(axis=1), axis=0).mul(100, axis=0)
    df_mean_ratios.plot(kind='bar', width=0.6, color=custom_colors, figsize=[6, 2.5], fontsize=10)
    print(df_mean_ratios)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.get_figure().gca().set_title("")
    ax.xaxis.set_label_text("")
    ax.legend(ncol=1, bbox_to_anchor=(1, 1), fontsize=9)
    custom_xlabels = df_mean['build'].tolist()
    ax.set_xticklabels(custom_xlabels)
    ax.grid(which='major', axis='y', linestyle='-.', linewidth=0.6, color='gray')
    plt.xticks(rotation='horizontal')
    plt.tight_layout()
    plt.savefig(output_dir + "/probe-patch-build-dist.pdf")


def viz_probe_patch_results(input_file, output_dir):
    df = pd.read_csv(input_file)
    df = df[df['mode'] == 'any-node-jumptab']
    df = df.loc[:, 'compiler':'short-cond']
    df.drop(df.columns[[3, 4, 5, 6]], axis=1, inplace=True)
    viz_probe_patch_package_dist(df, output_dir)
    viz_probe_patch_comp_dist(df, output_dir)
    viz_probe_patch_build_dist(df, output_dir)


def setup_sns():
    sns.set()
    sns.set_style("ticks")
    sns.set_context(rc={'patch.linewidth': 0.0})


custom_colors = [
    '#2c7fb8',
    '#fdae6b',
    '#31a354',
    '#006d2c',
    '#41b6c4',
    '#636363',
    '#41ae76',
    '#addd8e',
    '#08519c'
]

setup_sns()
latexify()

viz_graph_dict = {
    "viz_objdump_size_comp": ['"objdump-size-comparison.csv"', True],
    "viz_perf_overhead": ['"bcov.test.results.csv"', True],
    "viz_size_overhead": ['"bcov-size-overhead.csv"', True],
    "viz_size_overhead_dist": ['"patch.results.final.csv"', True],
    "viz_dbi_perf_overhead_results": ['"dbi.test.results.csv"', '"bcov.test.results.csv"', True],
    "viz_jumptab_comp": ['"jump-table.results.csv"', True],
    "viz_cfi_definition_completeness": ['"cfi.complete.results.csv"', False],
    "viz_cfg_based_function_identification": ['"cfg-based-function-results.csv"', False],
    "viz_jump_table_distributions": ['"bcov-jump-tables-dist.csv"', True],
    "viz_probe_patch_results": ['"patch.results.final.csv"', True]
}

for method, params in viz_graph_dict.items():
    if not params[-1]:
        continue
    params_text = ""
    for p in params[:-1]:
        params_text = params_text + 'data_dir + ' + p + ','
    func_call = method + "(" + params_text + " output_directory)"
    eval(func_call)
    plt.clf()
    fig = plt.figure()
    plt.close(fig)

print("finished successfully\n")
