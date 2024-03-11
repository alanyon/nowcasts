"""
Module containing plotting functions
"""
from itertools import chain
import os
import pandas as pd
import iris
import iris.plot as iplt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import cartopy.crs as ccrs


def get_x_y_ticks(lon_min, lon_max, lat_min, lat_max):
    """
    Returns lists of axes ticks and labels. This is normally straightforward,
    but gets complicated if crossing the dateline.

    :param lon_min: minimum longitude value to use for plotting
    :type lon_min: float
    :param lon_max: maximum longitude value to use for plotting
    :type lon_max: float
    :param lat_min: minimum latitude value to use for plotting
    :type lat_min: float
    :param lat_max: maximum latitude value to use for plotting
    :type lat_max: float

    :return: tuple containing x tick locations, labels to use with x ticks,
             ytick locations
    :rtype: list of ints, list of ints, list of ints
    """
    # Get central longitude value and distance in degrees from centre to
    # horizontal edge of desired plot.
    # Get original max_lon value if crossing dateline.
    if lon_max > 180:
        lon_max -= 360
        x_extent = int(((180 - lon_min) + (180 + lon_max)) / 2)
        if lon_min + x_extent > 180:
            central_lon = int(-180 + ((lon_min + x_extent) - 180))
        else:
            central_lon = int(lon_min + x_extent)
    else:
        x_extent = int((lon_max - lon_min) / 2)
        central_lon = int(lon_min + x_extent)

    # Less gridlines for big plots
    if x_extent > 100:
        x_step = 20
    else:
        x_step = 10

    # Define locations of xticks and define xlabels for plotting
    if -180 + x_extent <= central_lon <= 180 - x_extent:
        xlocs = range((central_lon - x_extent),
                      (central_lon + x_extent) + 5, x_step)
    elif central_lon < -180 + x_extent:
        xlocs = chain(
            range((180 - x_extent) + (180 + central_lon), 180 + 5, x_step),
            range(-180, (central_lon + x_extent + 5), x_step))
    elif central_lon > 180 - x_extent:
        xlocs = chain(
            range((180 - x_extent) - (180 - central_lon), 180 + 5, x_step),
            range(-180, (-180 + x_extent) - (180 - central_lon - 5), x_step))

    # Get lists of xlocs and xlabels
    xlocs = [x for x in xlocs]
    xlabels = [x for x in xlocs if x != -180]

    # ylocs more straightforward
    ylocs = np.arange(lat_min, lat_max + 10, x_step)

    return xlocs, xlabels, ylocs


def plot(cube, plot_title, img_fname, h_type, extents, contours=False):
    """
    Creates a plot using data from an iris cube. Saves image, and has an option
    to save the iris cube used in the plotting.

    :param cube: iris cube contain plotting data
    :type cube: iris.cube.Cube
    :param plot_title: plot title
    :type plot_title: str
    :param img_fname: image file name
    :type img_fname: str
    :param contours: idicates if contour plot is required, defaults to False
    :type contours: bool, optional
    :param diffs: idicates if contour comparison is required, defaults to False
    :type diffs: bool, optional
    """
    # Set lats/lons for global nowcast
    lon_min, lon_max, lat_min, lat_max = extents

    # Get x ticks and labels (complicated if crossing dateline)
    xlocs, xlabels, ylocs = get_x_y_ticks(lon_min, lon_max, lat_min, lat_max)

    # Define title for legend/colorbar
    if h_type == 'otn':
        title = 'OTanv rating'
    else:
        title = 'HAIC probability (%)'

    # Calculate aspect for plotting
    lon_width = lon_max - lon_min
    lat_height = lat_max - lat_min
    plot_aspect = lat_height / lon_width

    # Projection used for plotting, centred over central longitude
    central_lon = lon_min + int(lon_width / 2)
    proj = ccrs.PlateCarree(central_longitude=central_lon)

    # Define figure using plot aspect and projection
    fig = plt.figure(figsize=(15, 15 * plot_aspect))
    ax = plt.axes(projection=proj)

    # Plot data, with type of plot depending on key waord arguments
    if contours:
        try:
            if h_type == 'otn':
                levels = (0.05, 15., 20., 300.)
            else:
                levels = (0.10, 0.5, 0.8, 1.1)

            # Create contour plot
            cs = iplt.contourf(cube, levels=levels,
                               colors=('yellow', 'orange', 'red'))

            # Draw a black outline at the lowest contour
            iplt.contour(cube, levels=(levels[0], 100), colors='black',
                         linewidths=0.4)

        except ValueError as err:
            print('No data so contours will not work', err)

    else:

        # Create colormap
        cmap = colors.ListedColormap(['#610062', '#246ac3', '#00bfce',
                                      '#01f335', '#63fe00', '#fefe00',
                                      '#fcd402', '#fd8d00', '#fd1600',
                                      '#af0000'])

        # Convert probabilities to percentages if needed
        if h_type != 'otn':
            perc_cube = cube * 100
        else:
            perc_cube = cube

        # Make percentage of HAIC plot
        if h_type == 'otn':
            vmax = 25.
        else:
            vmax = 100.
        pcm = iplt.pcolormesh(perc_cube, cmap=cmap, vmax=vmax, vmin=0.000001)

    # Set extent of plot
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())

    # Draw gridlines
    try:
        ax.gridlines(draw_labels=False, xlocs=xlocs, ylocs=ylocs,
                     linewidth=0.5)
        grid = ax.gridlines(draw_labels=True, xlocs=xlabels, ylocs=ylocs,
                            linewidth=0.5)
        grid.xlabels_top = False
    except TypeError:
        grid = ax.gridlines(draw_labels=False, xlocs=xlabels, linewidth=0.5)

    # Draw coastlines and background stuff
    ax.coastlines(linewidths=0.2)
    ax.stock_img()

    # Title of plot
    plt.title(plot_title)

    # Plot a legend if contour plot (as long as there is some data in the cube)
    if contours:
        artists, _ = cs.legend_elements()
        labels = ['L', 'M', 'H']
        ax.legend(artists, labels, title=title, loc='upper right')

    # Include colour bar below image (unless contour plot required)
    else:
        pcm.cmap.set_under(alpha=0)
        cbar_ax = fig.add_axes([0.126, -0.05, 0.773, 0.04])
        cbar = fig.colorbar(pcm, cax=cbar_ax, orientation="horizontal")
        cbar.ax.set_title(title)

    print('here 3')
    # Save plot
    plt.savefig(img_fname, dpi=100, bbox_inches='tight')
    plt.close()


def sea_plots(ndf, rt_df, html_dir, vdts, fstr):
    """
    Creates seaborn verification plots.
    """
    # Make image directory if necessary
    first, last = [vdt.strftime('%Y%m%d%H%m') for vdt in [vdts[0], vdts[-1]]]
    img_dir = f'{html_dir}/verification_{first}-{last}'
    if not os.path.exists(img_dir):
        os.system(f'mkdir {img_dir}')

    # Run time means
    run_means = rt_df.groupby(['method']).mean()
    run_medians = rt_df.groupby(['method']).median()

    print('means', run_means)
    print('medians', run_medians)

    # Run time histogram
    sns.set_style('darkgrid')
    runs = sns.histplot(data=rt_df, x='run time (seconds)', y='method', 
                        hue='method', bins=100, legend=False)
    runs.set_xlabel('Processing time (seconds)', fontsize=12)

    # Add mean lines
    colours = {'LK': 'b', 'VET': 'saddlebrown', 'DARTS': 'g', 'proesmans': 'r'}
    for method, colour in colours.items():
        runs.axvline(run_medians.loc[method].values, c=colour, ls='--', 
                     alpha=0.4, lw=0.5)

    # Save and close plot
    plt.savefig(f'{img_dir}/{fstr}_run_times.png')
    plt.close()

    # Calculate mean fss scores
    fss_means = ndf.groupby(['method', 'lead', 'scale', 'threshold']).mean()

    # Line plot for each threshold, showing distribution over scales
    with sns.plotting_context(rc={'legend.fontsize':15, 
                                  'legend.title_fontsize':20}):
        g = sns.relplot(data=fss_means, x='lead', y='fss', col='threshold', 
                        hue='method', style='method', kind='line', col_wrap=2)
    g.set_xlabels('lead time (minutes)', fontsize=20)
    g.set_ylabels('FSS', fontsize=18)
    g.set_titles(size=20)
    plt.savefig(f'{img_dir}/{fstr}_threshold_means_all_scales.png')
    plt.close()

    # Line plots for each threshold and each scale
    for scale in ndf['scale'].unique():
        fss_scale = ndf.loc[ndf['scale'] == scale]
        scale_means = fss_scale.groupby(['method', 'lead', 'threshold']).mean()
        sns.relplot(data=scale_means, x='lead', y='fss', col='threshold', 
                    hue='method', style='method', kind='line', col_wrap=2)
        plt.savefig(f'{img_dir}/{fstr}_threshold_means_scale_{scale}_plot.png')
        plt.close()

        # # Plot histograms
        # for lead in ndf['lead'].unique():
        #     fss_sl = fss_scale.loc[fss_scale['lead'] == lead]

        #     fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        #     for ax, thr in zip(axs.flat, ndf['threshold'].unique()):
        #         fss_slt = fss_sl.loc[fss_sl['threshold'] == thr]
        #         sns.histplot(data=fss_slt, x='fss', y='method', hue='method',
        #                     bins=20, legend=False, ax=ax)
        #         ax.set_title(f'Threshold={thr}')
        #     fig.tight_layout()
        #     fig.savefig(f'{img_dir}/{fstr}_hists_scale_{scale}_lead_{lead}.png')
        #     plt.close()


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # Ignoring masked values and all kinds of edge cases to make a simple
        # example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
