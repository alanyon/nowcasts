"""
Module containing plotting functions

Functions:
    get_x_y_ticks: Returns x and y ticks and labels for a plot.
    plot: Creates a plot using data from an iris cube and saves it.
    verification_models_plot: Creates a verification for OF methods.
    verification_plot: Creates a verification plot.
    MidpointNormalize: Normalization class for diverging color scales.
"""
from itertools import chain

import cartopy
import cartopy.crs as ccrs
import iris.plot as iplt
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np


def get_x_y_ticks(lon_min, lon_max, lat_min, lat_max):
    """
    Returns lists of axes ticks and labels. This is normally straightforward,
    but gets complicated if crossing the dateline.

    Args:
        lon_min (float): minimum longitude value
        lon_max (float): maximum longitude value
        lat_min (float): minimum latitude value
        lat_max (float): maximum latitude value
    Returns:
        xlocs (list): list of x tick locations
        xlabels (list): list of x tick labels
        ylocs (list): list of y tick locations
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
            range(-180, (central_lon + x_extent + 5), x_step)
        )
    else:
        xlocs = chain(
            range((180 - x_extent) - (180 - central_lon), 180 + 5, x_step),
            range(-180, (-180 + x_extent) - (180 - central_lon - 5), x_step)
        )

    # Get lists of xlocs and xlabels
    xlocs = list(xlocs)
    xlabels = [x for x in xlocs if x != -180]

    # ylocs more straightforward
    ylocs = np.arange(lat_min, lat_max + 10, x_step)

    return xlabels, ylocs


def plot(cube, plot_title, img_fname, h_type, extents, contours=False):
    """
    Creates a plot using data from an iris cube. Saves image, and has an option
    to save the iris cube used in the plotting.

    Args:
        cube (iris.cube.Cube): Cube containing data to plot
        plot_title (str): Title of the plot
        img_fname (str): Filename to save the plot image
        h_type (str): Type of plot
        extents (tuple): Longitude and latitude extents
        contours (bool): Whether to plot contours or a pcolormesh.
                         Default is False (i.e. pcolormesh).
    Returns:
        None
    """
    # Set lats/lons for global nowcast
    lon_min, lon_max, lat_min, lat_max = extents

    # Get x ticks and labels (complicated if crossing dateline)
    xlabels, ylocs = get_x_y_ticks(lon_min, lon_max, lat_min, lat_max)

    # Define title for legend/colorbar
    if h_type == 'ot':
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
            if h_type == 'ot':
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
        if h_type != 'ot':
            perc_cube = cube * 100
        else:
            perc_cube = cube

        # Make percentage of HAIC plot
        if h_type == 'ot':
            vmax = 25.
        else:
            vmax = 100.
        pcm = iplt.pcolormesh(perc_cube, cmap=cmap, vmax=vmax, vmin=0.000001)

    # Set extent of plot
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())

    # Draw gridlines
    try:
        grid = ax.gridlines(draw_labels=True, xlocs=xlabels, ylocs=ylocs,
                            linewidth=0.5)
        grid.xlabels_top = False
    except TypeError:
        grid = ax.gridlines(draw_labels=False, xlocs=xlabels, linewidth=0.5)

    # Draw coastlines and background stuff
    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.COASTLINE,linewidth=0.2)
    ax.add_feature(cartopy.feature.BORDERS, linestyle=':',linewidth=0.3)
    ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
    ax.add_feature(cartopy.feature.RIVERS)

    # Title of plot
    plt.title(plot_title, fontsize=25)

    # Plot a legend if contour plot (as long as there is some data in
    # the cube)
    if contours:
        artists, _ = cs.legend_elements()
        labels = ['L', 'M', 'H']
        ax.legend(artists, labels, title=title, loc='upper right')

    # Include colour bar below image (unless contour plot required)
    else:
        pcm.cmap.set_under(alpha=0)
        cbar_ax = fig.add_axes([0.126, -0.05, 0.773, 0.04])
        cbar = fig.colorbar(pcm, cax=cbar_ax, orientation="horizontal")
        cbar.ax.set_title(title, fontsize=25)
        cbar.ax.tick_params(labelsize=20)

    # Save plot
    plt.savefig(img_fname, dpi=100, bbox_inches='tight')
    plt.close()


def verification_models_plot(scores, fname, f_str):
    """
    Creates a Fractions Skill Score (FSS) verification plot for
    different thresholds and scales, comparing optical flow methods.
    """
    # Create figure and axes
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Labels for legend
    labels = list(scores)

    # Plot for each method
    for m_ind, (method, method_stats) in enumerate(scores.items()):

        # Draw line plot for each lead time
        for ax, (thr, stats) in zip(axs.flat, method_stats.items()):

            # Plot scores as line
            ax.plot(stats['leads'], stats['scores'], 'o-', label=method)

             # Title, axes labels (on first method iteration)
            if m_ind == 0:
                if f_str == 'han':
                    title = f'Threshold={thr}%'
                else:
                    title = f'Threshold={thr}'
                ax.set_title(title)
                ax.set_xlabel('Lead time (minutes)')
                ax.set_ylabel('FSS')

    # Put legend outside of figure
    fig.tight_layout()
    fig.subplots_adjust(right=0.88)
    fig.legend(labels, loc='center right', title='Method')

    # Save and close plot
    fig.savefig(fname)
    plt.close()


def verification_plot(scores, fname, run_time):
    """
    Creates a Fractions Skill Score (FSS) verification plot for
    different thresholds and scales.

    Args:
        scores (dict): Dictionary containing FSS scores
        fname (str): Filename to save the plot image
        run_time (str): Run time of the nowcast model
    Returns:
        None
    """
    # Create figure and axes
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # Labels for legend
    labels = []

    # Make plot for each threshold
    for ind, (ax, (scale, thr_stats)) in enumerate(zip(axs.flat,
                                                       scores.items())):
        # Draw line plot for each lead time
        for thr, stats in thr_stats.items():

            # Only add to labels for one axis
            if ind == 0:
                labels.append(thr)

            # Plot scores as line
            ax.plot(stats['leads'], stats['scores'], 'o-')

        # Title, axes labels
        ax.set_title(f'Scale = {scale}')
        ax.set_xlabel('Lead time (minutes)')
        ax.set_ylabel('FSS')

    # Add title
    title = ('Fractions Skill Score Verification for Nowcast Run Time: '
             f'{run_time}')
    plt.suptitle(title, fontsize=25)

    # Put legend outside of figure
    fig.tight_layout()
    fig.subplots_adjust(right=0.88)
    fig.legend(labels, loc='center right', title='Threshold',
               prop={'size': 18})

    # Save and close plot
    fig.savefig(fname)
    plt.close()


class MidpointNormalize(colors.Normalize):
    """
    Normalize a color scale so that diverging colors are centered around
    a midpoint value. This is useful for visualizing data that has a
    natural midpoint, such as anomalies or differences from a baseline.
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        """
        Initialize the MidpointNormalize with a minimum, maximum, and
        midpoint value.

        Args:
            vmin (float): Minimum value for normalization
            vmax (float): Maximum value for normalization
            midpoint (float): Midpoint value for normalization
            clip (bool): Whether to clip values outside the range
                         [vmin, vmax]
        Returns:
            None
        """
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        """
        Normalize the value based on the midpoint.

        Args:
            value (array-like): Values to normalize
            clip (bool): Whether to clip values outside the range
                         [vmin, vmax]
        Returns:
            np.ma.masked_array: Normalized values, masked where
                                necessary
        """
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
