""""
Script to verify HAIC nowcasts.

Project: High Altitude Ice Crystals - developing object-orientated
         nowcast product
Author: Andre Lanyon
Last updated: 24/06/2025

Functions:
    main: Collects data from CSV files, verifies against satellite data.
"""
import os
import pandas as pd
from dateutil.rrule import rrule, HOURLY
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set plotting style
sns.set_style('darkgrid')

# Constants
DATADIR = '/data/scratch/andre.lanyon/HAIC'
START = '202501310000'
END = '202507312100'
THRESHOLDS = [20, 40, 60, 80]
LOC_NAMES = {'se_asia': 'Southeast Asia', 'africa': 'Africa', 
             'europe': 'Europe', 'south_america': 'South America'}
MIDDAY_TIMES = ['20Z', '13Z', '12Z', '08Z']


def main():
    """
    Collects data from CSV files, verifies against satellite data

    Args:
        None
    Returns:
        None
    """
    # Get dates and times between start and end at 6-hourly intervals
    vdts = rrule(HOURLY, dtstart=datetime.strptime(START, '%Y%m%d%H%M'),
                 until=datetime.strptime(END, '%Y%m%d%H%M'),
                 interval=3)
    print('Total times', len(list(vdts)))
    print('')
    
    # To collect all data in big dataframe
    big_df = pd.DataFrame()

    # Loop through locations
    for loc, midday_time in zip(LOC_NAMES, MIDDAY_TIMES):

        # Empty dataframe to hold all data
        all_df = pd.DataFrame()

        count_missing, count_empty = 0, 0
        # Loop through each valid time
        for vdt in vdts:

            # Load CSV file for this valid time
            fname = (f'{DATADIR}/'
                f'verification/{loc}_{vdt.strftime("%Y%m%d%H%M")}_scores.csv')
            if not os.path.exists(fname):
                # print(f'No CSV file found for {vdt}')
                count_missing += 1
                continue
            vdt_df = pd.read_csv(fname)
            
            # Add hour of nowcast initiation to dataframe
            vdt_df['Run Time'] = vdt.strftime('%HZ')

            # Move on if df is empty
            if vdt_df.empty:
                # print(f'No data for {vdt}')
                count_empty += 1
                continue

            # Add to large dataframe
            all_df = pd.concat([all_df, vdt_df])

        # Reset index
        all_df = all_df.reset_index(drop=True)

        print(LOC_NAMES[loc])
        print('Missing files', count_missing)
        print(f'Percent missing', count_missing/len(list(vdts))*100)
        print('Empty files', count_empty)
        print(f'Percent empty', count_empty/len(list(vdts))*100)
        print('Number of nowcasts generated', 
              len(list(vdts)) - count_missing - count_empty)
        print('Percent of nowcasts with data', 
                (len(list(vdts)) - count_missing 
                 - count_empty)/len(list(vdts))*100)
        print('')

        # Make some plots
        for var in ['Scale', 'Run Time']:
            four_plot(all_df, loc, var)
            summary_plot(all_df, loc, var)
            heatmap_plot(all_df, loc, var, midday_time)

            # Separate heatmaps by threshold
            for thr in THRESHOLDS:
                heatmap_plot(all_df, loc, var, midday_time, threshold=thr)

        # Difference plots
        diff_plot(all_df, loc, 'all')
        for thr in THRESHOLDS:
            diff_plot(all_df, loc, thr)
            for lead in [30, 60, 120]:
                for scale in [4, 16, 64]:
                    diff_plot(all_df, loc, thr, lead=lead, scale=scale)

        # Add location column for big df
        all_df['Domain'] = LOC_NAMES[loc]

        # Add to big df
        big_df = pd.concat([big_df, all_df])

    # Pickle big df
    big_df = big_df.reset_index(drop=True)
    p_name = f'{DATADIR}/verification/all_locs_{START}_{END}_scores.pkl'
    big_df.to_pickle(p_name)

    Unpickle big_df
    big_df = pd.read_pickle(p_name)

    # Make some overall plots
    four_plot(big_df, 'all', 'Scale')
    four_plot(big_df, 'all', 'Domain')
    four_plot(big_df, 'all', 'Run Time')
    for thr in THRESHOLDS:
        for lead in [30, 60, 120]:
            for scale in [4, 16, 64]:
                diff_plot(big_df, 'all', thr, lead=lead, scale=scale)
    diff_sub_plots(big_df)

    print('Finished')


def diff_plot(all_df, loc, thr, lead=None, scale=None):
    """
    Create a difference plot for the specified location, threshold, lead 
    time, and scale.

    Args:
        all_df (pd.DataFrame): Dataframe containing verification scores
        loc (str): Location name for saving plots
        thr (int): Threshold percentage to filter data
        lead (int, optional): Lead time in minutes to filter data
        scale (int, optional): Scale in km to filter data
    Returns:
        None
    """
    # Get data for the specified threshold
    if thr == 'all':
        thr_df = all_df
    else:
        thr_df = all_df[all_df['Threshold'] == thr]

    # Subset data based on lead and scale if specified
    if lead is None and scale is None:
        scale_df = thr_df
    else:
        # Get data for the specified lead time
        lead_df = thr_df[thr_df['Lead'] == lead]
        # Get data at the specified scale
        scale_df = lead_df[lead_df['Scale'] == scale]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot density using hexbin with log scale for counts
    hb = ax.hexbin(
        scale_df['Counts Diff'],
        scale_df['FSS'],
        gridsize=40,
        cmap='Blues',
        mincnt=1,
        linewidths=0.5,
        bins='log'  # Use log scale for counts per bin
    )
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Log(Counts per bin)')

    # Overlay regression line only (no scatter points)
    sns.regplot(
        data=scale_df,
        x='Counts Diff',
        y='FSS',
        ax=ax,
        scatter=False,
        lowess=True,
        line_kws={'color': 'red', 'lw': 2}
    )

    ax.set_xlabel('Counts Diff')
    ax.set_ylabel('FSS')
    ax.set_title(f'Density of FSS vs Counts Diff (thr={thr}, lead={lead}, '
                 f'scale={scale})')

    # Save and close plot
    if lead is None and scale is None:
        lead_str = 'all_leads'
        scale_str = 'all_scales'
    else:
        lead_str = f'lead_{lead}'
        scale_str = f'scale_{scale}'
    fname = (f'{DATADIR}/plots/{loc}_{START}_{END}_{lead_str}_{scale_str}_'
             f'{thr}_diff_plot.png')
    fig.savefig(fname)
    plt.close()


def diff_sub_plots(all_df):
    """
    Create a difference plot for the specified location, threshold, lead 
    time, and scale.

    Args:
        all_df (pd.DataFrame): Dataframe containing verification scores
        loc (str): Location name for saving plots
        thr (int): Threshold percentage to filter data
        lead (int, optional): Lead time in minutes to filter data
        scale (int, optional): Scale in km to filter data
    Returns:
        None
    """
    # Get data for 20% threshold
    thr_df = all_df[all_df['Threshold'] == 20]

    # Get data at 16 scale
    scale_df = thr_df[thr_df['Scale'] == 16]

    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(16, 16))

    # Loop through axs and lead times
    for ax, lead in zip(axs.flatten(), [30, 60, 90, 120]):

        # Get data for the specified lead time
        lead_df = scale_df[scale_df['Lead'] == lead]

        # Plot density using hexbin with log scale for counts
        hb = ax.hexbin(
            lead_df['Counts Diff'],
            lead_df['FSS'],
            gridsize=40,
            cmap='Blues',
            mincnt=1,
            linewidths=0.5  # Use log scale for counts per bin
        )
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('Log(Counts per bin)')

        # Overlay regression line only (no scatter points)
        sns.regplot(
            data=lead_df,
            x='Counts Diff',
            y='FSS',
            ax=ax,
            scatter=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 2}
        )

        # Titles/labels
        ax.set_xlabel('Counts Diff', weight='bold', fontsize=14)
        ax.set_ylabel('FSS', weight='bold', fontsize=14)
        ax.set_title(f'Lead Time: {lead}', weight='bold', fontsize=16)
        cbar = ax.collections[0].colorbar
        cbar.set_label('Counts per bin', fontsize=14, weight='bold')

        # Limit y-axis to 0-1
        ax.set_ylim(0, 1)

    # Save and close plot
    fname = f'{DATADIR}/plots/all_{START}_{END}_16_20_diff_sub_plots.png'
    fig.tight_layout()
    fig.savefig(fname)
    plt.close()


def four_plot(all_df, loc, leg_var):
    """
    Create four plots in the same figure for the given dataframe and 
    location.

    Args:
        df (pd.DataFrame): Dataframe containing verification scores
        loc (str): Location name for saving plots
        leg_var (str): Variable to use for legend (Scale or Run Time)
    Returns:
        None
    """
    # Create figure and axes
    fig, axs = plt.subplots(2, 2, figsize=(18, 10))

    # Create plot for each threshold
    for ind, (ax, thr) in enumerate(zip(axs.flatten(), THRESHOLDS)):

        # Get all data for this threshold
        thr_df = all_df[all_df['Threshold'] == thr]

        # Make plot
        sns.lineplot(data=thr_df, x='Lead', y='FSS', hue=leg_var,
                     ax=ax)

        if ind == 0:
            handles, labels = ax.get_legend_handles_labels()
        
        # Supress legend
        ax.get_legend().remove()

        # Add title
        ax.set_title(f'Threshold: {thr}%', weight='bold', fontsize=16)

        # Ensure y axes range from 0 to 1
        ax.set_ylim(0, 1)

        # Set format of axes labels, rotating x tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_xlabel('Lead time (minutes)', weight='bold', fontsize=14)
        ax.set_ylabel('FSS', weight='bold', fontsize=14)

        # Set x ticks and labels to be every 30 minutes
        lead_ticks = sorted(thr_df['Lead'].unique())
        ax.set_xticks(lead_ticks)

    # Put legend outside of figure
    fig.tight_layout()
    if leg_var =='Scale':
        fig.subplots_adjust(right=0.9)
    else:
        fig.subplots_adjust(right=0.82)
    fig.legend(handles, labels, loc='upper right', title=leg_var, 
                fontsize=18, title_fontproperties={'weight': 'bold', 'size': 20})

    # Save and close plot
    var_str = leg_var.lower().replace(' ', '_')
    fname = f'{DATADIR}/plots/{loc}_{START}_{END}_subplots_{var_str}.png'
    fig.savefig(fname)
    plt.close()


def heatmap_plot(all_df, loc, var, midday_time, threshold=None):
    """
    Create a heatmap plot for the given dataframe and location.

    Args:
        all_df (pd.DataFrame): Dataframe containing verification scores
        loc (str): Location name for saving plots
        var (str): Variable to use for heatmap (Scale or Run Time)
        midday_time (str): Approximate time of midday for the location
    Returns:
        None
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(16, 6))

    # Filter dataframe by threshold if provided
    if threshold is not None:
        all_df = all_df[all_df['Threshold'] == threshold]

    # Example: aggregate to mean FSS per Lead-Scale-Threshold
    heat_df = all_df.groupby(['Lead', var])['FSS'].mean().reset_index()

    # Pivot for heatmap: rows = Lead, cols = Scale, values = mean FSS
    heatmap_data = heat_df.pivot(index=var, columns='Lead', values='FSS')

    # Plot heatmap
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", 
                center=0.5, cbar_kws={'label': 'Mean FSS'}, ax=ax)

    # Rotate y-axis labels
    ax.set_yticklabels(heatmap_data.index, rotation=0)

    # Increase fontsize of axes and colorbar labels
    ax.set_xlabel("Lead Time (minutes)", fontsize=14, weight='bold')
    ax.set_ylabel(var, fontsize=14, weight='bold')
    cbar = ax.collections[0].colorbar
    cbar.set_label('Mean FSS', fontsize=14, weight='bold')

    # Set title with approximate time of midday
    title = f'{LOC_NAMES[loc]}: Approximate Time of Midday: {midday_time}'
    ax.set_title(title, fontsize=16, weight='bold')

    # Save and close plot
    var_str = var.lower().replace(' ', '_')
    if threshold is not None:
        var_str += f'_thr{threshold}'
    fname = f'{DATADIR}/plots/{loc}_{START}_{END}_heatmap_{var_str}.png'
    fig.savefig(fname)
    plt.close()


def summary_plot(all_df, loc, var):
    """
    Create a summary plot for all data in one figure.

    Args:
        all_df (pd.DataFrame): Dataframe containing verification scores
        loc (str): Location name for saving plots
        var (str): Variable to use for legend (Scale or Run Time)
    Returns:
        None
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot all in one
    sns.lineplot(data=all_df, x='Lead', y='FSS', hue=var, ax=ax)

    # Set x ticks and labels to be every 30 minutes
    lead_ticks = sorted(all_df['Lead'].unique())
    ax.set_xticks(lead_ticks)

    # Labels and title
    ax.set_xlabel("Lead Time (minutes)")
    ax.set_ylabel("FSS")

    # Improve legend
    ax.legend(title=var, loc='best', fontsize='small')

    # Save and close plot
    var_str = var.lower().replace(' ', '_')
    fname = f'{DATADIR}/plots/{loc}_{START}_{END}_all_{var_str}.png'
    fig.savefig(fname)
    plt.close()


if __name__ == "__main__":
    main()