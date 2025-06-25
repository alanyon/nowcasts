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

DATADIR = '/data/users/andre.lanyon/nowcasts/verification'
START = '202506231200'
END = '202506232200'
THRESHOLDS = [20, 40, 60, 80]


def main():
    """
    Collects data from CSV files, verifies against satellite data.

    Args:
        None
    Returns:
        None
    """
    # Get dates and times between start and end
    vdts = rrule(HOURLY, datetime.strptime(START, '%Y%m%d%H%M'),
                 until=datetime.strptime(END, '%Y%m%d%H%M'))
    
    # Empty dataframe to hold all data
    all_df = pd.DataFrame()

    # Loop through each valid time
    for vdt in vdts:

        # Load CSV file for this valid time
        fname = f'{DATADIR}/data/88_{vdt.strftime("%Y%m%d%H%M")}_scores.csv'
        if not os.path.exists(fname):
            print(f'No CSV file found for {vdt}')
            continue
        vdt_df = pd.read_csv(fname)

        # Add to large dataframe
        all_df = pd.concat([all_df, vdt_df])

    # Create figure and axes
    fig, axs = plt.subplots(2, 2, figsize=(18, 10))

    for ind, (ax, thr) in enumerate(zip(axs.flatten(), THRESHOLDS)):

        # Get all data for this threshold
        thr_df = all_df[all_df['Threshold'] == thr]

        # Make scatter plot
        sns.boxplot(data=thr_df, x='Lead', y='FSS', hue='Scale', ax=ax)

        if ind == 0:
            handles, labels = ax.get_legend_handles_labels()
        
        # Supress legend
        ax.get_legend().remove()

        # Add title
        ax.set_title(f'Threshold: {thr}%')

        # Ensure y axes range from 0 to 1
        ax.set_ylim(0, 1)

    # Put legend outside of figure
    fig.tight_layout()
    fig.subplots_adjust(right=0.88)
    fig.legend(handles, labels, loc='upper right', title='Scale')

    # Save and close plot
    fname = f'{DATADIR}/plots/verification_{START}_{END}.png'
    fig.savefig(fname)
    plt.close()




if __name__ == "__main__":
    main()