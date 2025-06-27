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

# Set plotting style
sns.set_style('darkgrid')

# Constants
DATADIR = '/data/scratch/andre.lanyon/HAIC'
START = '202501310000'
END = '202502031800'
THRESHOLDS = [20, 40, 60, 80]
LOC_NAMES = ['se_asia', 'africa', 'europe', 'south_america']


def main():
    """
    Collects data from CSV files, verifies against satellite data.

    Args:
        None
    Returns:
        None
    """
    # Get dates and times between start and end at 6-hourly intervals
    vdts = rrule(HOURLY, dtstart=datetime.strptime(START, '%Y%m%d%H%M'),
                 until=datetime.strptime(END, '%Y%m%d%H%M'),
                 interval=6)
    
    for loc in LOC_NAMES:

        # Empty dataframe to hold all data
        all_df = pd.DataFrame()

        # Loop through each valid time
        for vdt in vdts:

            # Load CSV file for this valid time
            fname = (f'{DATADIR}/'
                f'verification/{loc}_{vdt.strftime("%Y%m%d%H%M")}_scores.csv')
            if not os.path.exists(fname):
                print(f'No CSV file found for {vdt}')
                continue
            vdt_df = pd.read_csv(fname)

            # Add to large dataframe
            all_df = pd.concat([all_df, vdt_df])
        
        # Reset index
        all_df = all_df.reset_index(drop=True)

        # Create figure and axes
        fig, axs = plt.subplots(2, 2, figsize=(18, 10))

        # Create plot for each threshold
        for ind, (ax, thr) in enumerate(zip(axs.flatten(), THRESHOLDS)):

            # Get all data for this threshold
            thr_df = all_df[all_df['Threshold'] == thr]

            # Make plot
            sns.lineplot(data=thr_df, x='Lead', y='FSS', hue='Scale', 
                         style='Scale', markers=True, dashes=False, ax=ax)

            if ind == 0:
                handles, labels = ax.get_legend_handles_labels()
            
            # Supress legend
            ax.get_legend().remove()

            # Add title
            ax.set_title(f'Threshold: {thr}%', weight='bold', fontsize=16)

            # Ensure y axes range from 0 to 1
            ax.set_ylim(0, 1)

            # Set format of axes labels
            ax.set_xlabel('Lead time (minutes)', weight='bold', fontsize=14)
            ax.set_ylabel('FSS', weight='bold', fontsize=14)

            # Set x ticks and labels to be every 30 minutes
            lead_ticks = sorted(thr_df['Lead'].unique())
            ax.set_xticks(lead_ticks)

        # Put legend outside of figure
        fig.tight_layout()
        fig.subplots_adjust(right=0.88)
        fig.legend(handles, labels, loc='upper right', title='Scale', 
                   fontsize=18, title_fontproperties={'weight': 'bold', 'size': 20})

        # Save and close plot
        fname = f'{DATADIR}/plots/{loc}_{START}_{END}_subplots.png'
        fig.savefig(fname)
        plt.close()

        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot all in one
        sns.scatterplot(data=all_df, x='Lead', y='FSS', hue='Scale', 
                    style='Threshold',
                    ax=ax)

        # Set x ticks and labels to be every 30 minutes
        lead_ticks = sorted(all_df['Lead'].unique())
        ax.set_xticks(lead_ticks)

        # Labels and title
        ax.set_xlabel("Lead Time (minutes)")
        ax.set_ylabel("FSS")

        # Improve legend
        ax.legend(title="Scale / Threshold", loc='best', fontsize='small')

        # Save and close plot
        fname = f'{DATADIR}/plots/{loc}_{START}_{END}_all_plot.png'
        fig.savefig(fname)
        plt.close()


        # Create the figure
        fig, ax = plt.subplots(figsize=(16, 6))

        # Example: aggregate to mean FSS per Lead-Scale-Threshold
        heat_df = all_df.groupby(['Lead', 'Scale'])['FSS'].mean().reset_index()

        # Pivot for heatmap: rows = Lead, cols = Scale, values = mean FSS
        heatmap_data = heat_df.pivot(index='Scale', columns='Lead', values='FSS')

        # Plot heatmap
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", 
                    center=0.5, cbar_kws={'label': 'Mean FSS'}, ax=ax)

        # Rotate y-axis labels
        ax.set_yticklabels(heatmap_data.index, rotation=0)
        
        # Save and close plot
        fname = f'{DATADIR}/plots/{loc}_{START}_{END}__heatmap.png'
        fig.savefig(fname)
        plt.close()

        print('Finished')



if __name__ == "__main__":
    main()