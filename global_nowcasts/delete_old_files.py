"""
Script to delete old files.

Project: High Altitude Ice Crystals - developing object-orientated
         nowcast product
Author: Andre Lanyon
Last updated: 24/06/2025

Functions:
    main: Collects data from CSV files, verifies against satellite data.
"""
import os
from datetime import datetime, timedelta

# Set up environment variables
SCRATCH_DIR = os.environ['SCRATCH_DIR']
VDT_STR = os.environ['VDT_STR']


def main():
    """
    Collects data from CSV files, verifies against satellite data.

    Args:
        None
    Returns:
        None
    """
    # Convert valid time to datetime object
    vdt = datetime.strptime(VDT_STR, '%Y%m%dT%H%M')

    # Define the threshold date (1 day before the valid time)
    threshold_date = vdt - timedelta(days=1)

    # Loop through files in three data directories
    for d_dir in ['sat_files', 'ncast_files', 'sat_data']:

        for fname in os.listdir(f'{SCRATCH_DIR}/{d_dir}'):

            # Define directory path
            dir_path = os.path.join(SCRATCH_DIR, d_dir)

            # Check if directory exists
            if not os.path.exists(dir_path):
                print(f'Directory {dir_path} does not exist.')
                continue

            # Loop through files in the directory
            for fname in os.listdir(dir_path):

                # Check if file is older than threshold date
                file_date = datetime.strptime(fname, '%Y%m%d%H%M.nc')
                if file_date < threshold_date:

                    # Delete old file
                    os.remove(os.path.join(dir_path, fname))
                    print(f'Deleted old file: {fname}')


if __name__ == "__main__":
    main()