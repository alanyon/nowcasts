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
SATDIR = os.environ['SATDIR']
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

    # Loop through files in SATDIR
    for fname in os.listdir(SATDIR):

        # Check if file is older than threshold date
        file_date = datetime.strptime(fname, 'ETXY88_%Y%m%d%H%M.nc')
        if file_date < threshold_date:

            # Delete old file
            os.remove(os.path.join(SATDIR, fname))
            print(f'Deleted old file: {fname}')


if __name__ == "__main__":
    main()