"""
Script to extract satellite files from MASS for use in HAIC nowcasts.

Project: High Altitude Ice Crystals - developing object-orientated
         nowcast product
Author: Andre Lanyon
Last updated: 24/06/2025

Functions:
    main: Extracts HAIC satellite data from MASS.
"""
import os
from datetime import datetime, timedelta

# Set up environment variables
MASSDIR = os.environ['MASSDIR']
STEPS = int(os.environ['STEPS'])
VDT_STR = os.environ['VDT_STR']
SCRATCH_DIR = os.environ['SCRATCH_DIR']
SAT_NUM = 88


def main():
    """
    Extracts HAIC satellite data.

    Args:
        None
    Returns:
        None
    """
    print(VDT_STR)
    # Delete any empty netCDF files
    for fname in os.listdir(f'{SCRATCH_DIR}/sat_files'):
        file_path = os.path.join(SCRATCH_DIR, 'sat_files', fname)
        if os.path.getsize(file_path) == 0:
            os.remove(file_path)
            print(f'Deleted empty file: {fname}')

    # Convert valid time to datetime object
    vdt = datetime.strptime(VDT_STR, '%Y%m%dT%H%M')

    # Extract satellite data for 3 timesteps before and steps timesteps
    # after latest satellite time
    for step in range(-3, STEPS + 1):

        # Get date of satellite file to use
        sat_dt = vdt + timedelta(minutes=30*step)
        sat_date_str = sat_dt.strftime('%Y%m%d')
        sat_dt_str = sat_dt.strftime('%Y%m%d%H%M')

        # Define raw satellite file to extract
        r_fname = f'{SCRATCH_DIR}/sat_files/ETXY{SAT_NUM}_{sat_dt_str}.nc'

        # Extract from MASS if necessary
        if not os.path.exists(r_fname):

            # Get MASS file name
            mass_fname = (f'{MASSDIR}/{sat_date_str}'
                          f'/ETXY{SAT_NUM}_{sat_dt_str}.nc')

            # Extract from MASS
            try:
                os.system(f'moo get {mass_fname} {r_fname}')                    
            except Exception as e:
                print(f'Error extracting {mass_fname}: {e}')


if __name__ == "__main__":
    main()