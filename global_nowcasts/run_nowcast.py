"""
Script to read in and plot satellite files, then use satellite date to
create and plot nowcasts.

Project: High Altitude Ice Crystals - developing object-orientated
         nowcast product
Author: Andre Lanyon
Last updated: 24/06/2025

Functions:
    main: Extracts satellite data, then makes nowcasts.
    extract_sat_data: Extracts satellite files and regrids.
    plot_ncasts: Plots nowcast data and saves iris cubes.
    plot_sats: Plots satellite data.
    run_ncast: Creates nowcasts from satellite data.
    verify_csv: Verifies nowcast probs against satellite probs.
"""
import os
from datetime import datetime, timedelta

import pandas as pd
import iris
import copy
import numpy as np
import plotting
import utils as ut
from pysteps import motion, nowcasts, verification

# Get environment variables
HTML_DIR = os.environ['HTML_DIR']
SCRATCH_DIR = os.environ['SCRATCH_DIR']
STEPS = int(os.environ['STEPS'])
VDT_STR = os.environ['VDT_STR']

# Dictionary containing HAIC/OT satellite info
SAT_NUM = 88
N_TYPE = 'haic'
TITLE = 'HAIC risk'
# SE asia domain latitude/longitude extents and name
LOC_EXTENTS = [[90, 130, -10, 10], [0, 40, -10, 10], [-10, 30, 40, 60],
               [-80, -40, -10, 10]]
LOC_NAMES = ['se_asia', 'africa', 'europe', 'south_america']
# Scales and thresholds to look over for verification
SCALES = [2, 4, 8, 16, 32, 64]
THRESHOLDS = [[0.2, 0.4, 0.6, 0.8], [20, 40, 60, 80]]

# Opt-in to microsecond precision and saving split attributes
iris.FUTURE.date_microseconds = True
iris.FUTURE.save_split_attrs = True


def main():
    """
    Extracts satellite data, then makes nowcasts.

    Args:
        None
    Returns:
        None
    """
    # Extract satellite data
    sat_cubes_now, sat_cubes_verify = extract_sat_data()

    # Loop through locations
    for loc, loc_extent in zip(LOC_NAMES, LOC_EXTENTS):

        # Move to next iteration if satellite data not extracted
        if not sat_cubes_now[loc]:
            print('Insufficient satellite data for nowcast')
            exit()

        # # Plot satellite data
        # plot_sats(sat_cubes_now, loc, loc_extent)
        # plot_sats(sat_cubes_verify, loc, loc_extent)

        # Run nowcast using Lukas-Kanade optical flow methods
        ncast_cube, counts = run_ncast(sat_cubes_now[loc])

        # Verify nowcasts against satellite imagery
        verify_csv(sat_cubes_verify, loc, ncast_cube, counts, loc_extent)

        # Plot nowcasts and save iris cubes
        # plot_ncasts(ncast_cube, loc, loc_extent)


def extract_sat_data():
    """
    Extracts satellite files, regrids and processes to state to be used
    for nowcast.

    Args:
        None
    Returns:
        sat_cube_now (iris.cube.Cube): Cube with sat data for nowcast
        sat_cube_verify (iris.cube.Cube): Cube with sat data for
                                          verification
    """
    # Convert valid time to datetime object
    vdt = datetime.strptime(VDT_STR, '%Y%m%dT%H%M')

    # Cubelists to add cubes to
    sat_cubes_now = {loc: iris.cube.CubeList([]) for loc in LOC_NAMES}
    sat_cubes_verify = {loc: iris.cube.CubeList([]) for loc in LOC_NAMES}

    # Extract satellite data for 3 timesteps before and steps timesteps
    # after latest satellite time
    for step in range(-2, STEPS + 1):

        # Get date of satellite file to use
        sat_dt = vdt + timedelta(minutes=30*step)
        sat_dt_str = sat_dt.strftime('%Y%m%d%H%M')

        # Define raw satellite file to extract
        r_fname = f'{SCRATCH_DIR}/sat_files/ETXY{SAT_NUM}_{sat_dt_str}.nc'

        # Extract from MASS if necessary
        if not os.path.exists(r_fname):

            # Can't create nowcast without 3 sat files
            if step <= 0:
                print(f'No satellite data for {vdt} - cannot create nowcast')
                sat_cubes_now = {loc: False for loc in LOC_NAMES}
                sat_cubes_verify = {loc: False for loc in LOC_NAMES}
                return ({loc: iris.cube.CubeList([]) for loc in LOC_NAMES},
                        {loc: iris.cube.CubeList([]) for loc in LOC_NAMES})
            
            # Otherwise, move to next iteration
            else:
                continue

        # Filename of processed satellite file to be sought/created
        p_fnames = [f'{SCRATCH_DIR}/sat_data/{loc}_{sat_dt_str}.nc'
                    for loc in LOC_NAMES]

        # Extract satellite data if not already saved
        if any(not os.path.isfile(p_fname) for p_fname in p_fnames):

            # Load as cube and Regrid onto 2D
            reg_cube = ut.haic_equi_image(r_fname)

            if not reg_cube:
                if step <= 0:
                    return ({loc: iris.cube.CubeList([]) for loc in LOC_NAMES},
                            {loc: iris.cube.CubeList([]) for loc in LOC_NAMES})
                continue

            # Filter cube to only include only data for each location
            for loc_extent, p_fname in zip(LOC_EXTENTS, p_fnames):
                loc_reg_cube = copy.deepcopy(reg_cube)
                loc_reg_cube = filter_cube(loc_reg_cube, p_fname, loc_extent)

        # Loop through locations again
        for loc, p_fname in zip(LOC_NAMES, p_fnames):

            # Load regridded data and append to list
            reg_cube = iris.load_cube(p_fname)

            # Append to appropriate cubelist
            if step <= 0:
                sat_cubes_now[loc].append(reg_cube)
            else:
                sat_cubes_verify[loc].append(reg_cube)

    # Loop through locations again to merge cubes
    merged_cubes_now = {}
    merged_cubes_verify = {}
    for loc in LOC_NAMES:

        # Return False if no cubes to create nowcast or verify
        if any([len(sat_cubes_now[loc]) == 0, 
                len(sat_cubes_verify[loc]) == 0]):
            print(f'Cannot make nowcast for {vdt} {loc} - empty cube')
            merged_cubes_now[loc] = False
            merged_cubes_verify[loc] = False
            continue

        # Merge cubes to create single cube for nowcast and verification
        try:
            sat_cube_now = sat_cubes_now[loc].merge_cube()
            sat_cube_verify = sat_cubes_verify[loc].merge_cube()
            merged_cubes_now[loc] = sat_cube_now
            merged_cubes_verify[loc] = sat_cube_verify
        except:
            print('Could not merge cubes', vdt, loc)
            merged_cubes_now[loc] = False
            merged_cubes_verify[loc] = False

    return merged_cubes_now, merged_cubes_verify


def filter_cube(reg_cube, p_fname, loc_extent):
    """
    Filters regridded satellite cube to only include relevant data and 
    saves out.

    Args:
        reg_cube (iris.cube.Cube): Cube with regridded satellite data
        p_fname (str): Filename to save processed data
        loc_extent (list): List of longitude and latitude extents
    Returns:
        reg_cube (iris.cube.Cube): Cube with filtered regridded data
    """
    # Intersect to local domain
    lon_extent = tuple(loc_extent[:2])
    lat_extent = tuple(loc_extent[2:])
    reg_cube = reg_cube.intersection(longitude=lon_extent, latitude=lat_extent)

    # Delete attributes to enable merging
    attrs = reg_cube.attributes.copy()
    for attr in attrs:
        if attr == 'Conventions':
            continue
        del reg_cube.attributes[attr]

    # Set 'empty' gridpoints to 0
    mask = np.ma.getmask(reg_cube.data)
    reg_cube.data[mask] = 0.

    # Change values to zero where high satellite zenith angle
    reg_cube.data[np.where(reg_cube.data == -1.)] = 0.

    # Save regridded satellite cube
    iris.save(reg_cube, p_fname)

    return reg_cube


def plot_ncasts(ncast_cube, loc, loc_extent):
    """
    Plots nowcast data and saves iris cubes.

    Args:
        ncast_cube (iris.cube.Cube): Cube with nowcast data
        new_plots (bool, optional): Indicates whether to override
                                    existing plots, defaults to False
    Returns:
        None
    """
    # Get units from cube
    units = ncast_cube.coord('time').units

    # T+0 time
    f_ref_time_greg = ncast_cube.coord('forecast_reference_time').points[0]
    f_ref_time = units.num2date(f_ref_time_greg)

    # Draw plots for each haic nowcast
    for ncast in ncast_cube.slices(['latitude', 'longitude']):

        # Get valid time and lead time
        valid_time = units.num2date(ncast.coord('time').points[0])
        lead_time = int((valid_time - f_ref_time).total_seconds() / 60)

        # Dates/times in various formats for titles/fnames
        v_time_str = valid_time.strftime('%Y%m%d%H%M')
        f_time_str = f_ref_time.strftime('%Y%m%d%H%M')
        vf_time_str = f'{f_time_str}_{v_time_str}'
        date_plt = valid_time.strftime('%H:%MZ on %d-%m-%Y')

        # Define titles/fnames
        plot_title = (f'Nowcast {TITLE} at {date_plt}, '
                      f'lead time: T+{lead_time}')
        p_fname = f'{HTML_DIR}/{loc}_{N_TYPE}_now_{vf_time_str}.png'
        s_fname = f'{HTML_DIR}/{loc}_{N_TYPE}_now_shapes_{vf_time_str}.png'

        # Plot nowcast data
        plotting.plot(ncast, plot_title, p_fname, N_TYPE, loc_extent)
        plotting.plot(ncast, plot_title, s_fname, N_TYPE, loc_extent,
                      contours=True)


def plot_sats(sat_cubes, loc, loc_extent):
    """
    Plots satellite data.

    Args:
        sat_cube (iris.cube.Cube): Cube with satellite data to plot
        new_plots (bool, optional): Indicates whether to override
                                    existing plots, defaults to False
    Returns:
        None
    """
    # Get time units from cube
    units = sat_cubes[loc].coord('time').units

    # Draw plots for each valid time
    for sat_slice in sat_cubes[loc].slices(['latitude', 'longitude']):

        # Get date from cube for plot title
        date = units.num2date(sat_slice.coord('time').points[0])

        date_str = date.strftime('%Y%m%d%H%M')
        date_plt = date.strftime('%H:%MZ on %d-%m-%Y')

        # Define plot title, and image and cube file names
        plot_title = f'Satellite {TITLE} at {date_plt}'
        p_fname = f'{HTML_DIR}/{loc}_{N_TYPE}_sat_{date_str}.png'
        s_fname = f'{HTML_DIR}/{loc}_{N_TYPE}_sat_shapes_{date_str}.png'

        # Plot satellite data
        plotting.plot(sat_slice, plot_title, p_fname, N_TYPE, loc_extent)
        plotting.plot(sat_slice, plot_title, s_fname, N_TYPE, loc_extent,
                      contours=True)


def run_ncast(sat_cube):
    """
    Creates nowcasts from satellite data.

    Args:
        sat_cube (iris.cube.Cube): Cube with satellite data
    Returns:
        ncasts (iris.cube.Cube): Cube with nowcast data
    """
    # Get relevant satellite data from cube
    sats = sat_cube.data

    # Latest satellite data
    latest_sat = sats[-1]

    # Count occurences of more than each threshold in latest sat data
    counts = {}
    for i, thr in enumerate(THRESHOLDS[0]):
        counts[THRESHOLDS[1][i]] = np.count_nonzero(latest_sat >= thr)

    print('1', counts)

    # Calculate motion field
    oflow_method = motion.get_method('LK')
    motion_field = oflow_method(sats[-3:, :, :])

    # Run nowcast using Lucas-Kanade(LK) optical flow method
    extrapolate = nowcasts.get_method('extrapolation')
    ncast_data = extrapolate(sats[-1], motion_field, STEPS)

    # Define times (hours after 1970) from last satellite time and steps
    sat_time = sat_cube.coord('time').points[-1]
    # Use the same units as the time coordinate for consistency
    time_units = sat_cube.coord('time').units
    ncast_times = [sat_time + step * 0.5 for step in range(1, STEPS + 1)]
    time_coord = iris.coords.DimCoord(ncast_times, standard_name='time',
                                      units=time_units)

    # Get other coordinates from sat cube
    latitude = sat_cube.coord('latitude')
    longitude = sat_cube.coord('longitude')

    # Create nowcast cube
    ncasts = iris.cube.Cube(
        ncast_data,
        dim_coords_and_dims=[(time_coord, 0), (latitude, 1), (longitude, 2)]
    )

    # Add forecast reference time using latest satellite cube
    fr_coord = iris.coords.DimCoord(
        sat_cube.coord('time').points[-1],
        standard_name='forecast_reference_time',
        units=time_units
    )
    ncasts.add_aux_coord(fr_coord)

    return ncasts, counts


def verify_csv(sat_cubes, loc, ncast_cube, counts_t_0, loc_extent):
    """
    Verifies nowcast probabilities against satellite probabilities.

    Args:
        sat_cube (iris.cube.Cube): Cube with satellite data
        ncast_cube (iris.cube.Cube): Cube with nowcast data
    Returns:
        None
    """
    print(sat_cubes[loc])

    # Use fractions skill score method
    fss = verification.get_method('FSS')

    # Get units from nowcast cube
    units = ncast_cube.coord('time').units

    # To collect verification scores in (use percentage in labels for
    # HAIC)
    scores = {'Lead': [], 'Threshold': [], 'Scale': [], 'FSS': [],
              'Counts Diff': []}

    # Slices of cubes to loop over
    sat_slices = sat_cubes[loc].slices(['latitude', 'longitude'])
    now_slices = ncast_cube.slices(['latitude', 'longitude'])

    # Loop through all times in cubes
    for t_s_cube, t_n_cube in zip(sat_slices, now_slices):

        # Get valid time from cubes
        sat_time = t_s_cube.coord('time').points[0]
        now_time = t_n_cube.coord('time').points[0]

        # Count occurences of more than each threshold in sat data
        latest_sat = t_s_cube.data
        counts = {}
        for i, thr in enumerate(THRESHOLDS[0]):
            counts[THRESHOLDS[1][i]] = np.count_nonzero(latest_sat >= thr)

        # Get difference between counts and counts_t_0
        counts_diff = {thr: counts[thr] - counts_t_0[thr]
                       for thr in counts_t_0.keys()}

        # Move to next iteration if times do not match
        if sat_time != now_time:
            continue
        
        # Loop through each threshold
        for scale in SCALES:
            # Get lead time from nowcast cube
            f_ref_time_greg = ncast_cube.coord('forecast_reference_time')
            f_ref_time = units.num2date(f_ref_time_greg.points[0])
            valid_time = units.num2date(now_time)
            lead_time = int((valid_time - f_ref_time).total_seconds() / 60)

            # Calculate score for each scale value and add to scores
            # dictionary
            for thr_1, thr_2 in zip(*THRESHOLDS):
                score = fss(t_n_cube.data, t_s_cube.data, thr_1, scale)
                scores['Lead'].append(lead_time)
                scores['Threshold'].append(thr_2)
                scores['Scale'].append(scale)
                scores['FSS'].append(score)
                scores[f'Counts Diff'].append(counts_diff[thr_2])

    # Save scores to CSV file
    scores_df = pd.DataFrame(scores)
    dt_str = f_ref_time.strftime('%Y%m%d%H%M')
    scores_fname = f'{SCRATCH_DIR}/verification/{loc}_{dt_str}_scores.csv'
    scores_df.to_csv(scores_fname, index=False)


if __name__ == "__main__":
    main()
