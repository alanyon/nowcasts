"""
Script to read in and plot satellite files, then use satellite date to create
and plot nowcasts. Output can be seen at
http://www-nwp/~alanyon/global_nowcasts.shtml.

Project: High Altitude Ice Crystals - developing object-orientated nowcast
         product
Author: Andre Lanyon
Last updated: 11/1/1/2022
"""
import os
from datetime import datetime, timedelta
from dateutil.rrule import rrule, HOURLY
import pandas as pd
import iris
import pickle
import numpy as np
from pysteps import motion, nowcasts, verification
import itertools
import haic_utils_op as HU
import plotting

# Get environment variables
HTML_DIR = os.environ['HTML_DIR']
SATDIR = os.environ['SATDIR']
DATADIR = os.environ['DATADIR']
STEPS = int(os.environ['STEPS'])
CYCLE_TIME = os.environ['CYLC_TASK_CYCLE_POINT']

# Dictionary containing HAIC/OT satellite info
# SAT_DICT = {'han': [88, 'HAIC risk', [0.2, 0.4, 0.6, 0.8]], 
#             'otn': [87, 'overshooting tops', [5, 10, 15, 20]]}
SAT_DICT = {'han': [88, 'HAIC risk', [0.2, 0.4, 0.6, 0.8]]}
# SAT_DICT = {'otn': [87, 'overshooting tops', [5, 10, 15, 20]]}

# SE asia domain latitude/longitude extents and name
LOC_EXTENTS = [90, 150, -10, 20]
LOC_NAME = 'se_asia'
# Nowcasting methods and indices to start from in satellite cubes
METHODS = {'LK': -3, 'VET': -3, 'DARTS': -10, 'proesmans': -2}
# Scales to look over (in gridpoints)
SCALES = [2, 4, 8, 16, 32, 64]
# For pickling
P_FNAME = f'{DATADIR}/ndf_pickle'


def main():
    """
    Extracts satellite data, then makes nowcasts.
    """
    # Get dates and times based on initial cycle time
    first_vdt = datetime.strptime(CYCLE_TIME, '%Y%m%dT%H%MZ')
    vdts = rrule(HOURLY, interval=6, count=1, dtstart=first_vdt)

    # Pandas dataframe to add to
    ndf = pd.DataFrame({'LK': [], 'VET': [], 'DARTS': [], 'proesmans': [], 
                        'threshold': [], 'lead': [], 'scale': []})

    # Loop through each dt
    for vdt in vdts:

        # Go through same process for HAIC and OT satellite files
        for f_str in SAT_DICT:

            # Extract satellite data
            (sat_cube_now, 
             sat_cube_verify) = extract_satellite_data(vdt, SAT_DICT[f_str][0],
                                                       domain=True)

            # Move to next iteration if satellite data no successfully extracted
            if not sat_cube_now:
                print('Insufficient satellite data for nowcast')
                continue

            # Plot satellite data
            # plot_sats(sat_cube_now, f_str, domain=True)
            # plot_sats(sat_cube_verify, f_str, domain=True)

            # # # Run nowcast using 4 different optical flow methods
            ncast_cubes = {}
            for method, start_ind in METHODS.items():

                t1 = datetime.now()

                # Calculate nowcasts and add cube to dictionary
                ncast_cube = run_ncast(sat_cube_now, method, start_ind)
                ncast_cubes[method] = ncast_cube

                t2 = datetime.now()
                time_taken = (t2 - t1).total_seconds()
                print(f'Time taken for {method} method: {time_taken:.2f} seconds')

                # Verify nowcasts against satellite imagery
                # verify(sat_cube_verify, ncast_cube, f_str, fname=f'_{LOC_NAME}')

                # Plot nowcasts and save iris cubes
                # plot_ncasts(ncast_cube, f_str, domain=True)


            # # Make plots comparing nowcast methods
            # fname_str = f'_{LOC_NAME}'
            # verify_models(sat_cube_verify, ncast_cubes, f_str, fname=fname_str)

            # Verify and add to dataframe
            fname_str = f'_{LOC_NAME}'
            ndf = verify_pd(ndf, sat_cube_verify, ncast_cubes, f_str, 
                            fname=fname_str)

    # Pickle dataframe for later use
    file_object = open(P_FNAME, 'wb')
    pickle.dump(ndf, file_object)
    file_object.close()


def extract_satellite_data(vdt, sat_num, domain=False):
    """
    Extracts satellite files, regrids and processes to state to be used for
    nowcast

    :param vdt: valid date and time of most recent satellite data to use
    :type vdt: datetime.datetime
    :param sat_num: Number for type of satellite data (HAIC or OT)
    :type sat_num: int
    :param domain: indicates whether to use local domain, defaults to False 
    :type domain: bool, optional

    :return: cube containing satellite data
    :return type: iris.cube.Cube
    """
    # For filenames
    if domain:
        fname = f'_{LOC_NAME}'
    else:
        fname = ''

    # Cubelists to add cubes to
    sat_cubes_now = iris.cube.CubeList([])
    sat_cubes_verify = iris.cube.CubeList([])

    # Extract satellite data for 3 timesteps before and steps timesteps after
    # latest satellite time
    for step in range(-10, STEPS + 1):

        # Get date of satellite file to use
        sat_dt = vdt + timedelta(minutes=30*step)
        sat_dt_str = sat_dt.strftime('%Y%m%d%H%M')

        # Define raw satellite file to extract
        r_sat_fname = f'{SATDIR}/ETXY{sat_num}_{sat_dt_str}.nc'

        # Notify with print statement if satellite file not available
        if not os.path.exists(r_sat_fname):
            print(f'{r_sat_fname} does not exist')

            # Can't create nowcast without 3 sat files prior to valid date
            if step <= 0:
                return False, False

        # Filename of processed satellite file to be sought/created
        p_sat_fname = f'{DATADIR}/sat_data/{sat_num}_{sat_dt_str}{fname}.nc'

        # Extract satellite data if not already saved
        if not os.path.isfile(p_sat_fname):

            # Load as cube and Regrid onto 2D
            reg_cube = HU.haic_equi_image(r_sat_fname)

            # Intersect to local domain if required
            if domain:
                lon_extent = tuple(LOC_EXTENTS[:2])
                lat_extent = tuple(LOC_EXTENTS[2:])
                reg_cube = reg_cube.intersection(longitude=lon_extent,
                                                 latitude=lat_extent)

            # Delete attributes to enable merging
            for attr in ['history', 'satellites_processed', 'Conventions']:
                del reg_cube.attributes[attr]

            # Set 'empty' gridpoints to 0
            mask = np.ma.getmask(reg_cube.data)
            reg_cube.data[mask] = 0.

            # Change values to zero where high satellite zenith angle
            reg_cube.data[np.where(reg_cube.data == -1.)] = 0.

            # Save regridded satellite cube
            iris.save(reg_cube, p_sat_fname)

        # Otherwise, load regridded data and append to list
        else:
            reg_cube = iris.load_cube(p_sat_fname)

        # Append to appropriate cubelist
        if step <= 0:
            sat_cubes_now.append(reg_cube)
        else:
            sat_cubes_verify.append(reg_cube)

    # Merge into single cubes
    sat_cube_now = sat_cubes_now.merge_cube()
    sat_cube_verify = sat_cubes_verify.merge_cube()

    return sat_cube_now, sat_cube_verify


def plot_sats(sat_cube, f_str, new_plots=False, domain=False):
    """
    Plots satellite data.

    :param sat_cube: cube containing satellite data
    :type sat_cube: iris.cube.Cube
    :param f_str: String indicating dype of data (HAIC or OT)
    :type f_str: str
    :param new_plots: indicates whether to overide existing plots, defaults to
                      False
    :type new_plots: bool, optional
    :param domain: indicates whether to use local domain, defaults to False 
    :type domain: bool, optional
    """
    # Local domain variables, if needed
    if domain:
        extents = LOC_EXTENTS
        name = LOC_NAME
    else:
        extents = [0, 360, -90, 90]
        name = 'all'

    # Get time units from cube
    units = sat_cube.coord('time').units

    # Draw plots for each valid time
    for sat_slice in sat_cube.slices(['latitude', 'longitude']):

        # Get date from cube for plot title
        date = units.num2date(sat_slice.coord('time').points[0])
        date_str = date.strftime('%Y%m%d%H%M')
        date_plt = date.strftime('%H:%MZ on %d-%m-%Y')

        # Define plot title, and image and cube file names
        plot_title = f'Satellite {SAT_DICT[f_str][1]} at {date_plt}'
        p_fname = f'{HTML_DIR}/{name}_py_{f_str[:2]}_sat_{date_str}.png'
        s_fname = f'{HTML_DIR}/{name}_py_{f_str[:2]}_sat_shapes_{date_str}.png'

        # Plot satellite data (if plots not already created and new plots not
        # required)
        if new_plots or not os.path.exists(p_fname):
            plotting.plot(sat_slice, plot_title, p_fname, f_str, extents)
        if new_plots or not os.path.exists(s_fname):
            plotting.plot(sat_slice, plot_title, s_fname, f_str, extents,
                          contours=True)


def run_ncast(sat_cube, method, start_index):
    """
    Creates nowcasts from satellite data.

    :param sat_cube: cube containing satellite data
    :type sat_cube: iris.cube.Cube

    :return: cube containing nowcast data
    :rtype: iris.cube.Cube
    """
    # Get relevant satellite data from cube
    sats = sat_cube.data

    # Calculate motion field
    oflow_method = motion.get_method(method)
    motion_field = oflow_method(sats[start_index:, :, :])

    # Run nowcast using Lucas-Kanade(LK) optical flow method
    extrapolate = nowcasts.get_method('extrapolation')
    ncast_data = extrapolate(sats[-1], motion_field, STEPS)

    # Define times (hours after 1970) from last satellite time and steps
    sat_time = sat_cube.coord('time').points[-1]
    ncast_times = [sat_time + step * 0.5 for step in range(1, STEPS + 1)]
    time_coord = iris.coords.DimCoord(ncast_times, standard_name='time',
                                      units='hours since epoch')

    # Get other coordinates from sat cube
    latitude = sat_cube.coord('latitude')
    longitude = sat_cube.coord('longitude')

    # Create nowcast cube
    ncasts = iris.cube.Cube(ncast_data, dim_coords_and_dims = [(time_coord, 0),
                                                               (latitude, 1),
                                                               (longitude, 2)])

    # Add forecast reference time using latest satellite cube
    fr_coord = iris.coords.DimCoord(sat_cube.coord('time').points[-1],
                                    'forecast_reference_time',
                                    units='hours since epoch')
    ncasts.add_aux_coord(fr_coord)

    return ncasts


def verify(sat_cube, ncast_cube, f_str, fname=''):
    """
    Verifies nowcast probabilities against satellite probabilities.

    :param sat_cube: cube containing satellite data
    :type sat_cube: iris.cube.Cube
    :param ncast_cube: cube containing satellite data
    :type ncast_cube: iris.cube.Cube
    :param f_str: string indicating dype of data (HAIC or OT)
    :type f_str: str
    :param fname: extra bit to add to fname for local domain, defaults to empty 
                  string
    :type fname: str, optional

    """
    # Use fractions skill score method
    fss = verification.get_method('FSS')

    # Define thresholds
    thrs = SAT_DICT[f_str][2]
    
    # Get units from nowcast cube
    units = ncast_cube.coord('time').units

    # To collect vericication scores in (use percentage in labels for HAIC)
    all_scores = {scale: {thr: {'leads': [], 'scores': []} for thr in thrs} 
                  for scale in SCALES}

    # Slices of cubes to loop over
    sat_slices = sat_cube.slices(['latitude', 'longitude'])
    now_slices = ncast_cube.slices(['latitude', 'longitude'])

    # Loop through all times in cubes
    for t_s_cube, t_n_cube in itertools.product(sat_slices, now_slices):

        # Get valid time from cubes
        sat_time = t_s_cube.coord('time').points[0]
        now_time = t_n_cube.coord('time').points[0]

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

            # Calculate score for each scale value and add to scores dictionary
            for thr in thrs:
                score = fss(t_n_cube.data, t_s_cube.data, thr, scale)
                all_scores[scale][thr]['leads'].append(lead_time)
                all_scores[scale][thr]['scores'].append(score)

    # Make plot
    fname = f'{HTML_DIR}/verification_{f_str}{fname}.png'
    plotting.verification_plot(all_scores, fname, f_str)


def verify_models(sat_cube, ncasts, f_str, fname=''):
    """
    Verifies nowcast probabilities against satellite probabilities.

    :param sat_cube: cube containing satellite data
    :type sat_cube: iris.cube.Cube
    :param ncast_cube: cube containing satellite data
    :type ncast_cube: iris.cube.Cube
    :param f_str: string indicating dype of data (HAIC or OT)
    :type f_str: str
    :param fname: extra bit to add to fname for local domain, defaults to empty 
                  string
    :type fname: str, optional

    """
    # Use fractions skill score method
    fss = verification.get_method('FSS')

    # Define thresholds
    thrs = SAT_DICT[f_str][2]
    
    # To add scores to
    all_scores = {}

    # Loop through nowcast dictionary
    for method, ncast_cube in ncasts.items():

        # Get units from nowcast cube
        units = ncast_cube.coord('time').units

        # Create dictionary for method in main dictionary
        all_scores[method] = {thr: {'leads': [], 'scores': []} for thr in thrs}

        # Slices of cubes to loop over
        sat_slices = sat_cube.slices(['latitude', 'longitude'])
        now_slices = ncast_cube.slices(['latitude', 'longitude'])

        # Loop through all times in cubes
        for t_s_cube, t_n_cube in itertools.product(sat_slices, now_slices):

            # Get valid time from cubes
            sat_time = t_s_cube.coord('time').points[0]
            now_time = t_n_cube.coord('time').points[0]

            # Move to next iteration if times do not match
            if sat_time != now_time:
                continue

            # Get lead time from nowcast cube
            f_ref_time_greg = ncast_cube.coord('forecast_reference_time')
            f_ref_time = units.num2date(f_ref_time_greg.points[0])
            valid_time = units.num2date(now_time)
            lead_time = int((valid_time - f_ref_time).total_seconds() / 60)

            # Calculate score for each threshold and add to scores dictionary
            for thr in thrs:

                # Uses scale = 4
                score = fss(t_n_cube.data, t_s_cube.data, thr, 4)
                all_scores[method][thr]['leads'].append(lead_time)
                all_scores[method][thr]['scores'].append(score)

    # Make plot
    fname = f'{HTML_DIR}/verification_{f_str}{fname}_methods_scale_4.png'
    plotting.verification_models_plot(all_scores, fname, f_str)


def verify_pd(ndf, sat_cube, ncast_cubes, f_str, fname=''):

    # Use fractions skill score method
    fss = verification.get_method('FSS')

    # Define thresholds
    thrs = SAT_DICT[f_str][2]

    scores = {}

    # Verify for all methods
    for method, ncast_cube in ncast_cubes.items():

        # Get units from nowcast cube
        units = ncast_cube.coord('time').units

        # Slices of cubes to loop over
        sat_slices = sat_cube.slices(['latitude', 'longitude'])
        now_slices = ncast_cube.slices(['latitude', 'longitude'])

        # Loop through all times in cubes
        for t_s_cube, t_n_cube in itertools.product(sat_slices, now_slices):

            # Get valid time from cubes
            sat_time = t_s_cube.coord('time').points[0]
            now_time = t_n_cube.coord('time').points[0]

            # Move to next iteration if times do not match
            if sat_time != now_time:
                continue

            # Get lead time from nowcast cube
            f_ref_time_greg = ncast_cube.coord('forecast_reference_time')
            f_ref_time = units.num2date(f_ref_time_greg.points[0])
            valid_time = units.num2date(now_time)
            lead_time = int((valid_time - f_ref_time).total_seconds() / 60)

            # Calculate score for each threshold and add to dataframe
            for thr, scale in itertools.product(thrs, SCALES):
                score = fss(t_n_cube.data, t_s_cube.data, thr, scale)

                # string containing lead, threshold and scale information
                if f_str == 'han':
                    thr = int(thr * 100)
                lead_thresh_scale = f'{lead_time} {thr} {scale}'

                # Add to dictionary
                if lead_thresh_scale not in scores:
                    scores[lead_thresh_scale] = {mtd: None 
                                                 for mtd in ncast_cubes}
                scores[lead_thresh_scale][method] = score

    # Create new rows for dataframe
    s_dict = {'LK': [], 'VET': [], 'DARTS': [], 'proesmans': [], 
              'threshold': [], 'lead': [], 'scale': []}
    for l_t_s, m_scores in scores.items():

        # Get lead time, threshold and scale as integers
        lead, thresh, scale = [int(val_str) for val_str in l_t_s.split()]

        # Add values to dictionary
        s_dict['threshold'].append(thresh)
        s_dict['lead'].append(lead)
        s_dict['scale'].append(scale)
        for method, score in m_scores.items():
            s_dict[method].append(score)

    # Add to dataframe
    ndf = pd.concat([ndf, pd.DataFrame(s_dict)])

    return ndf


def plot_ncasts(ncast_cube, f_str, new_plots=False, domain=False):
    """
    Plots nowcast data and saves iris cubes.

    :param ncast_cube: cube containing nowcast data
    :type ncast_cube: iris.cube.Cube
    :param f_str: String indicating dype of data (HAIC or OT)
    :type f_str: str
    :param new_plots: indicates whether to overide existing plots, defaults to
                      False
    :type new_plots: bool, optional
    :param domain: indicates whether to use local domain, defaults to False 
    :type domain: bool, optional

    """
    # Local domain variables, if needed
    if domain:
        extents = LOC_EXTENTS
        name = LOC_NAME
    else:
        extents = [0, 360, -90, 90]
        name = 'all'

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
        plot_title = (f'Nowcast {SAT_DICT[f_str][1]} at {date_plt}, '
                      f'lead time: T+{lead_time}')
        p_fname = f'{HTML_DIR}/{name}_py_{f_str[:2]}_now_{vf_time_str}.png'
        s_fname = (f'{HTML_DIR}/{name}_py_{f_str[:2]}_now_shapes_'
                   f'{vf_time_str}.png')

        # Plot nowcast data (if plots not already created and new plots not
        # required)
        if new_plots or not os.path.exists(p_fname):
            plotting.plot(ncast, plot_title, p_fname, f_str, extents)
        # if new_plots or not os.path.exists(s_fname):
        #     plotting.plot(ncast, plot_title, s_fname, f_str, extents,
        #                   contours=True)
