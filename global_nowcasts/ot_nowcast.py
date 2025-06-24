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
import sys
from datetime import datetime, timedelta
from dateutil.rrule import rrule, HOURLY
import pandas as pd
import iris
import pickle
import numpy as np
from pysteps import motion, nowcasts, verification
import itertools
import utils as ut
import plotting

# Get environment variables
HTML_DIR = os.environ['HTML_DIR']
SATDIR = os.environ['SATDIR']
DATADIR = os.environ['DATADIR']
STEPS = int(os.environ['STEPS'])
TIMESTEP = int(os.environ['TIMESTEP'])
CYCLE_TIME = os.environ['CYLC_TASK_CYCLE_POINT']

# SE asia domain latitude/longitude extents and name
LOC_EXTENTS = [67.52, 97.88, 5.56, 37.4]
LOC_NAME = 'wcssp'

# For pickling
P_FNAME = f'{DATADIR}/ndf_pickle'


def main(new_data):
    """
    Extracts satellite data, then makes nowcasts.
    """
    # Only extract data if necessary
    if new_data == 'yes':

        # Get dates and times based on initial cycle time
        first_vdt = datetime.strptime(CYCLE_TIME, '%Y%m%dT%H%MZ')
        vdts = rrule(HOURLY, interval=6, count=3, dtstart=first_vdt)

        # Loop through each dt
        for vdt in vdts:

            # Extract satellite data
            sat_cube = extract_sat_data(vdt)

            # Move to next iteration if satellite data no successfully extracted
            if not sat_cube:
                print('Insufficient satellite data for nowcast')
                continue

            # # Plot satellite data
            # plot_sats(sat_cube_now, f_str, domain=True)
            # plot_sats(sat_cube_verify, f_str, domain=True)

            # Calculate nowcasts and add cube to dictionary
            ncast_cube = run_ncast(sat_cube)

            # Plot nowcasts and save iris cubes
            plot_ncasts(ncast_cube)


def extract_sat_data(vdt, domain=False):
    """
    Extracts satellite files, regrids and processes to state to be used for
    nowcast

    :param vdt: valid date and time of most recent satellite data to use
    :type vdt: datetime.datetime
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
    sat_cubes = iris.cube.CubeList([])

    # Extract satellite data for 3 timesteps before and steps timesteps 
    # after latest satellite time
    for step in range(-2, 1):

        # Get date of satellite file to use
        sat_dt = vdt + timedelta(minutes=TIMESTEP*step)
        sat_dt_str = sat_dt.strftime('%Y%m%d%H%M')
        sat_date_str = sat_dt.strftime('%Y%m%d')

        # Define raw satellite file to extract
        r_sat_fname = (f'{SATDIR}/COMPOSITE/XY/{sat_date_str}/'
                       f'ETXY87_{sat_dt_str}.nc')

        # Extract from MASS if not done already
        if not os.path.exists(r_sat_fname):

            # Extract tar file from MASS if necessary
            if not os.path.exists(f'{SATDIR}/{sat_date_str}.tar'):
                os.system('moo get moose:/adhoc/projects/autosatarchive/'
                          f'COMPOSITE/XY/{sat_date_str}.tar {SATDIR}')

            # Extract file from tar file
            os.system(f'tar -xf {SATDIR}/ETXY87_{sat_date_str}.tar '
                      f'"{r_sat_fname}"') 

        # Filename of processed satellite file to be sought/created
        p_sat_fname = f'{DATADIR}/sat_data/ETXY87_{sat_dt_str}{fname}.nc'

        # Extract satellite data if not already saved
        if not os.path.isfile(p_sat_fname):

            # Load as cube and Regrid onto 2D
            reg_cube = ut.haic_equi_image(r_sat_fname)

            if not reg_cube:
                if step <= 0:
                    return False
                else:
                    continue

            # Intersect to local domain if required
            if domain:
                lon_extent = tuple(LOC_EXTENTS[:2])
                lat_extent = tuple(LOC_EXTENTS[2:])
                reg_cube = reg_cube.intersection(longitude=lon_extent,
                                                 latitude=lat_extent)

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
            iris.save(reg_cube, p_sat_fname)

        # Otherwise, load regridded data and append to list
        else:
            reg_cube = iris.load_cube(p_sat_fname)

        # Append to appropriate cubelist
        sat_cubes.append(reg_cube)

    if len(sat_cubes) == 0:
        print(f'Cannot make nowcast for {vdt} - empty cube')
        return False

    sat_cube = sat_cubes.merge_cube()

    return sat_cube


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
        plot_title = f'Satellite {TITLE} at {date_plt}'
        p_fname = f'{HTML_DIR}/{name}_py_{f_str[:2]}_sat_{date_str}.png'
        s_fname = f'{HTML_DIR}/{name}_py_{f_str[:2]}_sat_shapes_{date_str}.png'

        # Plot satellite data (if plots not already created and new plots not
        # required)
        if new_plots or not os.path.exists(p_fname):
            plotting.plot(sat_slice, plot_title, p_fname, f_str, extents)
        if new_plots or not os.path.exists(s_fname):
            plotting.plot(sat_slice, plot_title, s_fname, f_str, extents,
                          contours=True)


def run_ncast(sat_cube):
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
    oflow_method = motion.get_method('LK')
    motion_field = oflow_method(sats)

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

    # Save nowcast cube
    t_0_vdt = sat_cube.coord('time').units.num2date(sat_time)
    t_0_vdt_str = t_0_vdt.strftime('%Y%m%d%H%M')
    fname = f'{DATADIR}/ncast_files/87_{t_0_vdt_str}.nc'
    iris.save(ncasts, fname)

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
            for thr_1, thr_2 in zip(THRESHOLDS):
                score = fss(t_n_cube.data, t_s_cube.data, thr_1, scale)
                all_scores[scale][thr_2]['leads'].append(lead_time)
                all_scores[scale][thr_2]['scores'].append(score)

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
    
    # To add scores to
    all_scores = {}

    # Loop through nowcast dictionary
    for method, ncast_cube in ncasts.items():

        # Get units from nowcast cube
        units = ncast_cube.coord('time').units

        # Create dictionary for method in main dictionary
        all_scores[method] = {thr: {'leads': [], 'scores': []} 
                              for thr in THRESHOLDS[1]}

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
            for thr_1, thr_2 in zip(THRESHOLDS):

                # Uses scale = 4
                score = fss(t_n_cube.data, t_s_cube.data, thr_1, 4)
                all_scores[method][thr_2]['leads'].append(lead_time)
                all_scores[method][thr_2]['scores'].append(score)

    # Make plot
    fname = f'{HTML_DIR}/verification_{f_str}{fname}_methods_scale_4.png'
    plotting.verification_models_plot(all_scores, fname, f_str)


def verify_pd(ndf, sat_cube, ncast_cubes, f_str, fname=''):

    # Use fractions skill score method
    fss = verification.get_method('FSS')

    # To add scores to
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
            thrs_1, thrs_2 = THRESHOLDS
            for (ind, thr_1), scale in itertools.product(enumerate(thrs_0), 
                                                         SCALES):
                thr_2 = thrs_2[ind]
                score = fss(t_n_cube.data, t_s_cube.data, thr_1, scale)

                # string containing lead, threshold and scale information
                lead_thresh_scale = f'{lead_time} {thr_2} {scale}'

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


def plot_ncasts(ncast_cube, domain=False):
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
        lead_time = (valid_time - f_ref_time).total_seconds() / 3600

        # Dates/times in various formats for titles/fnames
        v_time_str = valid_time.strftime('%Y%m%d%H%M')
        f_time_str = f_ref_time.strftime('%Y%m%d%H%M')
        vf_time_str = f'{f_time_str}_{v_time_str}'
        date_plt = valid_time.strftime('%H:%MZ %a %d/%m/%Y')

        # Define titles/fnames
        plot_title = (f'Overshooting Tops Nowcast\nVT {date_plt}, '
                      f'(T+{lead_time})')
        p_fname = f'{HTML_DIR}/{name}_py_ot_now_{vf_time_str}_{lead_time}.png'
        s_fname = (f'{HTML_DIR}/{name}_py_ot_now_shapes_'
                   f'{vf_time_str}.png')

        # Plot nowcast data
        plotting.plot(ncast, plot_title, p_fname, 'otn', extents)
        # if new_plots or not os.path.exists(s_fname):
        #     plotting.plot(ncast, plot_title, s_fname, f_str, extents,
        #                   contours=True)


if __name__ == "__main__":

    # try:
    new_data = sys.argv[1]
    # except:
    #     print('WARNING! Arguments not set correctly so will exit python '
    #           'script')
    #     exit()


    # Run main script
    main(new_data)