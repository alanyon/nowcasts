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
ENS_DIR = os.environ['ENSDIR']
HTML_DIR = os.environ['HTML_DIR']
SATDIR = os.environ['SATDIR']
DATADIR = os.environ['DATADIR']
TIMESTEP = int(os.environ['TIMESTEP'])
START_TIME = os.environ['START_TIME']
END_TIME = datetime.strptime(os.environ['END_TIME'], '%Y%m%d%H%M')

# Dictionary containing HAIC/OT satellite info
TITLE = 'HAIC risk', 
THRESHOLDS = [[0.2, 0.4, 0.6, 0.8], [20, 40, 60, 80]]

# SE asia domain latitude/longitude extents and name
LOC_EXTENTS = [67.52, 97.88, 5.56, 37.4]
LOC_NAME = 'wcssp'
# Scales to look over (in gridpoints)
SCALES = [2, 4, 8, 16, 32, 64]


def main(new_data):
    """
    Extracts satellite data, then makes nowcasts.
    """
    # Only extract data if necessary
    if new_data == 'yes':

        # Get dates and times based on initial cycle time
        first_vdt = datetime.strptime(START_TIME, '%Y%m%dT%H%MZ')
        vdts = rrule(HOURLY, interval=6, count=9, dtstart=first_vdt)

        # Loop through each dt
        for ind, vdt in enumerate(vdts[7:]):

            # Only need timesteps out to end of 28th June
            steps = 97 - (12 * ind)

            # Extract satellite data
            sat_cube_now, sat_cube_verify = extract_sat_data(vdt, steps, 
                                                             domain=True)

            # Move to next iteration if satellite data no successfully extracted
            if not sat_cube_now:
                print('Insufficient satellite data for nowcast')
                continue

            # Plot satellite data
            # plot_sats(sat_cube_now, 'haic', domain=True)
            # plot_sats(sat_cube_verify, 'haic', domain=True)

            # Calculate nowcasts and add cube to dictionary
            ncast_cube = run_ncast(sat_cube_now, steps)

            # Load ensemble forecast from same run time
            vdt_str = vdt.strftime('%Y%m%d%H%M')
            ens_cube = iris.load_cube(f'{ENS_DIR}/{vdt_str}_cube.nc')

            # Separate longitude and latitude extents
            lon_ext = (LOC_EXTENTS[0], LOC_EXTENTS[1])
            lat_ext = (LOC_EXTENTS[2], LOC_EXTENTS[3])

            # Take intersection of cube based on min and max lat and lon 
            # values
            ens_cube = ens_cube.intersection(longitude=lon_ext, 
                                             latitude=lat_ext)

            # Verify nowcasts against satellite imagery
            verify(sat_cube_verify, ncast_cube, ens_cube)

            # Plot nowcasts and save iris cubes
            plot_ncasts(ncast_cube, 'haic', domain=True)


def extract_sat_data(vdt, steps, domain=False):
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
    sat_cubes_now = iris.cube.CubeList([])
    sat_cubes_verify = iris.cube.CubeList([])

    # Extract satellite data for 3 timesteps before and steps timesteps after
    # latest satellite time
    for step in range(-3, steps + 1):

        # Get date of satellite file to use
        sat_dt = vdt + timedelta(minutes=TIMESTEP*step)
        sat_dt_str = sat_dt.strftime('%Y%m%d%H%M')

        # Don't go beyont end time
        if sat_dt > END_TIME:
            continue

        # Define raw satellite file to extract
        r_sat_fname = f'{SATDIR}/haic_{sat_dt_str}.nc'

        # Notify with print statement if satellite file not available
        if not os.path.exists(r_sat_fname):
            print(f'{r_sat_fname} does not exist')

            # Can't create nowcast without 3 sat files prior to valid date
            if step <= 0:
                return False, False

        # Filename of processed satellite file to be sought/created
        p_sat_fname = f'{DATADIR}/sat_data/haic_{sat_dt_str}{fname}.nc'

        # Extract satellite data if not already saved
        if not os.path.isfile(p_sat_fname):

            # Load as cube and Regrid onto 2D
            reg_cube = ut.haic_equi_image(r_sat_fname)

            if not reg_cube:
                if step <= 0:
                    return False, False
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
        if step <= 0:
            sat_cubes_now.append(reg_cube)
            if step == 0:
                sat_cubes_verify.append(reg_cube)
        else:
            sat_cubes_verify.append(reg_cube)

    if len(sat_cubes_now) == 0 or len(sat_cubes_verify) == 0:
        print(f'Cannot make nowcast for {vdt} - empty cube')
        return False, False

    try:
        sat_cube_now = sat_cubes_now.merge_cube()
        sat_cube_verify = sat_cubes_verify.merge_cube()
    except:
        print('Could not merge cubes', vdt)
        return False, False

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

        # Only need hourly satellite images from 28th June
        if not datetime(2022, 6, 28) <= date <= datetime(2022, 6, 29):
            continue
        if date.minute == 30:
            continue

        date_str = date.strftime('%Y%m%d%H%M')
        date_plt = date.strftime('%HZ on %d-%m-%Y')

        # Define plot title, and image and cube file names
        plot_title = f'Satellite at {date_plt}'
        p_fname = f'{HTML_DIR}/{name}_{f_str}_sat_{date_str}.png'

        # Plot satellite data (if plots not already created and new plots not
        # required)
        if new_plots or not os.path.exists(p_fname):
            plotting.plot(sat_slice, plot_title, p_fname, f_str, extents)


def run_ncast(sat_cube, steps):
    """
    Creates nowcasts from satellite data.

    :param sat_cube: cube containing satellite data
    :type sat_cube: iris.cube.Cube

    :return: cube containing nowcast data
    :rtype: iris.cube.Cube
    """
    # Define times (hours after 1970) from last satellite time and steps
    # (but limit by end time)
    sat_time_coord = sat_cube.coord('time')
    sat_time = sat_cube.coord('time').points[-1]
    ncast_times = [sat_time + step * TIMESTEP / 60 
                   for step in range(0, steps + 1)]
    ncast_times = [ndt for ndt in ncast_times 
                   if sat_time_coord.units.num2date(ndt) <= END_TIME]
    time_coord = iris.coords.DimCoord(ncast_times, standard_name='time',
                                      units='hours since epoch')

    # Get relevant satellite data from cube
    sats = sat_cube.data

    # Calculate motion field
    oflow_method = motion.get_method('LK')
    motion_field = oflow_method(sats[-3:, :, :])

    # Run nowcast using Lucas-Kanade(LK) optical flow method
    extrapolate = nowcasts.get_method('extrapolation')
    ncast_data = extrapolate(sats[-1], motion_field, len(ncast_times) - 1)

    # Add in latest satellite frame to include T+0 time
    ncast_data = np.concatenate([sats[-1:], ncast_data])

    # Get other coordinates from sat cube
    latitude = sat_cube.coord('latitude')
    longitude = sat_cube.coord('longitude')

    # Create nowcast cube
    ncasts = iris.cube.Cube(ncast_data, dim_coords_and_dims = [(time_coord, 0),
                                                               (latitude, 1),
                                                               (longitude, 2)])

    # Add forecast reference time using latest satellite cube
    fr_coord = iris.coords.DimCoord(sat_time, 'forecast_reference_time',
                                    units='hours since epoch')
    ncasts.add_aux_coord(fr_coord)

    # Save nowcast cube
    t_0_vdt = sat_time_coord.units.num2date(sat_time)
    t_0_vdt_str = t_0_vdt.strftime('%Y%m%d%H%M')
    fname = f'{DATADIR}/ncast_files/haic_{t_0_vdt_str}.nc'
    iris.save(ncasts, fname)

    return ncasts


def verify(sat_cube, ncast_cube, ens_cube):
    """
    Verifies nowcast probabilities against satellite probabilities.

    :param sat_cube: cube containing satellite data
    :type sat_cube: iris.cube.Cube
    :param ncast_cube: cube containing nowcast data
    :type ncast_cube: iris.cube.Cube
    :param ens_cube: cube containing ensemble data
    :type ens_cube: iris.cube.Cube
    """
    # Use fractions skill score method
    fss = verification.get_method('FSS')
    
    # Get units from nowcast cube
    units = ncast_cube.coord('time').units

    # To add scores to
    all_scores = {'Forecast Type': [], 'Scale': [], 'Threshold': [], 
                  'Valid Time': [], 'FSS Score': []}

    # Get forecast reference time (time of latest satellite image)
    f_ref_time_greg = ncast_cube.coord('forecast_reference_time')
    f_ref_time_greg_ens = ncast_cube.coord('forecast_reference_time')
    assert f_ref_time_greg == f_ref_time_greg_ens, 'Check file dates'
    f_ref_time = units.num2date(f_ref_time_greg.points[0])
    f_ref_time_str = f_ref_time.strftime('%Y%m%d%H%M')
    f_ref_title = f_ref_time.strftime('FSS Verification for 28/06/2022 '
                                      'using %HZ run on %d/%m/%Y')

    # Slices of cubes to loop over
    now_slices = ncast_cube.slices(['latitude', 'longitude'])
    sat_slices = sat_cube.slices(['latitude', 'longitude'])
    ens_slices = ens_cube.slices(['latitude', 'longitude'])

    # Loop through all times in cubes
    for t_s_cube, t_n_cube, t_e_cube in itertools.product(sat_slices, 
                                                          now_slices,
                                                          ens_slices):

        # Get valid time from cubes
        sat_time = t_s_cube.coord('time').points[0]
        now_time = t_n_cube.coord('time').points[0]
        ens_time = t_e_cube.coord('time').points[0]

        # Move to next iteration if times do not match
        if any([sat_time != now_time, now_time != ens_time,  
                sat_time != ens_time]):
            continue

        # Only verify 28th June
        valid_time = units.num2date(sat_time)
        if any([valid_time < datetime(2022, 6, 28),
                valid_time > datetime(2022, 6, 29)]):
            continue

        # Get lead time from ensemble cube
        lead_time = int(t_e_cube.coord('forecast_period').points[0])
        time_and_lead = f'{valid_time.hour:02d}Z (T+{lead_time})'

        # Regrid ensemble cube to be the same as satellite/nowcast
        t_e_cube = t_e_cube.regrid(t_s_cube, iris.analysis.Linear())

        # Regridding creates large areas of low values so make these 0
        t_e_cube.data[t_e_cube.data < 0.05] = 0

        # Make plot of ensemble at regridded resolution
        vt_str = valid_time.strftime('%Y-%m-%d %HZ')
        vt_str_2 = valid_time.strftime('%Y%m%d%H%M')
        plot_title = f'Ensemble at: {vt_str}, lead time: T+{lead_time}'
        fname = (f'{HTML_DIR}/{LOC_NAME}_haic_ens_{f_ref_time_str}_'
                 f'{vt_str_2}.png')
        plotting.plot(t_e_cube, plot_title, fname, 'ens', LOC_EXTENTS)

        # Loop through each threshold
        for scale in SCALES:

            # Calculate score for each scale value and add to scores dictionary
            for thr_1, thr_2 in zip(*THRESHOLDS):
                score_now = fss(t_n_cube.data, t_s_cube.data, thr_1, scale)
                score_ens = fss(t_e_cube.data, t_s_cube.data, thr_1, scale)
                all_scores['Forecast Type']. append('Nowcast')
                all_scores['Scale']. append(scale)
                all_scores['Threshold'].append(thr_2)
                all_scores['Valid Time'].append(time_and_lead)
                all_scores['FSS Score'].append(score_now)
                all_scores['Forecast Type']. append('Ensemble')
                all_scores['Scale']. append(scale)
                all_scores['Threshold'].append(thr_2)
                all_scores['Valid Time'].append(time_and_lead)
                all_scores['FSS Score'].append(score_ens)

    # Convert scores dictionary to dataframe
    scores_df = pd.DataFrame(all_scores)

    # Pickle scores for later use
    file_object = open(f'{ENS_DIR}/{f_ref_time_str}_scores', 'wb')
    pickle.dump(scores_df, file_object)
    file_object.close()

    # Make plot
    fname = f'{HTML_DIR}/verification_{f_ref_time_str}.png'
    plotting.verification_wcssp(scores_df, fname, f_ref_title)


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

        # Only need hourly nowcast images from 28th June
        if not datetime(2022, 6, 28) <= valid_time <= datetime(2022, 6, 29):
            continue
        if valid_time.minute == 30:
            continue

        lead_time = int((valid_time - f_ref_time).total_seconds() / 60 / 60)

        # Dates/times in various formats for titles/fnames
        v_time_str = valid_time.strftime('%Y%m%d%H%M')
        f_time_str = f_ref_time.strftime('%Y%m%d%H%M')
        vf_time_str = f'{f_time_str}_{v_time_str}'
        date_plt = valid_time.strftime('%HZ on %d-%m-%Y')

        # Define titles/fnames
        plot_title = (f'Nowcast at {date_plt}, lead time: T+{lead_time}')
        p_fname = f'{HTML_DIR}/{name}_{f_str}_now_{vf_time_str}.png'

        # Plot nowcast data (if plots not already created and new plots not
        # required)
        if new_plots or not os.path.exists(p_fname):
            plotting.plot(ncast, plot_title, p_fname, f_str, extents)


if __name__ == "__main__":

    # try:
    new_data = sys.argv[1]
    # except:
    #     print('WARNING! Arguments not set correctly so will exit python '
    #           'script')
    #     exit()


    # Run main script
    main(new_data)