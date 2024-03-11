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
import seaborn as sns
import utils as ut
import plotting

# Get environment variables
HTML_DIR = os.environ['HTML_DIR']
SATDIR = os.environ['SATDIR']
DATADIR = os.environ['DATADIR']
STEPS = int(os.environ['STEPS'])
CYCLE_TIME = os.environ['CYLC_TASK_CYCLE_POINT']

# HAIC/OT constants
# - sat nums: OT 87, HAIC 88
# - titles: OT overshooting tops, HAIC HAIC risk
# - thresholds: OT [[5, 10, 15, 20], [5, 10, 15, 20]] 
#               HAIC [[0.2, 0.4, 0.6, 0.8], [20, 40, 60, 80]]
# - f_strs: OT 'otn', HAIC 'han'
SAT_NUM = 87
TITLE = 'overshooting tops' 
THRESHOLDS = [[5, 10, 15, 20], [5, 10, 15, 20]]
F_STR = 'otn'

# SE asia domain latitude/longitude extents and name
LOC_EXTENTS = [90, 150, -10, 20]
LOC_NAME = 'se_asia'
# Nowcasting methods and indices to start from in satellite cubes
METHODS = {'LK': -3, 'VET': -3, 'DARTS': -10, 'proesmans': -2}
# Scales to look over (in gridpoints)
SCALES = [2, 4, 8, 16, 32, 64]
# For pickling
P_FNAME = f'{DATADIR}/data_pickle'


def main(new_data, domain):
    """
    Extracts satellite data, then makes nowcasts.
    """
    # Get dates and times based on initial cycle time
    first_vdt = datetime.strptime(CYCLE_TIME, '%Y%m%dT%H%MZ')
    vdts = rrule(HOURLY, interval=6, count=124, dtstart=first_vdt)

    # Only extract data if necessary
    if new_data == 'yes':

        # Pandas dataframe to add to
        fss_dict = {'method': [], 'threshold': [], 'lead': [], 'scale': [], 
                    'fss': []}
        run_times = {'method': [], 'run time (seconds)': []}

        # Loop through each dt
        for vdt in vdts:

            # Extract satellite data
            sat_cube_now, sat_cube_verify = extract_sat_data(vdt, domain)

            # Move to next iteration if satellite data not extracted
            if not sat_cube_now:
                print(f'Insufficient satellite data for nowcast ({vdt})')
                continue

            # Plot satellite data
            print('here 1')
            plot_sats(sat_cube_now, domain)
            plot_sats(sat_cube_verify, domain)

            # Run nowcast using 4 different optical flow methods
            ncast_cubes = {}
            for method, start_ind in METHODS.items():

                t1 = datetime.now()

                # Calculate nowcasts and add cube to dictionary
                ncast_cube = run_ncast(sat_cube_now, method, start_ind)
                ncast_cubes[method] = ncast_cube

                t2 = datetime.now()
                time_taken = (t2 - t1).total_seconds()

                # Add to run times dictionary
                run_times['method'].append(method)
                run_times['run time (seconds)'].append(time_taken)

                # Verify nowcasts against satellite imagery
                verify(fss_dict, sat_cube_verify, ncast_cube, method, 
                       fname=f'_{LOC_NAME}')

                # Plot nowcasts and save iris cubes
                plot_ncasts(ncast_cube, method, domain)

        # Convert dictioanry to dataframe
        ndf = pd.DataFrame(fss_dict)
        rt_df = pd.DataFrame(run_times)

        # Pickle data for later use
        file_object = open(P_FNAME, 'wb')
        pickle.dump([ndf, rt_df], file_object)
        file_object.close()

    else:
        with open(P_FNAME, 'rb') as file_object:
            unpickle = pickle.Unpickler(file_object)
            ndf, rt_df = unpickle.load()

    #Make verification plots
    plotting.sea_plots(ndf, rt_df, HTML_DIR, vdts, F_STR)


def extract_sat_data(vdt, domain):
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
    if domain == 'small':
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
        r_sat_fname = f'{SATDIR}/ETXY{SAT_NUM}_{sat_dt_str}.nc'

        # Notify with print statement if satellite file not available
        if not os.path.exists(r_sat_fname):
            print(f'{r_sat_fname} does not exist')

            # Can't create nowcast without 3 sat files prior to valid date
            if step <= 0:
                print(f'Cannot make nowcast for {vdt} - sat missing')
                return False, False

        # Filename of processed satellite file to be sought/created
        p_sat_fname = f'{DATADIR}/sat_data/{SAT_NUM}_{sat_dt_str}{fname}.nc'

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
            if domain == 'small':
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

            # Number of above zero points 
            num_haic = reg_cube.data[reg_cube.data > 0].shape[0]
            num_all = reg_cube.data.shape[0] * reg_cube.data.shape[1]
            frac_haic = num_haic / num_all

            # To filter out satellite images with missing data (not perfect)
            if frac_haic < 0.05:
                print('Skipping as probably missing data', sat_dt_str)
                if step <= 0:
                    return False, False
                else:
                    continue

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


def plot_sats(sat_cube, domain, new_plots=False):
    """
    Plots satellite data.

    :param sat_cube: cube containing satellite data
    :type sat_cube: iris.cube.Cube
    :param new_plots: indicates whether to overide existing plots, defaults to
                      False
    :type new_plots: bool, optional
    :param domain: indicates whether to use local domain, defaults to False 
    :type domain: bool, optional
    """
    # Local domain variables, if needed
    if domain == 'small':
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
        p_fname = f'{HTML_DIR}/{name}_py_{F_STR[:2]}_sat_{date_str}.png'
        s_fname = f'{HTML_DIR}/{name}_py_{F_STR[:2]}_sat_shapes_{date_str}.png'

        # Plot satellite data (if plots not already created and new plots not
        # required)
        print('here 2')
        if new_plots or not os.path.exists(p_fname):
            plotting.plot(sat_slice, plot_title, p_fname, F_STR, extents)
        # if new_plots or not os.path.exists(s_fname):
        #     plotting.plot(sat_slice, plot_title, s_fname, F_STR, extents,
        #                   contours=True)


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


def verify(fss_dict, sat_cube, ncast_cube, method, fname=''):
    """
    Verifies nowcast probabilities against satellite probabilities.

    :param sat_cube: cube containing satellite data
    :type sat_cube: iris.cube.Cube
    :param ncast_cube: cube containing satellite data
    :type ncast_cube: iris.cube.Cube
    :param fname: extra bit to add to fname for local domain, defaults to empty 
                  string
    :type fname: str, optional
    """
    # Use fractions skill score method
    fss = verification.get_method('FSS')
    
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

        # Loop through each threshold
        for scale in SCALES:

            # Calculate score for each scale value and add to fss dictionary
            for thr_1, thr_2 in zip(*THRESHOLDS):
                score = fss(t_n_cube.data, t_s_cube.data, thr_1, scale)
                fss_dict['method'].append(method)
                fss_dict['threshold'].append(thr_2)
                fss_dict['lead'].append(lead_time)
                fss_dict['scale'].append(scale)
                fss_dict['fss'].append(score)

                if (lead_time == 30 and scale == 64 
                        and thr_2 == 20 and score < 0.5):
                    print(f'Low fss - {f_ref_time}')

    return fss_dict


def plot_ncasts(ncast_cube, method, domain, new_plots=False):
    """
    Plots nowcast data and saves iris cubes.

    :param ncast_cube: cube containing nowcast data
    :type ncast_cube: iris.cube.Cube
    :param new_plots: indicates whether to overide existing plots, defaults to
                      False
    :type new_plots: bool, optional
    :param domain: indicates whether to use local domain, defaults to False 
    :type domain: bool, optional

    """
    # Local domain variables, if needed
    if domain == 'small':
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
        plot_title = (f'Nowcast {TITLE} at {date_plt}, '
                      f'lead time: T+{lead_time}')
        p_fname = (f'{HTML_DIR}/{name}_py_{F_STR[:2]}_now_{vf_time_str}_'
                   f'{method}.png')
        s_fname = (f'{HTML_DIR}/{name}_py_{F_STR[:2]}_now_shapes_'
                   f'{vf_time_str}_{method}.png')

        # Plot nowcast data (if plots not already created and new plots not
        # required)
        if new_plots or not os.path.exists(p_fname):
            plotting.plot(ncast, plot_title, p_fname, F_STR, extents)
        # if new_plots or not os.path.exists(s_fname):
        #     plotting.plot(ncast, plot_title, s_fname, F_STR, extents,
        #                   contours=True)


if __name__ == "__main__":

    try:
        new_data = sys.argv[1]
        domain = sys.argv[2]
    except:
        print('WARNING! Arguments not set correctly so will exit python '
              'script')
        exit()

    # Run main script
    main(new_data, domain)

    print('Finished')

