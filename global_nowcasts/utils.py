"""
Module to load in iris cube, regrid to 2D and save new cube
"""
import os
import numpy as np
import iris
from iris.coord_systems import GeogCS
from iris.coords import DimCoord
import iris.analysis as ier

# Get environment variables
try:
    DATADIR = os.environ['DATADIR']
except KeyError as err:
    raise IOError('Environment variable {} not set.'.format(str(err)))

# Some global definitions
RMDI = -32768.0 * 32768.0
SEMIMAJOR_AXIS = 6378169
SEMIMINOR_AXIS = 6356583.8


def empty_equi_cube(spacing):
    """
    Returns an empty equirectangular iris cube

    :param spacing: Latitude and longitude point spacing
    :type spacing: int

    :return: empty iris cube
    :rtype: iris.cube.Cube
    """
    # Define equirectangular crs
    crs = GeogCS(semi_major_axis=SEMIMAJOR_AXIS,
                 semi_minor_axis=SEMIMINOR_AXIS)

    # Define lat/lon grid with center at 0, 0
    lats = np.hstack((np.flipud(-np.arange(0 + spacing, 90.0, spacing)),
                      np.arange(0, 90.0, spacing)))
    lons = np.hstack((np.flipud(-np.arange(0 + spacing, 180.0, spacing)),
                      np.arange(0, 180.0, spacing)))

    # Set up cube
    x_coords = DimCoord(lons, standard_name='longitude', units='degrees_east',
                        coord_system=crs)
    y_coords = DimCoord(lats, standard_name='latitude', units='degrees_north',
                        coord_system=crs)

    # Build the Iris cube
    empty_data = RMDI * np.ones((lats.size, lons.size))
    cube = iris.cube.Cube(empty_data)
    cube.add_dim_coord(y_coords, 0)
    cube.add_dim_coord(x_coords, 1)
    cube.coord('latitude').guess_bounds()
    cube.coord('longitude').guess_bounds()

    return cube


def get_haic_cube(filename):
    """
    Returns an iris cube with the HAIC product

    :param filename: satellite file name
    :type filename: str

    :return: iris cube with the HAIC product
    :rtype: iris.cube.Cube
    """
    # Load cube
    try:
        cube = iris.load_cube(filename)
    except:
        print(f'Error loading {filename}.')
        return None

    # We have to specify the coord system of the auxiliary coords
    crs = GeogCS(semi_major_axis=SEMIMAJOR_AXIS,
                 semi_minor_axis=SEMIMINOR_AXIS)
    cube.coord('latitude').coord_system = crs
    cube.coord('longitude').coord_system = crs

    return cube


def regrid_haic_cube_to_2d(haic_cube, cube_2d):
    """
    Return an iris cube with the HAIC product as a 2D cube

    :param haic_cube: HAIC iris 1D cube loaded from file
    :type haic_cube: iris.cube.Cube
    :param cube_2d: template cube used to provide the 2D crs
    :type cube_2d: iris.cube.Cube

    :return: iris cube with the HAIC product as a 2D cube
    :rtype: iris.cube.Cube
    """
    # Regrid using 1D to 2D point in cell method (requires iris v1.10)
    return haic_cube.regrid(cube_2d, ier.PointInCell())


def haic_equi_image(filename):
    """
    Calls other functions in this module to load in HAIC iris cube, regrid to
    2D and save resulting cube to netCDF file

    :param filename: satellite file name
    :type filename: str
    """
    # Get HAIC dataset as iris cube
    haic_cube = get_haic_cube(filename)

    # Return if no cube was loaded
    if haic_cube is None:
        return None

    # Regrid to 2D
    haic_plot_cube = regrid_haic_cube_to_2d(haic_cube, empty_equi_cube(0.05))

    return haic_plot_cube
