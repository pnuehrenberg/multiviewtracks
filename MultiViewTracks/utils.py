import _pickle
import cv2
import struct
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd

def triangulate_point(pts_2d, views):
    '''Triangulate points from multiple views using either OpenCV.triangulatePoints or DLT.

    See https://github.com/opencv/opencv_contrib/blob/master/modules/sfm/src/triangulation.cpp for reference.
    '''

    assert len(pts_2d) == len(views), 'Number of 2d points and views did not match'
    n_views = len(views)
    if n_views > 2:
        design = np.zeros((3 * n_views, 4 + n_views))
        for i in range(n_views):
            for jj in range(3):
                for ii in range(4):
                    design[3 * i + jj, ii] = -views[i][jj, ii]
            design[3 * i + 0, 4 + i] = pts_2d[i][0]
            design[3 * i + 1, 4 + i] = pts_2d[i][1]
            design[3 * i + 2, 4 + i] = 1
        u, s, vh = svd(design, full_matrices=False)
        pt_4d = vh[-1, :4]
    else:
        pt_4d = cv2.triangulatePoints(views[0], views[1], np.array(pts_2d[0]).reshape(2, 1), np.array(pts_2d[1]).reshape(2, 1))
    pt_3d = cv2.convertPointsFromHomogeneous(pt_4d.reshape(1, 4)).ravel()
    return pt_3d

def save(dump, file_name):
    '''Save to a .pkl file

    Parameters
    ----------
    dump : object
        Python object to save
    file_name : str
        File path of saved object

    Returns
    -------
    bool
        Successful save?
    '''

    with open(file_name, 'wb') as fid:
        _pickle.dump(dump, fid)
    return True

def load(file_name):
    '''Loads a python object from a .pkl file

    Parameters
    ----------
    file_name : str
        File path of saved object

    Returns
    -------
    object
        Loaded python object
    '''

    with open(file_name, 'rb') as fid:
        dump = _pickle.load(fid)
    return dump

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character = '<'):
    '''Read next bytes of a COLMAP .bin file.

    See https://github.com/colmap/colmap/blob/dev/scripts/python/read_model.py for reference.
    '''

    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def plot_tracks_2d(tracks, ax=None, figsize=(30, 30), show=True, style='scatter', size=1.0):
    '''Plots the x and y components of tracks. Can be used to visualize reprojection errors.

    Parameters
    ----------
    tracks : dict
        A dictionary containing the tracks
    ax : matplotlib.pyplot.Axes, optional
        Axes for plotting
    figsize : (int, int), optional
        Size of the matplotlib output if ax is not specified. Defaults to (30, 30)
    show : bool, optional
        Show plot calling plt.show. Defaults to True.
    style : str, optional
        Plot style, one of "line", "scatter" or "errors". Defaults to "scatter".
    size : float, optional
        Line width or marker size. Defaults to 1.0.

    Returns
    -------
    matplotlib.pyplot.Axes
    '''

    assert style in ['scatter', 'line', 'errors'], 'style argument should be either "scatter", "line" or "errors"'
    assert style != 'errors' or (style == 'errors' and 'REPR_ERROR' in tracks[str(tracks['IDENTITIES'][0])]), 'calculate reprojection errors first'
    if style == 'errors':
        errors = np.concatenate([tracks[str(i)]['REPR_ERROR'] for i in tracks['IDENTITIES']])
        vmin = errors.min()
        vmax = errors.max()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    for i in tracks['IDENTITIES']:
        if style == 'scatter':
            ax.scatter(tracks[str(i)]['X'], tracks[str(i)]['Y'], s=size)
        elif style == 'errors':
            mappable = ax.scatter(tracks[str(i)]['X'],
                                  tracks[str(i)]['Y'],
                                  s=size, c=tracks[str(i)]['REPR_ERROR'],
                                  cmap=plt.get_cmap('Spectral_r'),
                                  vmin=vmin, # np.quantile(tracks[str(i)]['REPR_ERROR'], 0.05)
                                  vmax=vmax) # np.quantile(tracks[str(i)]['REPR_ERROR'], 0.95)
        elif style == 'line':
            ax.plot(tracks[str(i)]['X'], tracks[str(i)]['Y'], lw=size)
    ax.set_aspect('equal')

    if style == 'errors':
        fig = plt.gcf()
        axis_bounds = ax.get_position().get_points()
        cax = fig.add_axes([1.05,
                            axis_bounds[0, 1],
                            0.01,
                            axis_bounds[1, 1] - axis_bounds[0, 1]])
        fig.colorbar(mappable, cax=cax);

    if show:
        plt.show()
    return ax

def tracks_to_ply(tracks, uniform_color=None):
    '''Prepare tracks for ply file save.

    Parameters
    ----------
    tracks : dict
        A tracks dictionary, not a pooled dictionary, must contain z component
    uniform_color : (int, int, int), optional
        A shared color used for all tracks, otherwise random RGB generation

    Returns
    -------
    list
        A list of lists of per individual pre-formatted ply points (x y z r g b a)
    '''

    if not os.path.exists(file_name):
        os.mkdir(file_name)
    assert 'IDENTITIES' in tracks and 'FRAME_IDX' in tracks, 'Tracks should be in track dictionary format, if pooled, use tracks_from_pooled first'
    ply_tracks = []
    for i in tracks['IDENTITIES']:
        color = tuple(np.random.uniform(50, 230, 3))
        assert 'Z' in tracks[str(i)], 'Tracks do not contain Z, not supported for ply format'
        pts_3d = np.transpose([tracks[str(i)]['X'],
                               tracks[str(i)]['Y'],
                               tracks[str(i)]['Z'],
                               np.repeat(color[0],tracks[str(i)]['X'].size),
                               np.repeat(color[1],tracks[str(i)]['X'].size),
                               np.repeat(color[2],tracks[str(i)]['X'].size)])
        ply_tracks.append(pointcloud_to_ply(pts_3d))
    return ply_tracks

def pointcloud_to_ply(point_cloud):
    '''Returns pre-formatted ply points from input points, use Scene.get_pointcloud'''

    pts_ply = []
    for pt in point_cloud:
        pts_ply.append('{:f} {:f} {:f} {:.0f} {:.0f} {:.0f} 0\n'.format(*pt))
    return pts_ply

def write_ply(pts_ply, file_name):
    '''Write points to a .ply file for visualization

    Parameters
    ----------
    pts_ply : list
        The prepared points in ply format, for example using tracks_to_ply
    file_name : str
        The file name of the saved file

    Returns
    -------
    bool
        Successful save?
    '''

    with open(file_name, 'w') as fid:
        fid.write(('ply\n' + \
                   'format ascii 1.0\n' + \
                   'element vertex {:d}\n' + \
                   'property float x\n' + \
                   'property float y\n' + \
                   'property float z\n' + \
                   'property uchar red\n' + \
                   'property uchar green\n' + \
                   'property uchar blue\n' + \
                   'property uchar alpha\n' + \
                   'end_header\n' + \
                   '{}\n').format(len(pts_ply), ''.join(pts_ply)))

def compute_reprojection_errors(tracks, tracks_reprojected, identities=[]):
    '''Computes point-wise reprojection errors (distances) between tracks and their reprojections.

    Parameters
    ----------
    tracks : dict
        A dictionary containing the original tracks
    tracks_reprojected : dict
        A dictionary containing the reprojected tracks
    identities : list, optional
        A list of trajectory identities for which the reprojection errors should be computed

    Returns
    -------
    list
        A list of arrays containing the reprojection errors for each identity
    array
        An array containing the respective identities
    '''

    errors = []
    if len(identities) == 0:
        identities = tracks['IDENTITIES'][np.isin(tracks['IDENTITIES'], tracks_reprojected['IDENTITIES'])]
    for i in identities:
        shared = np.isin(tracks[str(i)]['FRAME_IDX'], tracks_reprojected[str(i)]['FRAME_IDX'])
        shared_reprojected = np.isin(tracks_reprojected[str(i)]['FRAME_IDX'], tracks[str(i)]['FRAME_IDX'])
        x = tracks[str(i)]['X'][shared]
        x_reprojected = tracks_reprojected[str(i)]['X'][shared_reprojected]
        y = tracks[str(i)]['Y'][shared]
        y_reprojected = tracks_reprojected[str(i)]['Y'][shared_reprojected]
        errors.append(np.sqrt(np.square(np.transpose([x, y]) - np.transpose([x_reprojected, y_reprojected])).sum(axis=1)))
    return errors, identities
