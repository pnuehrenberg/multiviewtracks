import cv2
import quaternion as quat
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation, Slerp

from .utils import *
from .tracks import *

class Camera:
    '''This is a class containing per camera preprocessing methods for triangulating tracks using the Scene class.

    Attributes
    ----------
    id : int
        The camera id within a COLMAP reconstruction
    name : str
        A common prefix used for all images of this camera in a COLMAP reconstruction
    fisheye : bool
        Was OPENCV_FISHEYE used as COLMAP camera model?
    image_names : np.ndarray
        Contains the image names of the reconstructed views of this camera
    view_idx : np.ndarray
        Contains the frame indices of all reconstructed views of this camera
    n_views : int
        Number of reconstrueced views of this camera
    R : scipy.spatial.transform.Rotation
        A Rotation instance holding extrinsic parameters (rotations) for this camera
    r : np.ndarray
        Stores the rotation matrices retrieved from R
    t : np.ndarray
        Stores COLMAP extrinsic camera parameters for each view of this camera
    k : np.ndarray
        The camera matrix of this camera
    d : np.ndarray
        The distortion parameters of this camera
    tracks : dict
        Contains the tracks visible from this camera or None
    tracks_undistorted : dict
        Contains undistorted tracks from Camera.undistort_tracks or None
    tracks_projected : dict
        Contains transformed tracks from Camera.transform_tracks or None
    verbose : bool
        Do you want some verbosity? Defaults to true
    '''

    def __init__(self, id, name, fisheye, extrinsics, intrinsics, tracks, verbose=True):
        '''The constructor of Camera class objects.

        Parameters
        ----------
        id : int
            The camera id within a COLMAP reconstruction
        name : str
            A common prefix used for all images of this camera in a COLMAP reconstruction
        fisheye : bool
            Was OPENCV_FISHEYE used as COLMAP camera model?
        extrinsics : dict
            A dictionary storing COLMAP extrinsic parameters of this camera
        intrinsics : np.ndarray
            COLMAP intrinsic camera parameters
        tracks : dict
            The tracks visible from this camera or None
        verbose : bool, optional
            Do you want some verbosity? Defaults to true
        '''

        self.id = id
        self.name = name
        self.fisheye = fisheye
        self.image_names = extrinsics['IMAGE_NAME']
        view_idx = extrinsics['FRAME_IDX']
        sort_idx = np.argsort(view_idx)
        self.view_idx = view_idx[sort_idx]
        self.n_views = self.view_idx.size
        self.R = Rotation.from_quat([[extrinsics['Q1'][sort_idx][idx],
                                      extrinsics['Q2'][sort_idx][idx],
                                      extrinsics['Q3'][sort_idx][idx],
                                      extrinsics['Q4'][sort_idx][idx]] for idx in range(self.n_views)])
        self.r = self.get_rotations()
        self.t = np.transpose([extrinsics['TX'][sort_idx],
                               extrinsics['TY'][sort_idx],
                               extrinsics['TZ'][sort_idx]])
        self.k = np.array([[intrinsics[0], 0, intrinsics[2]],
                           [0, intrinsics[1], intrinsics[3]],
                           [0, 0, 1]])
        self.d = intrinsics[4:8]
        self.tracks = tracks
        self.tracks_undistorted = None
        self.tracks_projected = None
        self.verbose = verbose
        if self.verbose:
            print('  Initialized', self)

    def __repr__(self):
        return 'Camera {} | {}, {} views, {} tracks{}'.format(self.id, self.name, self.n_views,
                                                              'with' if self.tracks is not None else 'no',
                                                              ', fisheye' if self.fisheye else '')

    def get_rotations(self):
        '''Returns a np.ndarray of rotation matrices transformed from R.'''

        quaternions = quat.as_quat_array(self.R.as_quat())
        return quat.as_rotation_matrix(quaternions)

    def interpolate(self):
        '''Interpolates the camera path by linearly interpolating t and using SLERP for R.'''

        view_idx = np.arange(self.view_idx.min(), self.view_idx.max() + 1)
        t = []
        for idx in range(3):
            t.append(np.interp(view_idx, self.view_idx, self.t[:, idx]))
        self.t = np.transpose(t)
        slerp = Slerp(self.view_idx, self.R)
        self.R = slerp(view_idx)
        self.r = self.get_rotations()
        self.view_idx = view_idx
        self.n_views = view_idx.size
        if self.verbose:
            print('  Interpolated', self)

    def undistort_tracks(self):
        '''Undistorts the tracks in image coordinates to normalized coordinates.'''

        if self.tracks is None or self.tracks_undistorted is not None:
            return
        pooled = tracks_to_pooled(self.tracks)
        reconstructed = np.isin(pooled['FRAME_IDX'], self.view_idx)
        for key in pooled:
            pooled[key] = pooled[key][reconstructed]
        pts_2d = np.transpose([pooled['X'], pooled['Y']]).reshape(-1, 1, 2).astype(np.float)
        if self.fisheye:
            pts_2d = cv2.fisheye.undistortPoints(pts_2d, self.k, self.d).reshape(-1, 2)
        else:
            pts_2d = cv2.undistortPoints(pts_2d, self.k, self.d).reshape(-1, 2)
        pooled['X'] = pts_2d[:, 0]
        pooled['Y'] = pts_2d[:, 1]
        self.tracks_undistorted = tracks_from_pooled(pooled)
        if self.verbose:
            print('  Undistorted tracks for Camera {} | {}'. format(self.id, self.name))

    def project_tracks(self):
        '''Projects the tracks to world coordinates with unknown depth using r and t, Camera.undistort_tracks first if necessary.'''

        if self.tracks is None or self.tracks_projected is not None:
            return
        elif self.undistort_tracks is None:
            self.undistort_tracks()
        pooled = tracks_to_pooled(self.tracks_undistorted)
        pooled['Z'] = np.repeat(np.nan, pooled['X'].size)
        for idx in self.view_idx:
            pts_2d = np.transpose([pooled['X'][pooled['FRAME_IDX'] == idx],
                                   pooled['Y'][pooled['FRAME_IDX'] == idx],
                                   np.repeat(1, np.sum(pooled['FRAME_IDX'] == idx))])
            if pts_2d.shape[0] > 0:
                pts_3d = []
                for pt_2d in pts_2d:
                    pt_3d = pt_2d.reshape(1, 3) - self.t[self.view_idx == idx].reshape(1, 3)
                    pt_3d = pt_3d @ inv(self.r[self.view_idx == idx]).reshape(3, 3).T
                    pts_3d.append(pt_3d.ravel())
                pts_3d = np.array(pts_3d)
                pooled['X'][pooled['FRAME_IDX'] == idx] = pts_3d[:, 0]
                pooled['Y'][pooled['FRAME_IDX'] == idx] = pts_3d[:, 1]
                pooled['Z'][pooled['FRAME_IDX'] == idx] = pts_3d[:, 2]
        self.tracks_projected = tracks_from_pooled(pooled)
        if self.verbose:
            print('  Projected tracks for Camera {} | {}'. format(self.id, self.name))

    def frames_in_view(self, i):
        '''Returns the frame indices in which individual i is observed in the camera views.'''

        if self.tracks is None or i not in self.tracks['IDENTITIES']:
            return np.array([])
        return self.tracks[str(i)]['FRAME_IDX'][np.isin(self.tracks[str(i)]['FRAME_IDX'], self.view_idx)]

    def view(self, idx):
        '''Returns the projection matrix of the camera at the specified frame index.'''

        assert idx in self.view_idx, 'View {} not reconstructed in Camera {} | {}'.format(idx, self.id, self.name)
        r = self.r[self.view_idx == idx].reshape(3, 3)
        t = self.t[self.view_idx == idx].reshape(3, 1)
        return np.append(r, t, axis=1)

    def position(self, idx, i, kind=''):
        '''Returns the position of individual i at the specified frame index.

        Parameters
        ----------
        idx : int
            The frame index
        i : int
            The individual's identity
        kind : str, optional
            One of "", "undistorted", "projected". Defaults to ""

        Returns
        -------
        np.ndarray
            The position of individual i at frame index idx
        '''

        assert idx in self.frames_in_view(i), 'Individual {} not in view {} of camera {} | {}'.format(i, idx, self.id, self.name)
        if kind == '':
            tracks = self.tracks
        elif kind == 'undistorted':
            tracks = self.tracks_undistorted
        elif kind == 'projected':
            tracks = self.tracks_projected
        else:
            tracks = None
        assert tracks is not None, 'Specify valid tracks kind: "", "undistorted" or "projected" and run Camera.undistort_tracks or Camera.project_tracks first.'
        if kind in ['', 'undistorted']:
            return np.array([tracks[str(i)]['X'][tracks[str(i)]['FRAME_IDX'] == idx],
                             tracks[str(i)]['Y'][tracks[str(i)]['FRAME_IDX'] == idx]]).ravel()
        else:
            return np.array([tracks[str(i)]['X'][tracks[str(i)]['FRAME_IDX'] == idx],
                             tracks[str(i)]['Y'][tracks[str(i)]['FRAME_IDX'] == idx],
                             tracks[str(i)]['Z'][tracks[str(i)]['FRAME_IDX'] == idx]]).ravel()


    def projection_center(self, idx):
        '''Return the 3d coordinates of the idx-th view projection center.'''

        return (-self.r[self.view_idx == idx].reshape(3, 3).T @ self.t[self.view_idx == idx].ravel()).ravel()
