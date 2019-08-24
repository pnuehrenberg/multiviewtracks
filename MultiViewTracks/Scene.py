import os
import numpy as np
from glob import glob
from sklearn.decomposition import PCA

from .Camera import Camera
from .utils import *
from .tracks import *

class Scene:
    '''This is a class for triangulating tracks using the camera parameters of a COLMAP reconstruction.

    Attributes
    ----------
    model_path : str
        Path to the COLMAP model .bin files
    tracks_path : str
        Path to the tracks .pkl files
    fisheye : bool
        Did you use OPENCV_FISHEYE in COLMAP reconstruction?
    verbose : bool
        Do you want a bit of verbosity?
    extrinsics : dict
        Stores COLMAP extrinsic camera parameters
    intrinsics : dict
        Stores COLMAP intrinsic camera parameters
    cameras : dict
        Stores Camera class instance for each reconstructed camera
    tracks : dict
        Stores tracks for each camera
    tracks_triangulated : dict
        Stores the triangulated multiple-view tracks, otherwise None
    tracks_projected : dict
        Stores the projected single-view tracks, otherwise None
    tracks_3d : dict
        Stores the combined 3d tracks, otherwise None
    pts_3d : np.ndarray
        Stores the sparse COLMAP point cloud, otherwise None
    '''

    def __init__(self, model_path, tracks_path, fisheye, verbose=True):
        '''
        The constructor of Scene class objects.

        Parameters
        ----------
        model_path : str
            Path to the COLMAP model .bin files
        tracks_path : str
            Path to the tracks .pkl files
        fisheye : bool
            Did you use OPENCV_FISHEYE in COLMAP reconstruction?
        camera_names : list
            Contains the camera names (image prefixes) read from the COLMAP reconstruction
        verbose : bool, optional
            Do you want a bit of verbosity? Defaults to True
        '''

        self.model_path = model_path
        self.tracks_path = tracks_path
        self.fisheye = fisheye
        self.verbose = verbose
        self.get_extrinsics()
        self.get_intrinsics()
        self.get_tracks()
        self.tracks_triangulated = None
        self.tracks_projected = None
        self.tracks_3d = None
        self.sparse = None

    def get_extrinsics(self):
        '''Read the COLMAP extxrinsic camera parameters.

        See https://github.com/colmap/colmap/blob/dev/scripts/python/read_model.py for reference.
        '''

        extrinsics_file = os.path.join(self.model_path, 'images.bin')
        if self.verbose:
            print('Reading extrinsics from {}'.format(extrinsics_file))
        views = []
        with open(extrinsics_file, 'rb') as fid:
            n_views = read_next_bytes(fid, 8, 'Q')[0]
            for idx in np.arange(n_views):
                binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence='idddddddi')
                image_name = ''
                current_char = read_next_bytes(fid, 1, 'c')[0]
                while current_char != b'\x00':
                    image_name += current_char.decode('utf-8')
                    current_char = read_next_bytes(fid, 1, 'c')[0]
                binary_image_properties = list(binary_image_properties)
                binary_image_properties.append(image_name)
                views.append(binary_image_properties)
                n_pts_2d = read_next_bytes(fid, num_bytes=8, format_char_sequence='Q')[0]
                x_y_id_s = read_next_bytes(fid, num_bytes=24 * n_pts_2d, format_char_sequence='ddq' * n_pts_2d)
        extrinsics = {}
        for idx, key in enumerate(['IMAGE_ID', 'Q1', 'Q2', 'Q3', 'Q4', 'TX', 'TY', 'TZ', 'CAMERA_ID', 'IMAGE_NAME']):
            extrinsics[key] = np.array([view[idx] for view in views])
            if 'ID' in key:
                extrinsics[key] = extrinsics[key].astype(np.int)
            elif key == 'IMAGE_NAME':
                extrinsics[key] = extrinsics[key].astype(np.str)
            else:
                extrinsics[key] = extrinsics[key].astype(np.float)
        extrinsics['FRAME_IDX'] = np.array([os.path.splitext(os.path.basename(image_name))[0].split('_')[-1] \
                                            for image_name in extrinsics['IMAGE_NAME']]).astype(np.int)
        sort_idx = np.argsort(extrinsics['IMAGE_ID'])
        for key in extrinsics:
            extrinsics[key] = extrinsics[key][sort_idx]
        self.extrinsics = extrinsics
        self.camera_names = np.unique(['_'.join(os.path.basename(image_name).split('_')[:-1]) for image_name in extrinsics['IMAGE_NAME']])

    def get_intrinsics(self):
        '''Read the COLMAP intrinsic camera parameters. Camera model should be OPENCV or OPENCV_FISHEYE.

        See https://github.com/colmap/colmap/blob/dev/scripts/python/read_model.py for reference.
        '''

        intrinsics_file = os.path.join(self.model_path, 'cameras.bin')
        if self.verbose:
            print('Reading intrinsics from {}'.format(intrinsics_file))
        intrinsics = {}
        with open(intrinsics_file, 'rb') as fid:
            n_cameras = read_next_bytes(fid, 8, 'Q')[0]
            for idx in range(n_cameras):
                camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence='iiQQ')
                camera_id = camera_properties[0]
                num_params = 8
                params = read_next_bytes(fid, num_bytes=8 * num_params, format_char_sequence='d' * num_params)
                intrinsics[camera_id] = np.array(params)
        self.intrinsics = intrinsics

    def get_sparse(self):
        '''Read the COLMAP reconstruction point cloud.

        See https://github.com/colmap/colmap/blob/dev/scripts/python/read_model.py for reference.
        '''

        points_3d_file = os.path.join(self.model_path, 'points3D.bin')
        if self.verbose:
            print('Reading intrinsics from {}'.format(points_3d_file))
        pts_3d = []
        with open(points_3d_file, 'rb') as fid:
            n_pts = read_next_bytes(fid, 8, 'Q')[0]
            for idx in range(n_pts):
                binary_point_line_properties = read_next_bytes(fid, num_bytes=43, format_char_sequence='QdddBBBd')
                pt_3d_idx = binary_point_line_properties[0]
                xyz = np.array(binary_point_line_properties[1:4])
                rgb = np.array(binary_point_line_properties[4:7])
                error = np.array(binary_point_line_properties[7])
                track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence='Q')[0]
                track_elems = read_next_bytes(fid, num_bytes=8 * track_length, format_char_sequence='ii' * track_length)
                pts_3d.append(np.concatenate([xyz, rgb]))
        self.sparse = np.array(pts_3d).reshape(-1, 6)

    def get_tracks(self):
        '''Read the tracks .pkl files. The file names should match the camera name, but can have a prefix (i.e. prefix[camera_name].pkl)'''

        self.tracks = {}
        if self.verbose:
            print('Reading tracks from {}'.format(self.tracks_path))
        for camera_id, camera_name in zip(sorted(self.intrinsics), self.camera_names):
            tracks = glob(os.path.join(self.tracks_path, '*{}.pkl'.format(camera_name)))
            if len(tracks) != 1:
                self.tracks[camera_id] = None
                if self.verbose:
                    print('  Could not read tracks for Camera {} | {}: {}'.format(camera_id, camera_name, ' '.join(tracks)))
            else:
                self.tracks[camera_id] = load(tracks[0])
                if self.verbose:
                    print('  Read tracks for Camera {} | {}: {}'.format(camera_id, camera_name, ' '.join(tracks)))

    def get_cameras(self):
        '''Creates Camera objects for each unique image prefix with COLMAP reconstruction parameters.'''

        self.cameras = {}
        if self.verbose:
            print('Creating cameras')
        for camera_id, camera_name in zip(sorted(self.intrinsics), self.camera_names):
            self.cameras[camera_id] = Camera(camera_id,
                                            camera_name,
                                            self.fisheye,
                                            {key: self.extrinsics[key][self.extrinsics['CAMERA_ID'] == camera_id] \
                                             for key in self.extrinsics},
                                            self.intrinsics[camera_id],
                                            self.tracks[camera_id],
                                            self.verbose)

    def interpolate_cameras(self):
        '''Interpolates the camera paths of the Scene using Camera.interpolate.'''

        if self.verbose:
            print('Interpolating cameras')
        for camera_id in self.cameras:
            self.cameras[camera_id].interpolate()

    def undistort_tracks(self):
        '''Undistorts the tracks of all cameras using Camera.undistort_tracks.'''

        for camera_id in self.cameras:
            self.cameras[camera_id].undistort_tracks()

    def project_tracks(self):
        '''Projects the tracks of all cameras using Camera.project_tracks.'''

        for camera_id in self.cameras:
            self.cameras[camera_id].project_tracks()

    def triangulate_multiview_tracks(self):
        '''Triangulate all trajectory points that are observed in more than one view.'''

        if self.verbose:
            print('Triangulating multiple-view trajectories')
        self.undistort_tracks()
        pooled = {'X': [], 'Y': [], 'Z': [], 'FRAME_IDX': [], 'IDENTITY': []}
        identities = np.unique(np.concatenate([self.cameras[camera_id].tracks['IDENTITIES'] \
                                               for camera_id in self.cameras \
                                               if self.cameras[camera_id].tracks is not None]))
        for i in identities:
            view_idx = np.concatenate([self.cameras[camera_id].frames_in_view(i) for camera_id in self.cameras])
            for idx in np.unique(view_idx):
                if np.sum(view_idx == idx) < 2:
                    continue
                pts_2d = []
                views = []
                for camera_id in self.cameras:
                    if idx not in self.cameras[camera_id].frames_in_view(i):
                        continue
                    pts_2d.append(self.cameras[camera_id].position(idx, i, 'undistorted'))
                    views.append(self.cameras[camera_id].view(idx))
                pt_3d = triangulate_point(pts_2d, views)
                pooled['X'].append(pt_3d[0])
                pooled['Y'].append(pt_3d[1])
                pooled['Z'].append(pt_3d[2])
                pooled['IDENTITY'].append(i)
                pooled['FRAME_IDX'].append(idx)
        for key in pooled:
            pooled[key] = np.array(pooled[key])
        self.tracks_triangulated = tracks_from_pooled(pooled)

    def project_singleview_tracks(self):
        '''Project all trajectory points that are observed in only one view to an interpolated detph.
        Use this if the tracks are mostly planar and uncomplete.'''

        if self.verbose:
            print('Projecting single-view trajectories')
        self.undistort_tracks()
        self.project_tracks()
        if self.tracks_triangulated is None:
            self.triangulate_multiview_tracks()
        pooled = {'X': [], 'Y': [], 'Z': [], 'FRAME_IDX': [], 'IDENTITY': []}
        identities = np.unique(np.concatenate([self.cameras[camera_id].tracks['IDENTITIES'] \
                                               for camera_id in self.cameras \
                                               if self.cameras[camera_id].tracks is not None]))
        for i in identities:
            view_idx = np.concatenate([self.cameras[camera_id].frames_in_view(i) for camera_id in self.cameras])
            frame_idx = np.arange(view_idx.min(), view_idx.max() + 1)
            z = np.interp(frame_idx,
                          self.tracks_triangulated[str(i)]['FRAME_IDX'],
                          self.tracks_triangulated[str(i)]['Z'])
            for idx in np.unique(view_idx):
                if np.sum(view_idx == idx) != 1:
                    continue
                camera_id = [camera_id for camera_id in self.cameras \
                             if idx in self.cameras[camera_id].frames_in_view(i)][0]
                center = self.cameras[camera_id].projection_center(idx)
                depth = z[frame_idx == idx][0]
                pt_projected = self.cameras[camera_id].position(idx, i, 'projected')
                ray = pt_projected - center
                scale = (depth - center[2]) / ray[2]
                pt_3d = center + ray * scale
                pooled['X'].append(pt_3d[0])
                pooled['Y'].append(pt_3d[1])
                pooled['Z'].append(pt_3d[2])
                pooled['IDENTITY'].append(i)
                pooled['FRAME_IDX'].append(idx)
        for key in pooled:
            pooled[key] = np.array(pooled[key])
        self.tracks_projected = tracks_from_pooled(pooled)

    def get_tracks_3d(self):
        '''Combine triangulated multiple-view trajectories and projected single-view tracks.'''

        if self.verbose:
            print('Combining mutliple-view and projected trajectories')
        if self.tracks_triangulated is None:
            self.triangulate_multiview_tracks()
        if self.tracks_projected is None:
            self.project_singleview_tracks()
        pooled_triangulated = tracks_to_pooled(self.tracks_triangulated)
        pooled_projected = tracks_to_pooled(self.tracks_projected)
        pooled = {key: np.concatenate([pooled_triangulated[key], pooled_projected[key]]) \
                  for key in pooled_triangulated}
        sort_idx = np.argsort(pooled['FRAME_IDX'])
        for key in pooled:
            pooled[key] = pooled[key][sort_idx]
        self.tracks_3d = tracks_from_pooled(pooled)

    def rotate(self):
        '''Rotates the tracks and 3d point cloud using PCA, so that the first two principal components of the camera paths are x and y'''

        pts_3d = []
        for camera_id in self.cameras:
            pts_3d.append(np.array([self.cameras[camera_id].projection_center(idx) \
                                       for idx in self.cameras[camera_id].view_idx]))
        pts_3d = np.concatenate(pts_3d)
        pca = PCA(n_components=3)
        pca.fit(pts_3d)
        if self.tracks_triangulated is not None:
            if self.verbose:
                print('Rotating triangulated multiple-view trajectories')
            self.tracks_triangulated = rotate_tracks(self.tracks_triangulated, pca)
        if self.tracks_projected is not None:
            if self.verbose:
                print('Rotating projected single-view trajectories')
            self.tracks_projected = rotate_tracks(self.tracks_projected, pca)
        if self.tracks_3d is not None:
            if self.verbose:
                print('Rotating combined 3d trajectories')
            self.tracks_3d = rotate_tracks(self.tracks_3d, pca)
        if self.sparse is not None:
            if self.verbose:
                print('Rotating sparse point cloud')
            self.sparse[:, :3] = pca.transform(self.sparse[:, :3])

    def scale(self, camera_ids, world_distance):
        '''Scales the tracks and 3d point cloud according to a known camera-to-camera distance.

        Parameters
        ----------
        camera_ids : (int, int)
            The camera ids used to calculated the distance for scaling
        world_distance : float
            The known real-world distance between the two specified cameras

        Returns
        -------
        np.ndarray
            The reconstruction errors calculated as the difference between reconstruted and measured distance
        '''

        cameras = [self.cameras[camera_ids[0]], self.cameras[camera_ids[1]]]
        reconstructed = [np.isin(cameras[0].view_idx, cameras[1].view_idx),
                         np.isin(cameras[1].view_idx, cameras[0].view_idx)]
        pts_3d = [np.array([cameras[0].projection_center(idx) for idx in cameras[0].view_idx]),
                  np.array([cameras[1].projection_center(idx) for idx in cameras[1].view_idx])]
        distances = np.sqrt(np.square(pts_3d[0][reconstructed[0]] - pts_3d[1][reconstructed[1]]).sum(axis=1))
        scale = world_distance / distances.mean()
        if self.tracks_triangulated is not None:
            if self.verbose:
                print('Scaling triangulated multiple-view trajectories')
            self.tracks_triangulated = scale_tracks(self.tracks_triangulated, scale)
        if self.tracks_projected is not None:
            if self.verbose:
                print('Scaling projected single-view trajectories')
            self.tracks_projected = scale_tracks(self.tracks_projected, scale)
        if self.tracks_3d is not None:
            if self.verbose:
                print('Scaling combined 3d trajectories')
            self.tracks_3d = scale_tracks(self.tracks_3d, scale)
        if self.sparse is not None:
            if self.verbose:
                print('Scaling sparse point cloud')
            self.sparse[:, :3] = self.sparse[:, :3] * scale
        errors = (distances * scale) - world_distance
        return errors
