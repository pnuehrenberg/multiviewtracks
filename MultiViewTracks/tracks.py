import numpy as np

def tracks_to_pooled(tracks):
    '''Convert the tracks dictionary into a dictionary with wide table format.'''

    assert 'IDENTITIES' in tracks and 'FRAME_IDX' in tracks, 'Tracks should be in track dictionary format'
    pooled = {}
    for i in tracks['IDENTITIES']:
        if len(pooled) == 0:
            pooled = {key: [] for key in tracks[str(i)]}
            pooled['IDENTITY'] = []
        for key in tracks[str(i)]:
            pooled[key].append(tracks[str(i)][key])
        pooled['IDENTITY'].append(np.repeat(i, tracks[str(i)][key].shape[0]).astype(np.int))
    for key in pooled:
        pooled[key] = np.concatenate(pooled[key], axis=0)
    return pooled

def tracks_from_pooled(pooled):
    '''Convert tracks dictionary from a dictionary with wide table format.'''

    assert 'IDENTITY' in pooled and 'FRAME_IDX' in pooled, 'Pooled should be in pooled dictionary format'
    tracks = {'IDENTITIES': [], 'FRAME_IDX': []}
    for i in np.unique(pooled['IDENTITY']):
        tracks[str(i)] = {key: [] for key in pooled if key != 'IDENTITY'}
        for key in tracks[str(i)]:
            tracks[str(i)][key] = pooled[key][pooled['IDENTITY'] == i]
        tracks['IDENTITIES'].append(i)
        tracks['FRAME_IDX'].append(tracks[str(i)]['FRAME_IDX'])
    tracks['IDENTITIES'] = np.array(tracks['IDENTITIES'], dtype=np.int)
    tracks['FRAME_IDX'] = np.unique(np.concatenate(tracks['FRAME_IDX'])).astype(np.int)
    return tracks

def rotate_tracks(tracks, pca):
    '''Rotate tracks with a given pca transform.

    Parameters
        tracks (dict): A dictionary containing the tracks
        pca (sklearn.decomposition.PCA): A fitted PCA instance used for transformation

    Returns:
        dict: The rotated tracks
    '''

    pooled = tracks_to_pooled(tracks)
    pts_3d = np.transpose([pooled['X'], pooled['Y'], pooled['Z']])
    pts_3d = pca.transform(pts_3d)
    pooled['X'] = pts_3d[:, 0]
    pooled['Y'] = pts_3d[:, 1]
    pooled['Z'] = pts_3d[:, 2]
    return tracks_from_pooled(pooled)

def scale_tracks(tracks, scale):
    '''Scale tracks with a given scale.

    Parameters
        tracks (dict): A dictionary containing the tracks
        scale (float): The scale used for transformation

    Returns:
        dict: The scales tracks
    '''

    pooled = tracks_to_pooled(tracks)
    pts_3d = np.transpose([pooled['X'], pooled['Y'], pooled['Z']]) * scale
    pooled['X'] = pts_3d[:, 0]
    pooled['Y'] = pts_3d[:, 1]
    pooled['Z'] = pts_3d[:, 2]
    return tracks_from_pooled(pooled)

def interpolate_trajectory(trajectory):
    '''Linearly interpolate X, Y and Z of one trajectory or sub-trajectory.'''

    trajecty_interpolated = {}
    frame_idx = np.arange(trajectory['FRAME_IDX'].min(), trajectory['FRAME_IDX'].max() + 1)
    sort_idx = np.argsort(trajectory['FRAME_IDX'])
    for key in trajectory:
        if key in ['X', 'Y', 'Z']:
            trajecty_interpolated[key] = np.interp(frame_idx,
                                                   trajectory['FRAME_IDX'][sort_idx],
                                                   trajectory[key][sort_idx])
    trajecty_interpolated['FRAME_IDX'] = frame_idx
    return trajecty_interpolated

def interpolate_tracks(tracks):
    '''Linearly interpolate X, Y and Z components of track dictionary.'''

    assert 'IDENTITIES' in tracks and 'FRAME_IDX' in tracks, 'Tracks should be in track dictionary format, if pooled, use tracks_from_pooled first'
    tracks_interpolated = {'FRAME_IDX': []}
    for i in tracks['IDENTITIES']:
        tracks_interpolated[str(i)] = interpolate_trajectory(tracks[str(i)])
        tracks_interpolated['FRAME_IDX'].append(tracks_interpolated[str(i)]['FRAME_IDX'])
    tracks_interpolated['FRAME_IDX'] = np.unique(np.concatenate(tracks_interpolated['FRAME_IDX'])).astype(np.int)
    tracks_interpolated['IDENTITIES'] = tracks['IDENTITIES'].astype(np.int)
    return tracks_interpolated

def interpolate_subtracks(sub_tracks):
    '''Linearly interpolate X, Y and Z components of subtrack dictionary.'''

    assert 'IDENTITIES' in sub_tracks and not 'FRAME_IDX' in sub_tracks, 'Sub-tracks should be in sub-tracks dictionary format'
    for i in sub_tracks['IDENTITIES']:
        sub_tracks[str(i)] = [interpolate_trajectory(sub_trajectory) for sub_trajectory in sub_tracks[str(i)]]
    return sub_tracks

def trajectory_to_subtrajectories(trajectory, max_dist):
    '''Split one trajectory into sub-trajectories with specified maximum distance.'''

    components = [component for component in trajectory if component in ['X', 'Y', 'Z']]
    positions = np.transpose([trajectory[component] for component in components])
    distances = np.sqrt(np.square(np.diff(positions, axis=0)).sum(axis=1))
    sub_trajectories = {key: np.split(trajectory[key], np.argwhere(distances > max_dist).ravel() + 1) for key in trajectory}
    sub_trajectories = [{key: sub_trajectories[key][idx] for key in sub_trajectories} \
                        for idx in range(len(sub_trajectories['FRAME_IDX']))]
    return sub_trajectories

def trajectory_from_subtrajectories(sub_trajectories):
    '''Returns a trajectory joined from sub-trajectories'''

    assert len(sub_trajectories) > 0, 'Sub-trajectories must contain at least one trajectory'
    trajectory = {key: np.concatenate([sub_trajectories[idx][key] for idx in range(len(sub_trajectories))]) \
                  for key in sub_trajectories[0]}
    assert trajectory['FRAME_IDX'].size == np.unique(trajectory['FRAME_IDX']).size, 'Sub-trajectories share frames'
    return trajectory

def tracks_to_subtracks(tracks, max_dist):
    '''Split tracks into sub-trajectories.

    Paramteters:
        tracks (dict): The tracks dictionary
        max_dist (float): Maximum distance between consequtive frames that is allowed in a sub-trajectory

    Returns:
        dict: tracks dictionary with sub-trajectories
    '''

    assert 'IDENTITIES' in tracks and 'FRAME_IDX' in tracks, 'Tracks should be in track dictionary format, if pooled, use tracks_from_pooled first'
    sub_tracks = {'IDENTITIES': tracks['IDENTITIES']}
    for i in tracks['IDENTITIES']:
        sub_tracks[str(i)] = trajectory_to_subtrajectories(tracks[str(i)], max_dist)
    return sub_tracks

def tracks_from_subtracks(sub_tracks):
    '''Returns tracks joined from given sub-tracks'''

    assert 'IDENTITIES' in sub_tracks and 'FRAME_IDX' not in sub_tracks, 'Sub-tracks should be in sub-tracks dictionary format'
    tracks = {'IDENTITIES': sub_tracks['IDENTITIES'], 'FRAME_IDX': []}
    for i in sub_tracks['IDENTITIES']:
        tracks[str(i)] = trajectory_from_subtrajectories(sub_tracks[str(i)])
        tracks['FRAME_IDX'].append(tracks[str(i)]['FRAME_IDX'])
    tracks['FRAME_IDX'] = np.unique(np.concatenate(tracks['FRAME_IDX'])).astype(np.int)
    return tracks
