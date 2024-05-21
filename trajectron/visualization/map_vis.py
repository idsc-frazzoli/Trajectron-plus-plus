import os.path

import matplotlib.pyplot as plt
import tqdm
import numpy as np
from typing import List
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm
from pyquaternion import Quaternion
import matplotlib.patches as patches


def get_agent_occs_from_sample(nusc: NuScenes, sample_token: str):
    # For each sample in the scene, store the ego pose.
    sample_record = nusc.get('sample', sample_token)

    # Poses are associated with the sample_data. Here we use the lidar sample_data.
    sample_data_record = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
    pose_record = nusc.get('ego_pose', sample_data_record['ego_pose_token'])
    # Calculate the pose on the map and append.
    pos = [pose_record['translation'][0], pose_record['translation'][1]]
    heading = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
    size = [2, 4]
    ego_occ = (pos, heading, size)
    map_pose = pose_record['translation']

    # retrieve poses of other agents(pose and size in sample annotation)
    annotation_tokens = sample_record['anns']
    agent_occs = {}
    for annotation_token in annotation_tokens:
        annotation = nusc.get('sample_annotation', annotation_token)
        category = annotation['category_name']
        instance_token = annotation['instance_token']
        if len(annotation['attribute_tokens']):
            attribute = nusc.get('attribute', annotation['attribute_tokens'][0])['name']
        else:
            continue
        if 'pedestrian' in category and not 'stroller' in category and not 'wheelchair' in category:
            pass
        elif 'vehicle' in category and 'bicycle' not in category and 'motorcycle' not in category and 'parked' not in attribute:
            pos = [annotation['translation'][0], annotation['translation'][1]]
            heading = Quaternion(annotation['rotation']).yaw_pitch_roll[0]
            size = annotation['size']
            agent_occs[instance_token] = (pos, heading, size)
        else:
            continue
    agent_init_occ = agent_occs
    return map_pose, ego_occ, agent_init_occ


def get_agent_occs_from_scene(nusc: NuScenes, scene_token: str):
    map_poses = []
    agent_init_occs = []  # [{ann_token: (pos, heading, size)}]
    ego_occs = []  # [(pos, heading, size)]

    # For each sample in the scene, store the ego pose.
    sample_tokens = nusc.field2token('sample', 'scene_token', scene_token)
    for sample_token in sample_tokens:
        map_pose, ego_occ, agent_init_occ = get_agent_occs_from_sample(nusc, sample_token)
        map_poses.append(map_pose)
        ego_occs.append(ego_occ)
        agent_init_occs.append(agent_init_occ)
    return map_poses, agent_init_occs, ego_occs


def get_single_agent_occs(agent_init_occs, ann_token: str):
    future_occ = {}
    for timestep in range(len(agent_init_occs)):
        occs = agent_init_occs[timestep]
        for token in occs.keys():
            if token == ann_token:
                future_occ[timestep] = occs[token]
    return future_occ


def plot_agent_future(ax: plt.Axes, agent_init_occs, ann_token: str, cur_time_step: int):
    agent_occs = get_single_agent_occs(agent_init_occs, ann_token)
    agent_future_pos = []
    time_step = cur_time_step
    horizon_length = 6
    while time_step in agent_occs.keys() and time_step < cur_time_step + horizon_length:
        agent_future_pos.append([agent_occs[time_step][0][0], agent_occs[time_step][0][1]])
        time_step += 1
    if len(agent_future_pos) > 1:
        agent_future_pos = np.array(agent_future_pos)
        ax.plot(agent_future_pos[:, 0], agent_future_pos[:, 1],
                'ro-',
                linewidth=0.5, markersize=1.0, alpha=0.5)


def plot_map_patch(nusc_map: NuScenesMap, map_poses: np.ndarray):
    # Settings
    patch_margin = 2
    min_diff_patch = 30
    # Render the map patch with the current ego poses.
    min_patch = np.floor(map_poses.min(axis=0) - patch_margin)
    max_patch = np.ceil(map_poses.max(axis=0) + patch_margin)
    diff_patch = max_patch - min_patch
    if any(diff_patch < min_diff_patch):
        center_patch = (min_patch + max_patch) / 2
        diff_patch = np.maximum(diff_patch, min_diff_patch)
        min_patch = center_patch - diff_patch / 2
        max_patch = center_patch + diff_patch / 2
    my_patch = (min_patch[0], min_patch[1], max_patch[0], max_patch[1])
    fig, ax = nusc_map.render_map_patch(my_patch, nusc_map.non_geometric_layers, figsize=(10, 10),
                                        alpha=0.3,
                                        render_egoposes_range=True,
                                        render_legend=True)
    return fig, ax


def plot_agent_occ(ax: plt.Axes, pos: List[float], heading: float, size: List[float]):
    x = pos[0]
    y = pos[1]
    width = size[1]
    height = size[0]
    occ = patches.Rectangle((x, y), width / 2, height / 2, angle=np.rad2deg(heading), linewidth=1,
                            edgecolor='r', alpha=0.5, facecolor='none')
    ax.add_patch(occ)
    length = 2
    ax.arrow(x, y, length * np.cos(heading), length * np.sin(heading), head_width=1)


def plot_ego_occ_and_future(ax: plt.Axes, pos: List[float], heading: float, size: List[float], ego_future: np.ndarray):
    x = pos[0]
    y = pos[1]
    width = size[1]
    height = size[0]
    occ = patches.Rectangle((x, y), width / 2, height / 2, angle=np.rad2deg(heading), linewidth=1,
                            edgecolor='r', alpha=0.5, facecolor='none')
    ax.add_patch(occ)
    length = 2
    ax.arrow(x, y, length * np.cos(heading), length * np.sin(heading), head_width=1)

    ax.plot(ego_future[:, 0], ego_future[:, 1], 'ro-', markersize=1.0, alpha=1.0, zorder=2)


def plot_agents_and_ego_on_map_at_time(nusc: NuScenes,
                                       nusc_map: NuScenesMap,
                                       scene_token: str,
                                       time_step: int):
    map_poses, agent_init_occs, ego_occs = get_agent_occs_from_scene(nusc, scene_token)

    map_poses = np.vstack(map_poses)[:, :2]
    fig, ax = plot_map_patch(nusc_map, map_poses)

    ego_occ = ego_occs[time_step]
    future_length = 6
    future_end = min(len(ego_occs), time_step + future_length)
    ego_future = map_poses[time_step:future_end, :]
    plot_ego_occ_and_future(ax, pos=ego_occ[0], heading=ego_occ[1], size=ego_occ[2], ego_future=ego_future)
    for ann_token, occ in agent_init_occs[time_step].items():
        plot_agent_occ(ax, pos=occ[0], heading=occ[1], size=occ[2])
        plot_agent_future(ax, agent_init_occs, ann_token, time_step)

    plt.axis('off')
    return fig, ax


def get_ego_pose_at_time(nusc: NuScenes, scene_token: str, time_step: int):
    map_poses, agent_init_occs, ego_occs = get_agent_occs_from_scene(nusc, scene_token)
    ego_pos = map_poses[time_step][:2]
    ego_heading = ego_occs[time_step][1]
    return ego_pos, ego_heading


def render_agents_and_ego_on_map(nusc: NuScenes,
                                 nusc_map: NuScenesMap,
                                 scene_tokens: List = None,
                                 verbose: bool = True,
                                 out_path: str = None):
    # Ids of scenes with a bad match between localization and map.
    scene_blacklist = [499, 515, 517]

    # Get logs by location.
    log_location = nusc_map.map_name
    log_tokens = [l['token'] for l in nusc.log if l['location'] == log_location]
    assert len(log_tokens) > 0, 'Error: This split has 0 scenes for location %s!' % log_location

    # Filter scenes.
    scene_tokens_location = [e['token'] for e in nusc.scene if e['log_token'] in log_tokens]
    if scene_tokens is not None:
        scene_tokens_location = [t for t in scene_tokens_location if t in scene_tokens]
    assert len(scene_tokens_location) > 0, 'Error: Found 0 valid scenes for location %s!' % log_location

    map_poses = []
    agent_init_occs = []  # [{ann_token: (pos, heading, size)}]
    future_length = 6
    ego_occs = []  # [(pos, heading, size)]
    if verbose:
        print('Adding ego poses to map...')
    for scene_token in tqdm(scene_tokens_location, disable=not verbose):
        # Check that the scene is from the correct location.
        scene_record = nusc.get('scene', scene_token)
        scene_name = scene_record['name']
        scene_id = int(scene_name.replace('scene-', ''))
        log_record = nusc.get('log', scene_record['log_token'])
        assert log_record['location'] == log_location, \
            'Error: The provided scene_tokens do not correspond to the provided map location!'

        # Print a warning if the localization is known to be bad.
        if verbose and scene_id in scene_blacklist:
            print('Warning: %s is known to have a bad fit between ego pose and map.' % scene_name)

        # For each sample in the scene, store the ego pose.
        sample_tokens = nusc.field2token('sample', 'scene_token', scene_token)
        for sample_token in sample_tokens:
            map_pose, ego_occ, agent_init_occ = get_agent_occs_from_sample(nusc, sample_token)
            map_poses.append(map_pose)
            ego_occs.append(ego_occ)
            agent_init_occs.append(agent_init_occ)

    # Check that ego poses aren't empty.
    assert len(ego_occs) > 0, 'Error: Found 0 ego poses. Please check the inputs.'

    # Compute number of close ego poses.
    if verbose:
        print('Creating plot...')
    map_poses = np.vstack(map_poses)[:, :2]

    # Plot in the same axis as the map.
    # Make sure these are plotted "on top".
    for time_step in range(len(ego_occs)):
        fig, ax = plot_map_patch(nusc_map, map_poses)
        ego_occ = ego_occs[time_step]
        future_end = min(len(ego_occs), time_step + future_length)
        ego_future = map_poses[time_step:future_end, :]
        plot_ego_occ_and_future(ax, pos=ego_occ[0], heading=ego_occ[1], size=ego_occ[2], ego_future=ego_future)
        for _, occ in agent_init_occs[time_step].items():
            plot_agent_occ(ax, pos=occ[0], heading=occ[1], size=occ[2])

        plt.axis('off')
        if out_path is not None:
            plt.savefig(out_path + '_{}.png'.format(time_step), bbox_inches='tight', pad_inches=0)


def get_nusc(root_path: str):
    nusc = NuScenes(version='v1.0-mini', dataroot=root_path, verbose=False)
    return nusc


def get_nusc_map(root_path: str, map_name: str = 'boston-seaport'):
    nusc_map = NuScenesMap(dataroot=root_path, map_name=map_name)
    return nusc_map


def get_scene_token(nusc: NuScenes, scene_name: str):
    for scene in nusc.scene:
        if scene['name'] == scene_name:
            return scene['token']
    print('Error: Scene %s not found in NuScenes database.' % scene_name)
    return None


def main():
    root_path = '/home/ysli/Desktop/dis-constrained-planning/predictors/Trajectron-plus-plus/experiments/nuScenes/v1.0-mini'
    nusc = get_nusc(root_path)
    scene_name = 'scene-0796'
    scene_token = get_scene_token(nusc, scene_name)
    scene = nusc.get('scene', scene_token)
    log_token = scene['log_token']
    log = nusc.get('log', log_token)
    location = log['location']
    nusc_map = get_nusc_map(root_path, location)
    if not os.path.exists(root_path + '/scene_figs/'+scene_name+'/'):
        os.makedirs(root_path + '/scene_figs/'+scene_name)
    render_agents_and_ego_on_map(nusc, nusc_map, scene_tokens=[scene_token], verbose=False,
                                 out_path=root_path + '/scene_figs/'+scene_name+'/')


if __name__ == '__main__':
    main()
