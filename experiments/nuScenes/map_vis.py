import matplotlib.pyplot as plt
import tqdm
import numpy as np
from typing import List
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm
from pyquaternion import Quaternion
import matplotlib.patches as patches


def render_agents_and_ego_on_map(nusc: NuScenes,
                                 nusc_map: NuScenesMap,
                                 scene_tokens: List = None,
                                 verbose: bool = True,
                                 out_path: str = None,
                                 render_egoposes: bool = True,
                                 render_egoposes_range: bool = True,
                                 render_legend: bool = True):
    # Settings
    patch_margin = 2
    min_diff_patch = 30

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
            sample_record = nusc.get('sample', sample_token)

            # Poses are associated with the sample_data. Here we use the lidar sample_data.
            sample_data_record = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
            pose_record = nusc.get('ego_pose', sample_data_record['ego_pose_token'])
            # Calculate the pose on the map and append.
            pos = [pose_record['translation'][0], pose_record['translation'][1]]
            heading = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            size = [2, 4]
            ego_occs.append((pos, heading, size))
            map_poses.append(pose_record['translation'])

            # retrieve poses of other agents(pose and size in sample annotation)
            annotation_tokens = sample_record['anns']
            agent_occs = {}
            for annotation_token in annotation_tokens:
                annotation = nusc.get('sample_annotation', annotation_token)
                category = annotation['category_name']

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
                    agent_occs[annotation_token] = (pos, heading, size)
                else:
                    continue
            agent_init_occs.append(agent_occs)

    # Check that ego poses aren't empty.
    assert len(ego_occs) > 0, 'Error: Found 0 ego poses. Please check the inputs.'

    # Compute number of close ego poses.
    if verbose:
        print('Creating plot...')
    map_poses = np.vstack(map_poses)[:, :2]

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

    # Plot in the same axis as the map.
    # Make sure these are plotted "on top".
    for time_step in range(len(ego_occs)):
        fig, ax = nusc_map.render_map_patch(my_patch, nusc_map.non_geometric_layers, figsize=(10, 10),
                                            alpha=0.3,
                                            render_egoposes_range=render_egoposes_range,
                                            render_legend=render_legend)
        ego_occ = ego_occs[time_step]
        x = ego_occ[0][0]
        y = ego_occ[0][1]
        width = ego_occ[2][1]
        height = ego_occ[2][0]
        heading = ego_occ[1]
        occ = patches.Rectangle((x, y), width / 2, height / 2, angle=np.rad2deg(heading), linewidth=1,
                                edgecolor='r', alpha=0.5, facecolor='none')
        ax.add_patch(occ)
        length = 2
        ax.arrow(x, y, length * np.cos(heading), length * np.sin(heading), head_width=1)

        future_end = min(len(ego_occs), time_step+future_length)
        ax.plot(map_poses[time_step:future_end, 0], map_poses[time_step:future_end, 1], 'k-.', alpha=1.0, zorder=2)

        for _, occ in agent_init_occs[time_step].items():
            x = occ[0][0]
            y = occ[0][1]
            width = occ[2][1]
            height = occ[2][0]
            heading = occ[1]
            occ = patches.Rectangle((x, y), width / 2, height / 2, angle=np.rad2deg(heading), linewidth=1,
                                    edgecolor='r', alpha=0.5, facecolor='none')
            ax.add_patch(occ)
            length = 2
            ax.arrow(x, y, length * np.cos(heading), length * np.sin(heading), head_width=1)

        plt.axis('off')
        if out_path is not None:
            plt.savefig(out_path+'_{}.png'.format(time_step), bbox_inches='tight', pad_inches=0)


nusc = NuScenes(version='v1.0-mini', dataroot='./v1.0-mini', verbose=False)

# Render ego poses.
nusc_map_bos = NuScenesMap(dataroot='./v1.0-mini', map_name='boston-seaport')
for scene in nusc.scene:
    if scene['name'] == 'scene-0103':
        token = scene['token']
        render_agents_and_ego_on_map(nusc, nusc_map_bos, scene_tokens=[token], verbose=False,
                                     out_path='./')
        # out_path = './scene-0103.avi'
        # nusc.render_scene(scene['token'], out_path=out_path)
