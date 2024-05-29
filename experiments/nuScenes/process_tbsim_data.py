import sys
import os
import numpy as np
import pandas as pd
import dill
import argparse
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.map_expansion.map_api import NuScenesMap
from kalman_filter import NonlinearKinematicBicycle
from sklearn.model_selection import train_test_split
import h5py
from typing import List
sys.path.append("../../trajectron")
from environment import Environment, Scene, Node, GeometricMap, derivative_of

FREQUENCY = 2
dt = 1 / FREQUENCY
data_columns_vehicle = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']])
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_tuples([('heading', '°'), ('heading', 'd°')]))
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']]))

data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

curv_0_2 = 0
curv_0_1 = 0
total = 0

standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }
    },
    'VEHICLE': {
        'position': {
            'x': {'mean': 0, 'std': 80},
            'y': {'mean': 0, 'std': 80}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 15},
            'y': {'mean': 0, 'std': 15},
            'norm': {'mean': 0, 'std': 15}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 4},
            'y': {'mean': 0, 'std': 4},
            'norm': {'mean': 0, 'std': 4}
        },
        'heading': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            '°': {'mean': 0, 'std': np.pi},
            'd°': {'mean': 0, 'std': 1}
        }
    }
}


def load_data_from_file(data_path: str):
    file_name = 'data.hdf5'
    f = h5py.File(data_path+file_name, 'r')
    # for key in f.keys():
    #     print('key', key)
    #     dataset = f[key]
    #     action_positions = dataset['action_positions']
    #     action_yaws = dataset['action_yaws']
    #     yaw = dataset['yaw']
    #     centroid = dataset['centroid']
    #     curr_speed = dataset['curr_speed']

    return f


def split_dataset(data_names: List[str]):
    train_data_names, test_data_names = train_test_split(data_names, test_size=0.1, random_state=0)
    split_data = {}
    split_data['train'] = train_data_names
    split_data['test'] = test_data_names
    return split_data


def augment_scene(scene, angle):
    def rotate_pc(pc, alpha):
        M = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        return M @ pc

    data_columns_vehicle = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']])
    data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_tuples([('heading', '°'), ('heading', 'd°')]))
    data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']]))

    data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name, non_aug_scene=scene)

    alpha = angle * np.pi / 180

    for node in scene.nodes:
        if node.type == 'PEDESTRIAN':
            x = node.data.position.x.copy()
            y = node.data.position.y.copy()

            x, y = rotate_pc(np.array([x, y]), alpha)

            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            ax = derivative_of(vx, scene.dt)
            ay = derivative_of(vy, scene.dt)

            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay}

            node_data = pd.DataFrame(data_dict, columns=data_columns_pedestrian)

            node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep)
        elif node.type == 'VEHICLE':
            x = node.data.position.x.copy()
            y = node.data.position.y.copy()

            heading = getattr(node.data.heading, '°').copy()
            heading += alpha
            heading = (heading + np.pi) % (2.0 * np.pi) - np.pi

            x, y = rotate_pc(np.array([x, y]), alpha)

            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            ax = derivative_of(vx, scene.dt)
            ay = derivative_of(vy, scene.dt)

            v = np.stack((vx, vy), axis=-1)
            v_norm = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1, keepdims=True)
            heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1.))
            heading_x = heading_v[:, 0]
            heading_y = heading_v[:, 1]

            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('velocity', 'norm'): np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1),
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay,
                         ('acceleration', 'norm'): np.linalg.norm(np.stack((ax, ay), axis=-1), axis=-1),
                         ('heading', 'x'): heading_x,
                         ('heading', 'y'): heading_y,
                         ('heading', '°'): heading,
                         ('heading', 'd°'): derivative_of(heading, dt, radian=True)}

            node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)

            node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep,
                        non_aug_node=node)

        scene_aug.nodes.append(node)
    return scene_aug


def augment(scene):
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    scene_aug.map = scene.map
    return scene_aug


def trajectory_curvature(t):
    path_distance = np.linalg.norm(t[-1] - t[0])

    lengths = np.sqrt(np.sum(np.diff(t, axis=0) ** 2, axis=1))  # Length between points
    path_length = np.sum(lengths)
    if np.isclose(path_distance, 0.):
        return 0, 0, 0
    return (path_length / path_distance) - 1, path_length, path_distance


def process_scene(raw_data, data_name, map_path, map_name, env):
    scene_id = int(data_name.replace('scene-', ''))
    data = pd.DataFrame(columns=['frame_id',
                                 'type',
                                 'node_id',
                                 'robot',
                                 'x', 'y', 'z',
                                 'length',
                                 'width',
                                 'height',
                                 'heading'])

    num_frame = 40
    frame_id = 0
    data_at_scene = raw_data[data_name]
    while frame_id < num_frame:
        num_agents = 10
        for agent_id in range(1, num_agents):
            pos = data_at_scene['centroid'][agent_id, frame_id, :]
            heading = data_at_scene['yaw'][agent_id, frame_id]
            our_category = env.NodeType.VEHICLE
            node_id = data_name+'_'+str(agent_id)
            data_point = pd.Series({'frame_id': frame_id,
                                    'type': our_category,
                                    'node_id': node_id,
                                    'robot': False,
                                    'x': pos[0],
                                    'y': pos[1],
                                    'z': 0.0,
                                    'length': 4.0,
                                    'width': 2.0,
                                    'height': 1.8,
                                    'heading': heading})
            data = data.append(data_point, ignore_index=True)

        # Ego Vehicle
        our_category = env.NodeType.VEHICLE
        ego_id = 0
        pos = data_at_scene['centroid'][ego_id, frame_id, :]
        heading = data_at_scene['yaw'][ego_id, frame_id]
        data_point = pd.Series({'frame_id': frame_id,
                                'type': our_category,
                                'node_id': 'ego',
                                'robot': True,
                                'x': pos[0],
                                'y': pos[1],
                                'z': 0.0,
                                'length': 4,
                                'width': 1.7,
                                'height': 1.5,
                                'heading': heading,
                                'orientation': None})
        data = data.append(data_point, ignore_index=True)

        frame_id += 1

    if len(data.index) == 0:
        return None

    data.sort_values('frame_id', inplace=True)
    max_timesteps = data['frame_id'].max()

    x_min = np.round(data['x'].min() - 50)
    x_max = np.round(data['x'].max() + 50)
    y_min = np.round(data['y'].min() - 50)
    y_max = np.round(data['y'].max() + 50)

    data['x'] = data['x'] - x_min
    data['y'] = data['y'] - y_min

    scene = Scene(timesteps=max_timesteps + 1, dt=dt, name=str(scene_id), aug_func=augment)

    # Generate Maps
    nusc_map = NuScenesMap(dataroot=map_path, map_name=map_name)

    type_map = dict()
    x_size = x_max - x_min
    y_size = y_max - y_min
    patch_box = (x_min + 0.5 * (x_max - x_min), y_min + 0.5 * (y_max - y_min), y_size, x_size)
    patch_angle = 0  # Default orientation where North is up
    canvas_size = (np.round(3 * y_size).astype(int), np.round(3 * x_size).astype(int))
    homography = np.array([[3., 0., 0.], [0., 3., 0.], [0., 0., 3.]])
    layer_names = ['lane', 'road_segment', 'drivable_area', 'road_divider', 'lane_divider', 'stop_line',
                   'ped_crossing', 'stop_line', 'ped_crossing', 'walkway']
    map_mask = (nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size) * 255.0).astype(
        np.uint8)
    map_mask = np.swapaxes(map_mask, 1, 2)  # x axis comes first
    # PEDESTRIANS
    map_mask_pedestrian = np.stack((map_mask[9], map_mask[8], np.max(map_mask[:3], axis=0)), axis=0)
    type_map['PEDESTRIAN'] = GeometricMap(data=map_mask_pedestrian, homography=homography, description=', '.join(layer_names))
    # VEHICLES
    map_mask_vehicle = np.stack((np.max(map_mask[:3], axis=0), map_mask[3], map_mask[4]), axis=0)
    type_map['VEHICLE'] = GeometricMap(data=map_mask_vehicle, homography=homography, description=', '.join(layer_names))

    map_mask_plot = np.stack(((np.max(map_mask[:3], axis=0) - (map_mask[3] + 0.5 * map_mask[4]).clip(
        max=255)).clip(min=0).astype(np.uint8), map_mask[8], map_mask[9]), axis=0)
    type_map['VISUALIZATION'] = GeometricMap(data=map_mask_plot, homography=homography, description=', '.join(layer_names))

    scene.map = type_map
    del map_mask
    del map_mask_pedestrian
    del map_mask_vehicle
    del map_mask_plot

    for node_id in pd.unique(data['node_id']):
        node_frequency_multiplier = 1
        node_df = data[data['node_id'] == node_id]

        if node_df['x'].shape[0] < 2:
            continue

        if not np.all(np.diff(node_df['frame_id']) == 1):
            # print('Occlusion')
            continue  # TODO Make better

        node_values = node_df[['x', 'y']].values
        x = node_values[:, 0]
        y = node_values[:, 1]
        heading = node_df['heading'].values
        if node_df.iloc[0]['type'] == env.NodeType.VEHICLE and not node_id == 'ego':
            # Kalman filter Agent
            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            velocity = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1)

            filter_veh = NonlinearKinematicBicycle(dt=scene.dt, sMeasurement=1.0)
            P_matrix = None
            for i in range(len(x)):
                if i == 0:  # initalize KF
                    # initial P_matrix
                    P_matrix = np.identity(4)
                elif i < len(x):
                    # assign new est values
                    x[i] = x_vec_est_new[0][0]
                    y[i] = x_vec_est_new[1][0]
                    heading[i] = x_vec_est_new[2][0]
                    velocity[i] = x_vec_est_new[3][0]

                if i < len(x) - 1:  # no action on last data
                    # filtering
                    x_vec_est = np.array([[x[i]],
                                          [y[i]],
                                          [heading[i]],
                                          [velocity[i]]])
                    z_new = np.array([[x[i + 1]],
                                      [y[i + 1]],
                                      [heading[i + 1]],
                                      [velocity[i + 1]]])
                    x_vec_est_new, P_matrix_new = filter_veh.predict_and_update(
                        x_vec_est=x_vec_est,
                        u_vec=np.array([[0.], [0.]]),
                        P_matrix=P_matrix,
                        z_new=z_new
                    )
                    P_matrix = P_matrix_new

            curvature, pl, _ = trajectory_curvature(np.stack((x, y), axis=-1))
            if pl < 1.0:  # vehicle is "not" moving
                x = x[0].repeat(max_timesteps + 1)
                y = y[0].repeat(max_timesteps + 1)
                heading = heading[0].repeat(max_timesteps + 1)
            global total
            global curv_0_2
            global curv_0_1
            total += 1
            if pl > 1.0:
                if curvature > .2:
                    curv_0_2 += 1
                    node_frequency_multiplier = 3*int(np.floor(total/curv_0_2))
                elif curvature > .1:
                    curv_0_1 += 1
                    node_frequency_multiplier = 3*int(np.floor(total/curv_0_1))

        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)
        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)

        if node_df.iloc[0]['type'] == env.NodeType.VEHICLE:
            v = np.stack((vx, vy), axis=-1)
            v_norm = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1, keepdims=True)
            heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1.))
            heading_x = heading_v[:, 0]
            heading_y = heading_v[:, 1]

            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('velocity', 'norm'): np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1),
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay,
                         ('acceleration', 'norm'): np.linalg.norm(np.stack((ax, ay), axis=-1), axis=-1),
                         ('heading', 'x'): heading_x,
                         ('heading', 'y'): heading_y,
                         ('heading', '°'): heading,
                         ('heading', 'd°'): derivative_of(heading, dt, radian=True)}
            node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)
        else:
            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay}
            node_data = pd.DataFrame(data_dict, columns=data_columns_pedestrian)

        node = Node(node_type=node_df.iloc[0]['type'], node_id=node_id, data=node_data, frequency_multiplier=node_frequency_multiplier)
        node.first_timestep = node_df['frame_id'].iloc[0]
        if node_df.iloc[0]['robot'] == True:
            node.is_robot = True
            scene.robot = node

        scene.nodes.append(node)

    return scene


def process_data(data_path, map_path, map_name, output_path):
    # read from saved data, split to training and testing data
    raw_data = load_data_from_file(data_path)
    data_name_list = [key for key in raw_data.keys()]
    split_data = split_dataset(data_name_list)

    for data_class in ['train', 'test']:
        env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'], standardization=standardization)
        attention_radius = dict()
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE)] = 20.0
        attention_radius[(env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN)] = 20.0
        attention_radius[(env.NodeType.VEHICLE, env.NodeType.VEHICLE)] = 30.0

        env.attention_radius = attention_radius
        env.robot_type = env.NodeType.VEHICLE
        scenes = []

        for data_name in split_data[data_class]:
            scene = process_scene(raw_data, data_name, map_path, map_name, env)
            if scene is not None:
                if data_class == 'train':
                    scene.augmented = list()
                    angles = np.arange(0, 360, 15)
                    for angle in angles:
                        scene.augmented.append(augment_scene(scene, angle))
                scenes.append(scene)

        print(f'Processed {len(scenes):.2f} scenes')

        env.scenes = scenes

        if len(scenes) > 0:
            mini_string = '_tbsim_scene0103'
            data_dict_path = os.path.join(output_path, 'nuScenes_' + data_class + mini_string + '_full.pkl')
            with open(data_dict_path, 'wb') as f:
                dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
            print('Saved Environment!')

        global total
        global curv_0_2
        global curv_0_1
        print(f"Total Nodes: {total}")
        print(f"Curvature > 0.1 Nodes: {curv_0_1}")
        print(f"Curvature > 0.2 Nodes: {curv_0_2}")
        total = 0
        curv_0_1 = 0
        curv_0_2 = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--map_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--map_name', type=str, required=True)
    args = parser.parse_args()
    process_data(args.data_path, args.map_path, args.map_name, args.output_path)
