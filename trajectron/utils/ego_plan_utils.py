import numpy as np


def get_ego_plan_noisy(ego_plan_original: np.ndarray):
    modified_ego_plan = ego_plan_original
    modified_ego_plan += np.random.normal(0, 0.2, size=(7, 8))
    return modified_ego_plan


def get_ego_plan_stop(ego_plan_original: np.ndarray):
    init_state = ego_plan_original[0, :]
    init_state[[2, 3, 4, 5, 7]] = 0
    init_state = init_state.reshape([1, 8])
    modified_ego_plan = np.repeat(init_state, 7, axis=0)
    return modified_ego_plan


def get_ego_plan_const_vel(ego_plan_original: np.ndarray):
    cur_state = np.copy(ego_plan_original[0, :])
    modified_ego_plan = [np.copy(cur_state)]
    velocity = cur_state[2:4]
    dt = 0.5
    for timestep in range(6):
        cur_state[:2] += velocity*dt
        modified_ego_plan.append(np.copy(cur_state))
    return np.array(modified_ego_plan)


