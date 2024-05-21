from utils import prediction_output_to_trajectories
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns


def plot_trajectories(ax,
                      prediction_dict,
                      histories_dict,
                      futures_dict,
                      line_alpha=0.7,
                      line_width=0.2,
                      edge_width=2,
                      circle_edge_width=0.5,
                      node_circle_size=1,
                      batch_num=0,
                      kde=False,
                      pos_local_to_map=None,
                      angle_local_to_map=None):
    cmap = ['k', 'b', 'y', 'g', 'r']
    for node in histories_dict:
        history = histories_dict[node]
        future = futures_dict[node]
        predictions = prediction_dict[node]
        if pos_local_to_map is not None and angle_local_to_map is not None:
            if len(history) > 0:
                history[:, 0] += pos_local_to_map[0]
                history[:, 1] += pos_local_to_map[1]
            if len(future) > 0:
                future[:, 0] += pos_local_to_map[0]
                future[:, 1] += pos_local_to_map[1]
            if len(predictions) > 0:
                predictions[:, :, :, 0] += pos_local_to_map[0]
                predictions[:, :, :, 1] += pos_local_to_map[1]

        if np.isnan(history[-1]).any():
            continue
        ax.plot(history[:, 0], history[:, 1], 'k--', zorder=10)
        for sample_num in range(prediction_dict[node].shape[1]):

            if kde and predictions.shape[1] >= 50:
                line_alpha = 0.2
                for t in range(predictions.shape[2]):
                    sns.kdeplot(predictions[batch_num, :, t, 0], predictions[batch_num, :, t, 1],
                                ax=ax, shade=True, shade_lowest=False,
                                color=np.random.choice(cmap), alpha=0.8)

            ax.plot(predictions[batch_num, sample_num, :, 0], predictions[batch_num, sample_num, :, 1],
                    color=cmap[node.type.value],
                    linewidth=line_width, alpha=line_alpha)

            ax.plot(future[:, 0],
                    future[:, 1],
                    'w--',
                    path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()])
            # Current Node Position
            circle = plt.Circle((history[-1, 0],
                                 history[-1, 1]),
                                node_circle_size,
                                facecolor='g',
                                edgecolor='k',
                                lw=circle_edge_width,
                                zorder=3)
            ax.add_artist(circle)

    # ax.axis('equal')


def visualize_prediction(ax,
                         prediction_output_dict,
                         dt,
                         max_hl,
                         ph,
                         robot_node=None,
                         map=None,
                         pos_local_to_map=None,
                         angle_local_to_map=None,
                         **kwargs):
    prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(prediction_output_dict,
                                                                                      dt,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)

    assert (len(prediction_dict.keys()) <= 1)
    if len(prediction_dict.keys()) == 0:
        return
    ts_key = list(prediction_dict.keys())[0]

    prediction_dict = prediction_dict[ts_key]
    histories_dict = histories_dict[ts_key]
    futures_dict = futures_dict[ts_key]

    if map is not None:
        ax.imshow(map.as_image(), origin='lower', alpha=0.3, zorder=0)
    plot_trajectories(ax, prediction_dict, histories_dict, futures_dict, pos_local_to_map=pos_local_to_map,
                      angle_local_to_map=angle_local_to_map, *kwargs)


def visualize_distribution(ax,
                           prediction_distribution_dict,
                           map=None,
                           pi_threshold=0.05,
                           pos_local_to_map=None,
                           angle_local_to_map=None,
                           **kwargs):
    if map is not None:
        ax.imshow(map.as_image(), origin='lower', alpha=0.3)

    for node, pred_dist in prediction_distribution_dict.items():
        if pred_dist.mus.shape[:2] != (1, 1):
            return
        means = pred_dist.mus.squeeze().cpu().numpy()
        covs = pred_dist.get_covariance_matrix().squeeze().cpu().numpy()
        pis = pred_dist.pis_cat_dist.probs.squeeze().cpu().numpy()
        is_z_mode = len(means.shape) == 2
        num_z = 1 if is_z_mode else means.shape[1]
        if pred_dist.mus.shape[:2] == (1, 1):
            for timestep in range(means.shape[0]):
                for z_val in range(num_z):
                    if is_z_mode:
                        mean = means[timestep, :]
                        covar = covs[timestep, :]
                        pi = pis[timestep]
                    else:
                        mean = means[timestep, z_val]
                        covar = covs[timestep, z_val]
                        pi = pis[timestep, z_val]

                    if pi < pi_threshold:
                        continue

                    v, w = linalg.eigh(covar)
                    v = 2. * np.sqrt(2.) * np.sqrt(v)
                    u = w[0] / linalg.norm(w[0])

                    # Plot an ellipse to show the Gaussian component
                    angle = np.arctan(u[1] / u[0])
                    angle = 180. * angle / np.pi  # convert to degrees
                    if pos_local_to_map is None or angle_local_to_map is None:
                        ell = patches.Ellipse(mean, v[0], v[1], 180. + angle,
                                              color='blue' if node.type.name == 'VEHICLE' else 'orange')
                    else:
                        ell = patches.Ellipse(mean + pos_local_to_map, v[0], v[1], 180. + angle + angle_local_to_map,
                                              color='blue' if node.type.name == 'VEHICLE' else 'orange')
                    ell.set_edgecolor(None)
                    ell.set_clip_box(ax.bbox)
                    ell.set_alpha(pi / 2)
                    ax.add_artist(ell)
