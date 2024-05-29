import imageio
import matplotlib.pyplot as plt
import os
import argparse

def create_gif_from_saved_figs(fig_dir: str, t_start: int, t_end: int):
    timesteps = list(range(t_start, t_end))
    with imageio.get_writer(fig_dir+'/pred.gif', mode='I', duration=1.0) as writer:
        for t in timesteps:
            filename = fig_dir + '/pred_' + str(t) + '.png'
            image = imageio.imread(filename)
            plt.imshow(image)
            plt.text(0, 0, 'timestep '+str(t))

            writer.append_data(image)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, required=True)
    args = parser.parse_args()
    # log_dir = '/home/ysli/Desktop/dis-constrained-planning/predictors/Trajectron-plus-plus/experiments/nuScenes/models'
    create_gif_from_saved_figs(args.log_dir, t_start=8, t_end=20)
