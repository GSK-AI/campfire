###
###
import argparse
import pathlib

import numpy as np 
import pandas as pd
import yaml

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


def main(config) -> None:
    """
    Main Function
    """
    seed = config["seed"]
    model_name = config["model_name"]

    #load probing_df
    linear_probe_data_dir = config["linear_probe_data_dir"]
    plot_save_dir = config["plot_save_dir"]
    plot_data = pd.read_csv(linear_probe_data_dir+'linear_probing_'+str(model_name)+'.csv')

    splits = ['train','val','test']
    markers = ['o','x','^']

    feature_cols = [col for col in plot_data.columns if col.startswith('embedding')]
    x = plot_data[feature_cols].values
    y = plot_data['TARGET'].values

    pca = PCA(n_components=2)
    pca_x = pca.fit_transform(x)

    cmap = plt.cm.get_cmap('Spectral_r', len(np.unique(y)))
    plt.style.use('bmh')

    for split,mark in zip(splits,markers):
        split_data = plot_data.loc[plot_data['split']==split]
        feature_cols = [col for col in split_data.columns if col.startswith('embedding')]
        x = split_data[feature_cols].values
        y = split_data['TARGET'].values

        pca_x = pca.transform(x)


        plt.scatter(pca_x[:,0], pca_x[:,1], c=y, cmap=cmap,marker=mark,label=split)



    plt.xlabel('PC_1')
    plt.ylabel('PC_2')
    plt.title('PCA Explained Variance: {}%'.format(np.round(np.sum(pca.explained_variance_ratio_),3)*100))
    plt.colorbar(ticks=range(len(np.unique(y))), label='Target')
    legend1 = plt.legend()
    for handle in legend1.legendHandles:
        handle.set_facecolor('black')
        handle.set_edgecolor('black')
    plt.savefig(plot_save_dir+model_name+'_pca_embed.png',dpi=500, bbox_inches='tight')




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, default="config_run.yaml", help="Config file"
    )

    args = parser.parse_args()

    yaml_file = open(f"{args.config}")
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    main(config)
