import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import matplotlib.patches as mpatches


def main(config) -> None:
    """
    Main Function:

    Load subdirectories containing feature csvs 



    """
    test_metric_dirs = config["test_metric_dirs"]
    test_out_metric_dirs = config["test_out_metric_dirs"]
    model_names = config["model_names"]
    plot_save_dir = config["plot_save_dir"]

    test_data_to_plot = []
    test_out_data_to_plot = []

    for metric_dir in test_metric_dirs:

        with open(metric_dir, 'rb') as handle:
            metrics = pickle.load(handle)

        Nruns = len(metrics)

        acc = np.zeros(Nruns)

        for i in range(Nruns):
            acc[i] = metrics[i]["accuracy"]

        # Create a list of these arrays
        test_data_to_plot.append(acc)

    for metric_dir in test_out_metric_dirs:

        with open(metric_dir, 'rb') as handle:
            metrics = pickle.load(handle)

        Nruns = len(metrics)

        acc = np.zeros(Nruns)

        for i in range(Nruns):
            acc[i] = metrics[i]["accuracy"]

        # Create a list of these arrays
        test_out_data_to_plot.append(acc)

    

    plt.style.use('bmh')
    # Create a figure instance
    fig = plt.figure()

    # Create an axes instance
    ax = fig.add_axes([0,0,1,1])

    ax.set_xticklabels(model_names)

    # Create the boxplot
    bp = ax.boxplot(test_data_to_plot)

    plt.ylabel('Accuracy (Plate: Seen || Channels: Seen)')

    # plt.ylim(0.25,0.5)
    # plt.axhline(1/9)
    # Show the plot
    plt.tight_layout()
    plt.savefig(plot_save_dir+'test_model_comparisons.png',dpi=500, bbox_inches='tight')

    plt.style.use('bmh')
    # Create a figure instance
    fig = plt.figure()

    # Create an axes instance
    ax = fig.add_axes([0,0,1,1])

    ax.set_xticklabels(model_names)

    # Create the boxplot
    bp = ax.boxplot(test_out_data_to_plot)

    plt.ylabel('Accuracy (Plate: UnSeen || Channels: Seen)')

    # plt.ylim(0.25,0.5)
    # plt.axhline(1/9)
    # Show the plot
    plt.tight_layout()
    plt.savefig(plot_save_dir+'test_out_model_comparisons.png',dpi=500, bbox_inches='tight')


    Nmodels = len(test_data_to_plot)
    models = []
    for i in range(Nmodels):
        models.append(np.array([test_data_to_plot[i],test_out_data_to_plot[i]]))

    fig, ax = plt.subplots()

    # Colors for the two arrays
    colors = ['darkorange', 'dodgerblue']

    for i in range(Nmodels):
        model = models[i]
        for j in range(2):
            array = model[j]
            # Create a box plot for each array
            # Adjust positions so that box plots of the same model are close to each other
            ax.boxplot(array, positions=[i*2+j], patch_artist=True,
                    boxprops=dict(facecolor=colors[j]))

    # Set x-axis labels to model names
    ax.set_xticks([i*2+0.5 for i in range(len(models))])
    ax.set_xticklabels(model_names)
    for i in range(len(models)):
        plt.axvline(i*2+1.5,color='k',linestyle='--')

    patch1 = mpatches.Patch(color='darkorange', label='Seen Plates')
    patch2 = mpatches.Patch(color='dodgerblue', label='Held-Out Plates')
    plt.legend(handles=[patch1, patch2])
    plt.ylabel('Accuracy (Seen Channels)')
    plt.savefig(plot_save_dir+'model_comparisons.png',dpi=500, bbox_inches='tight')



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, default="config_run.yaml", help="Config file"
    )

    args = parser.parse_args()

    yaml_file = open(f"{args.config}")
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    main(config)
