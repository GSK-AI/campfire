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

    ax.set_xticklabels(model_names,fontsize=7)

    # Create the boxplot
    bp = ax.boxplot(test_data_to_plot)

    plt.ylabel('Accuracy (Plate: Seen || Channels: UnSeen)')

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

    ax.set_xticklabels(model_names,fontsize=7)

    # Create the boxplot
    bp = ax.boxplot(test_out_data_to_plot)

    plt.ylabel('Accuracy (Plate: UnSeen || Channels: UnSeen)')

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
    plt.ylabel('Accuracy (UnSeen Channels)')
    plt.savefig(plot_save_dir+'model_comparisons.png',dpi=500, bbox_inches='tight')


    plt.style.use('bmh')
    # Create a figure instance
    fig = plt.figure()
    # Create an array for the x positions of the bars
    x = np.arange(len(models)) * 2  # We multiply by 2 to leave space for each model's two bars

    # Calculate means and standard deviations
    metric1_means = [np.mean(model[0]) for model in models]
    metric1_stds = [np.std(model[0]) for model in models]

    metric2_means = [np.mean(model[1]) for model in models]
    metric2_stds = [np.std(model[1]) for model in models]

    # Create error bar plots
    plt.errorbar(x, metric1_means, yerr=metric1_stds, fmt='o', label='Seen Plates')
    plt.errorbar(x + 0.5, metric2_means, yerr=metric2_stds, fmt='o', label='Held-Out Plates')  # We add 0.5 to shift the second metric's bars to the right

    # Add vertical dotted lines
    for i in range(len(models) - 1):
        plt.axvline(x=2*(i)+1.25, color='grey', linestyle='dotted')
    # Set up x-axis labels and title
    plt.xticks(x + 0.25, model_names)  # We add 0.25 to center the labels
    # plt.xlabel('Model')
    plt.ylabel('Accuracy (Seen Channels)')
    plt.legend()
    plt.savefig(plot_save_dir+'mean_model_comparisons.png',dpi=500, bbox_inches='tight')



    Nmodels = len(test_data_to_plot)
    rels = []
    for i in range(Nmodels):
        rels.append(np.array(abs(test_data_to_plot[i]-test_out_data_to_plot[i])))


    plt.style.use('bmh')
    # Create a figure instance
    fig = plt.figure()

    # Create an axes instance
    ax = fig.add_axes([0,0,1,1])

    ax.set_xticklabels(model_names,fontsize=7)

    # Create the boxplot
    bp = ax.boxplot(rels)

    plt.ylabel('Accuracy (Plate: UnSeen || Channels: UnSeen)')

    # plt.ylim(0.25,0.5)
    # plt.axhline(1/9)
    # Show the plot
    plt.tight_layout()
    plt.savefig(plot_save_dir+'relative_test_out_comparisons.png',dpi=500, bbox_inches='tight')



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, default="config_run.yaml", help="Config file"
    )

    args = parser.parse_args()

    yaml_file = open(f"{args.config}")
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    main(config)
