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
    test_metric_dirs = config["batch_generalisation_test_metric_dirs"]
    test_out_metric_dirs = config["batch_generalisation_test_out_metric_dirs"]
    model_names = config["batch_generalisation_model_names"]
    plot_save_dir = config["batch_generalisation_plot_save_dir"]

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


    Nmodels = len(test_data_to_plot)
    models = []
    for i in range(Nmodels):
        models.append(np.array([test_data_to_plot[i],test_out_data_to_plot[i]]))

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
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(plot_save_dir+'mean_model_comparisons.png',dpi=500, bbox_inches='tight')



    Nmodels = len(test_data_to_plot)
    mean_change = np.zeros(Nmodels)
    std_change = np.zeros(Nmodels)
    for i in range(Nmodels):
        fractional_change = (test_out_data_to_plot[i]-test_data_to_plot[i])/test_data_to_plot[i]
        mean_change[i]= np.mean(fractional_change)
        std_change[i] = np.std(fractional_change)

    print(mean_change,flush=True)

    plt.style.use('bmh')
    # Create a figure instance
    plt.figure()

    x = np.arange(len(models))*0.15

    # Create error bar plots
    plt.errorbar(x, mean_change, yerr=std_change, fmt='o')

    plt.xticks(x, model_names)  # We add 0.25 to center the labels

    # Set up x-axis labels and title
    plt.xticks(x, model_names)  # We add 0.25 to center the labels
    # plt.xlabel('Model')
    plt.ylabel('Relative Change in Accuracy from IID to OOD')
    plt.savefig(plot_save_dir+'relative_model_comparisons.png',dpi=500, bbox_inches='tight')


    # iid_test_metric_dirs = config["iid_test_metric_dirs"]
    # iid_test_out_metric_dirs = config["iid_test_out_metric_dirs"]
    # ood_test_metric_dirs = config["ood_test_metric_dirs"]
    # ood_test_out_metric_dirs = config["ood_test_out_metric_dirs"]


    # iid_test_data_to_plot = []
    # iid_test_out_data_to_plot = []
    # ood_test_data_to_plot = []
    # ood_test_out_data_to_plot = []

    # for metric_dir in iid_test_metric_dirs:

    #     with open(metric_dir, 'rb') as handle:
    #         metrics = pickle.load(handle)

    #     Nruns = len(metrics)

    #     acc = np.zeros(Nruns)

    #     for i in range(Nruns):
    #         acc[i] = metrics[i]["accuracy"]

    #     # Create a list of these arrays
    #     iid_test_data_to_plot.append(acc)

    # for metric_dir in iid_test_out_metric_dirs:

    #     with open(metric_dir, 'rb') as handle:
    #         metrics = pickle.load(handle)

    #     Nruns = len(metrics)

    #     acc = np.zeros(Nruns)

    #     for i in range(Nruns):
    #         acc[i] = metrics[i]["accuracy"]

    #     # Create a list of these arrays
    #     iid_test_out_data_to_plot.append(acc)


    # for metric_dir in ood_test_metric_dirs:

    #     with open(metric_dir, 'rb') as handle:
    #         metrics = pickle.load(handle)

    #     Nruns = len(metrics)

    #     acc = np.zeros(Nruns)

    #     for i in range(Nruns):
    #         acc[i] = metrics[i]["accuracy"]

    #     # Create a list of these arrays
    #     ood_test_data_to_plot.append(acc)

    # for metric_dir in ood_test_out_metric_dirs:

    #     with open(metric_dir, 'rb') as handle:
    #         metrics = pickle.load(handle)

    #     Nruns = len(metrics)

    #     acc = np.zeros(Nruns)

    #     for i in range(Nruns):
    #         acc[i] = metrics[i]["accuracy"]

    #     # Create a list of these arrays
    #     ood_test_out_data_to_plot.append(acc)


    # Nmodels = len(test_data_to_plot)
    # iid_mean_change = np.zeros(Nmodels)
    # iid_std_change = np.zeros(Nmodels)
    # for i in range(Nmodels):
    #     fractional_change = (ood_test_data_to_plot[i]-iid_test_data_to_plot[i])/iid_test_data_to_plot[i]
    #     iid_mean_change[i]= np.mean(fractional_change)
    #     iid_std_change[i] = np.std(fractional_change)

    # Nmodels = len(test_data_to_plot)
    # ood_mean_change = np.zeros(Nmodels)
    # ood_std_change = np.zeros(Nmodels)
    # for i in range(Nmodels):
    #     fractional_change = (ood_test_out_data_to_plot[i]-iid_test_out_data_to_plot[i])/iid_test_out_data_to_plot[i]
    #     ood_mean_change[i]= np.mean(fractional_change)
    #     ood_std_change[i] = np.std(fractional_change)


    # plt.style.use('bmh')
    # # Create a figure instance
    # fig = plt.figure()
    # # Create an array for the x positions of the bars
    # x = np.arange(len(models)) * 2  # We multiply by 2 to leave space for each model's two bars

    # # Create error bar plots
    # plt.errorbar(x, iid_mean_change, yerr=iid_std_change, fmt='o', label='Seen Plates')
    # plt.errorbar(x + 0.5, ood_mean_change, yerr=ood_std_change, fmt='o', label='Held-Out Plates')  # We add 0.5 to shift the second metric's bars to the right

    # # Add vertical dotted lines
    # for i in range(len(models) - 1):
    #     plt.axvline(x=2*(i)+1.25, color='grey', linestyle='dotted')
    # # Set up x-axis labels and title
    # plt.xticks(x + 0.25, model_names)  # We add 0.25 to center the labels
    # # plt.xlabel('Model')
    # plt.ylabel('Change in Accuracy')
    # plt.legend()
    # plt.savefig(plot_save_dir+'channel_comparisons.png',dpi=500, bbox_inches='tight')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, default="config_run.yaml", help="Config file"
    )

    args = parser.parse_args()

    yaml_file = open(f"{args.config}")
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    main(config)
