import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import pandas as pd
from pandas.plotting import table 
from scipy import stats

def main(config) -> None:
    """
    Main Function:

    Load subdirectories containing feature csvs 



    """
    test_metric_dirs = config["held_out_compounds_test_metric_dirs"]
    test_out_metric_dirs = config["held_out_compounds_test_out_metric_dirs"]
    model_names = config["held_out_compounds_model_names"]
    plot_save_dir = config["held_out_compounds_plot_save_dir"]

    test_data_to_plot = []
    test_out_data_to_plot = []

    for metric_dir in test_metric_dirs:

        with open(metric_dir, 'rb') as handle:
            metrics = pickle.load(handle)

        acc = np.array(metrics)

        # Create a list of these arrays
        test_data_to_plot.append(acc)

    for metric_dir in test_out_metric_dirs:

        with open(metric_dir, 'rb') as handle:
            metrics = pickle.load(handle)


        acc = np.array(metrics)

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
    plt.legend(loc=2)
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



    data_dict = {}

    N = len(model_names)

    anchor_model = test_metric_dirs = config["comparison_model_name"]
    anchor_index = np.where(np.array(model_names)==anchor_model)[0][0]

    for i in range(N):

        model_metrics = test_data_to_plot[i]

        result_dict = {}

        result_dict['IID Accuracy'] = str(np.round(np.mean(model_metrics),3)) + '±' + str(np.round(np.std(model_metrics),3)) 

        model_out_metrics = test_out_data_to_plot[i]
        result_dict['OOD Accuracy'] = str(np.round(np.mean(model_out_metrics),3)) + '±' + str(np.round(np.std(model_out_metrics),3)) 

        ratio = (model_out_metrics - model_metrics)/model_metrics

        result_dict['(OOD - IID / OOD) Accuracy'] = str(np.round(np.mean(ratio),3)) + '±' + str(np.round(np.std(ratio),3)) 

        data_dict[model_names[i]] = result_dict

        if i != anchor_index:

            t_statistic, p_value = stats.ttest_ind(test_data_to_plot[anchor_index], test_data_to_plot[i])

            data_dict[model_names[i]]['IID p-value'] = np.round(p_value,3)

            t_statistic, p_value = stats.ttest_ind(test_out_data_to_plot[anchor_index], test_out_data_to_plot[i])

            data_dict[model_names[i]]['OOD p-value'] = np.round(p_value,6)


    plt.style.use('bmh')
    df = pd.DataFrame(data_dict).T  # your DataFrame

    fig, ax = plt.subplots(figsize=(12, 2)) # set size frame
    ax.axis('off')

    tbl = table(ax, df, loc='center', cellLoc='center')

    plt.savefig(plot_save_dir+'result_table.png',dpi=500, bbox_inches='tight')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, default="config_run.yaml", help="Config file"
    )

    args = parser.parse_args()

    yaml_file = open(f"{args.config}")
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    main(config)
