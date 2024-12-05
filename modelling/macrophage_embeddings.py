import numpy as np 
import pandas as pd 
from sklearn.decomposition import PCA
import argparse
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def compute_principle_components(df: pd.DataFrame, embedding_layer_name: str) -> pd.DataFrame:
    """
    Computes 2 principal components from columns starting with the embedding layer name.

    Args:
        df (pd.DataFrame): The input dataframe containing embedding columns.
        embedding_layer_name (str): The prefix of the embedding layer columns.

    Returns:
        pd.DataFrame: A dataframe with the 2 principal components.
    """
    # get the columns starting with the embedding layer name
    embedding_columns = [col for col in df.columns if col.startswith(embedding_layer_name)]
    # get the values of the embedding columns
    X = df[embedding_columns].values
    y = df['target'].values
    # compute the principle components using PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X)
    # create a dataframe with the principle components
    df_pc = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
    df_pc['target'] = y
    return df_pc,pca

def apply_pca(pca, df: pd.DataFrame, embedding_layer_name: str) -> pd.DataFrame:
    """
    Applies PCA to a new dataframe using the PCA object.

    Args:
        pca: The PCA object.
        df (pd.DataFrame): The input dataframe containing embedding columns.
        embedding_layer_name (str): The prefix of the embedding layer columns.

    Returns:
        pd.DataFrame: A dataframe with the 2 principal components.
    """
    # get the columns starting with the embedding layer name
    embedding_columns = [col for col in df.columns if col.startswith(embedding_layer_name)]
    # get the values of the embedding columns
    X = df[embedding_columns].values
    y = df['target'].values
    # compute the principle components using PCA
    principal_components = pca.transform(X)
    # create a dataframe with the principle components
    df_pc = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
    df_pc['target'] = y
    return df_pc


def merge_plate_and_controls(plate_data: pd.DataFrame, controls_dict: dict) -> pd.DataFrame:
    """
    Processes the 'plate_data' DataFrame by adding 'split' and 'compound' columns based on the 'controls_dict'.

    Args:
        plate_data (pd.DataFrame): DataFrame containing plate information with 'ROW', 'COLUMN', and 'PLATE_BARCODE' columns.
        controls_dict (dict): Dictionary containing control DataFrames keyed by plate barcode.

    Returns:
        pd.DataFrame: The modified 'plate_data' DataFrame with added 'split' and 'compound' columns.
    """
    def apply_controls(row):
        controls = controls_dict[row['PLATE_BARCODE']]
        split_value = controls.iloc[int(row['ROW']-1), int(row['COLUMN']-1)]
        if pd.notnull(split_value):
            split, compound = split_value.split('_', 1)
            return pd.Series([split, compound])
        else:
            return pd.Series([np.nan, np.nan])

    plate_data[['split', 'compound']] = plate_data.apply(apply_controls, axis=1)
    return plate_data


def load_plate_data(list_of_paths: list, last_layer: str) -> pd.DataFrame:
    """
    Loads plate data from a list of paths, applies mean across rows which share the values in columns ROW, COLUMN, and PLATE_BARCODE, 
    and only uses feature columns beginning with the last_layer name.

    Args:
        list_of_paths (list): List of paths to plate data.
        last_layer (str): The prefix of the feature columns.

    Returns:
        pd.DataFrame: The concatenated and aggregated plate data.
    """
    df = pd.concat([pd.read_csv(path) for path in list_of_paths], ignore_index=True)
    feature_cols = [col for col in df.columns if 
    col.startswith(last_layer)]
    df = df.groupby(['ROW', 'COLUMN', 'PLATE_BARCODE'])[feature_cols].mean().reset_index()
    return df

# function that plots id and ood priciple components on same scatter plot, with colour of points determined by target
def plot_principle_components(id_principle_components: pd.DataFrame, ood_principle_components: pd.DataFrame, save_path: str) -> None:
    """
    Plots the principle components of in-distribution and out-of-distribution plates on the same scatter plot with color determined by target.

    Args:
        id_principle_components (pd.DataFrame): The in-distribution principle components.
        ood_principle_components (pd.DataFrame): The out-of-distribution principle components.
        save_path (str): The path to save the plot.

    Returns:
        None
    """

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot the first dataset
    scatter1 = plt.scatter(id_principle_components['PC1'], id_principle_components['PC2'], 
                        c=pd.Categorical(id_principle_components['target']).codes, cmap='tab10', 
                        label='Original Data', alpha=0.6)

    # Plot the second dataset
    scatter2 = plt.scatter(ood_principle_components['PC1'], ood_principle_components['PC2'], 
                        c=pd.Categorical(ood_principle_components['target']).codes + len(pd.Categorical(id_principle_components['target']).categories), 
                        cmap='plasma', label='New Data', alpha=0.6)

    # Creating a legend
    unique_targets_1 = np.unique(id_principle_components['target'].values)
    unique_targets_2 = np.unique(ood_principle_components['target'].values)
    all_handles = []

    # Handles for the first data set
    for i, label in enumerate(unique_targets_1):
        all_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=scatter1.cmap(scatter1.norm(i)), markersize=8, label=label))
        
    # Handles for the second data set
    for i, label in enumerate(unique_targets_2):
        all_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=scatter2.cmap(scatter2.norm(i + len(unique_targets_1))), markersize=8, label=label))

    plt.legend(handles=all_handles, title="Classes", loc="upper right")

    # Adding axis labels and title
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(save_path+'/principle_components.png', bbox_inches='tight')
    return None



def get_centroids(plate_data: pd.DataFrame, last_layer: str) -> dict:
    """
    Computes the centroids of the feature columns grouped by the 'target' column.

    Args:
        plate_data (pd.DataFrame): The input dataframe containing feature columns and a 'target' column.
        last_layer (str): The prefix of the feature columns.

    Returns:
        dict: A dictionary where keys are unique targets and values are the centroids of the feature columns.
    """
    targets = np.unique(plate_data['target'].values)
    feature_cols = [col for col in plate_data.columns if col.startswith(last_layer)]
    centroid_df = plate_data.groupby(['target'])[feature_cols].mean().reset_index()
    centroids = {target: centroid_df.loc[centroid_df['target'] == target, feature_cols].values for target in targets}
    return centroids



def distance_to_centroids(plate_data: pd.DataFrame, centroid_dict: dict, last_layer: str) -> pd.DataFrame:
    """
    Computes the average cosine similarity distance to centroids for each target in the plate data and returns it as a DataFrame.

    Args:
        plate_data (pd.DataFrame): The input dataframe containing feature columns and a 'target' column.
        centroid_dict (dict): A dictionary where keys are unique targets and values are the centroids of the feature columns.
        last_layer (str): The prefix of the feature columns.

    Returns:
        pd.DataFrame: A DataFrame where each row represents a target and columns represent average distances to each centroid.
    """
    targets = np.unique(plate_data['target'].values)
    feature_cols = [col for col in plate_data.columns if col.startswith(last_layer)]
    sim_dict = {}

    for target in targets:
        sim_dict[target] = {}
        tmp = plate_data.loc[plate_data['target'] == target]
        latents = tmp[feature_cols].values

        for key in centroid_dict.keys():
            centroid = centroid_dict[key]
            sims = cosine_similarity(latents, centroid)
            av_dist_to_centroid = np.mean(sims, axis=0)
            sim_dict[target][key + '-centroid'] = av_dist_to_centroid[0]

    sim_df = pd.DataFrame.from_dict(sim_dict, orient='index').reset_index().rename(columns={'index': 'target'})
    return sim_df


def rank_distances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ranks the distances in the DataFrame for each column.

    Args:
        df (pd.DataFrame): The input dataframe containing distances.

    Returns:
        pd.DataFrame: A DataFrame with ranked distances.
    """
    ranked_df = df.rank(axis=0, ascending=False).astype(int)
    return ranked_df

def process_and_save_distances(plate_data: pd.DataFrame, centroids: dict, last_layer: str, save_path: str) -> None:
    """
    Computes the distance to centroids for all plates, ranks the distances, and saves the results as CSV and heatmap images.

    Args:
        plate_data (pd.DataFrame): The input dataframe containing plate data.
        centroids (dict): A dictionary where keys are unique targets and values are the centroids of the feature columns.
        last_layer (str): The prefix of the feature columns.
        save_path (str): The path to save the results.

    Returns:
        None
    """
    # compute the distance to centroids for all plates
    distance_df = distance_to_centroids(plate_data, centroids, last_layer)
    
    # save the distance to centroids
    distance_df.to_csv(save_path + '/distance_to_centroids.csv', index=False)

    # plot the distance dataframe as a heatmap   
    plt.figure(figsize=(10, 6))
    sns.heatmap(distance_df.set_index('target').T, cmap='viridis', annot=True, fmt=".2f")
    plt.title('Average Cosine Similarity Distance to Centroids')
    plt.savefig(save_path + '/distance_to_centroids_heatmap.png', bbox_inches='tight')

    # rank the distances
    ranked_df = rank_distances(distance_df)
    ranked_df.to_csv(save_path + '/ranked_distances.csv', index=False)

    # plot the ranked dataframe as a heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(ranked_df.set_index('target').T, cmap='viridis', annot=True, fmt="d")
    plt.title('Ranked Distances to Centroids')
    plt.savefig(save_path + '/ranked_distances_heatmap.png', bbox_inches='tight')

def main(config) -> None:
    """
    Main Function:
    """

    paths_to_plates = config["paths_to_plates"]
    paths_to_controls = config["paths_to_controls"]
    plate_barcodes = config["plate_barcodes"]
    id_barcodes = config["id_barcodes"]

    # load all plates, aggregate them by well, and concat into single data frame
    plate_data = load_plate_data(paths_to_plates, config["last_layer"])

    # take list of plate barcodes and list of control paths and create dictionary of control dataframes
    controls_dict = {plate: pd.read_csv(path,index_col=0) for plate, path in zip(plate_barcodes, paths_to_controls)}

    # add the split and compound columns using the controls_dict
    plate_data = merge_plate_and_controls(plate_data, controls_dict)

    #only keep rows where compound is in keep_compounds
    plate_data = plate_data[plate_data['compound'].isin(config["plot_compounds"])]

    plate_data['target'] = plate_data['compound'] + '_' + plate_data['PLATE_BARCODE']
    
    #sort rows of plate_data by target
    plate_data = plate_data.sort_values(by='target')

    # compute the principle components of plates with barcodes that are in-distribution
    plate_data_id = plate_data[plate_data['PLATE_BARCODE'].isin(id_barcodes)]
    id_principle_components,pca_function = compute_principle_components(plate_data_id, config["last_layer"])

    # apply the PCA to the out-of-distribution plates
    plate_data_ood = plate_data[~plate_data['PLATE_BARCODE'].isin(id_barcodes)]
    ood_principle_components = apply_pca(pca_function, plate_data_ood, config["last_layer"])

    # save the principle components
    id_principle_components.to_csv(config["save_path"]+'/id_pc.csv', index=False)
    ood_principle_components.to_csv(config["save_path"]+'/ood_pc.csv', index=False)

    # plot the principle components
    plot_principle_components(id_principle_components, ood_principle_components, config["save_path"])

    # get the centroids of all plates
    centroids = get_centroids(plate_data, config["last_layer"])

    # process and save distances
    process_and_save_distances(plate_data, centroids, config["last_layer"], config["save_path"])

    return None

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, default="config_run.yaml", help="Config file"
    )

    args = parser.parse_args()

    yaml_file = open(f"{args.config}")
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    main(config)
