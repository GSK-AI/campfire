import numpy as np 
import pandas as pd 
from sklearn.decomposition import PCA
import argparse
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from utils.projection import scalar_projection
from utils.zprime import robust_zprime 

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

#function that takes list of pair of pos and neg controls and returns a list of zprime values
def compute_zprime(plate_df: pd.DataFrame, controls: list, last_layer: str) -> list:
    """
    Computes the zprime value for each pair of positive and negative controls.

    Args:
        plate_df (pd.DataFrame): DataFrame containing plate data.
        controls (list): List of tuples containing pairs of positive and negative controls.
        last_layer (str): The prefix of the feature columns.

    Returns:
        list: The list of controls with appended zprime values.
    """

    feature_cols = [col for col in plate_df.columns if 
    col.startswith(last_layer)]

    for i, (AD, Healthy) in enumerate(controls):
        on_target, off_target = scalar_projection(
            df=plate_df[["compound"] + feature_cols],
            feature_col=feature_cols,
            ref_col="compound",
            ref_origin=AD,
            ref_target=Healthy,
            num_bootstrap=0,
        )

        plate_df["Phenoscore"] = on_target

        origin_ = plate_df[plate_df["compound"] == AD]["Phenoscore"].values
        target_ = plate_df[plate_df["compound"] == Healthy]["Phenoscore"].values

        z_prime = robust_zprime(origin_, target_)[0]

        controls[i].append(z_prime)

    return controls

# function to plot the zprime values as a symmetric heatmap
def plot_zprime_heatmap(controls: list, vmin: float, vmax: float, save_path: str) -> None:
    """
    Plots a symmetric heatmap of the zprime values.

    Args:
        controls (list): List of tuples containing pairs of positive and negative controls with appended zprime values.
        vmin (float): The minimum value of the colormap.
        vmax (float): The maximum value of the colormap.

    Returns:
        None
    """
    controls_df = pd.DataFrame(controls, columns=['AD', 'Healthy', 'Zprime'])
    # Ensure the DataFrame is symmetrical by adding missing pairs with NaN values
    all_compounds = sorted(set(controls_df['AD']).union(set(controls_df['Healthy'])))
    symmetrical_df = pd.DataFrame(index=all_compounds, columns=all_compounds)

    for _, row in controls_df.iterrows():
        symmetrical_df.loc[row['AD'], row['Healthy']] = row['Zprime']
        symmetrical_df.loc[row['Healthy'], row['AD']] = row['Zprime']

    # Convert the DataFrame to numeric and fill NaN values with 0
    symmetrical_df = symmetrical_df.astype(float).fillna(0)

    # Reorder the rows and columns to have M1 rows first, then M2 rows
    m1_rows = [compound for compound in all_compounds if 'M1' in compound]
    m2_rows = [compound for compound in all_compounds if 'M2' in compound]
    ordered_compounds = m1_rows + m2_rows
    symmetrical_df = symmetrical_df.loc[ordered_compounds, ordered_compounds]

    # Plot the symmetrical heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(symmetrical_df, annot=True, cmap='viridis', vmax=1, vmin=-1)
    # label the axes
    plt.xlabel('Target')
    plt.ylabel('Reference')
    # label the colorbar
    cbar = plt.gca().collections[0].colorbar
    cbar.set_label('Z\'')
    plt.savefig(save_path+'/z_prime.png', bbox_inches='tight')
    return None

def main(config) -> None:
    """
    Main Function:
    """

    paths_to_plates = config["paths_to_plates"]
    paths_to_controls = config["paths_to_controls"]
    plate_barcodes = config["plate_barcodes"]
    id_barcodes = config["id_barcodes"]
    m2_plates = config["m2_plates"]
    vmin = config["vmin"]
    vmax = config["vmax"]
    #if vmin and vmax are not specified in config, set them to 0 and 1 respectively
    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = 1
        

    # load all plates, aggregate them by well, and concat into single data frame
    plate_data = load_plate_data(paths_to_plates, config["last_layer"])

    # take list of plate barcodes and list of control paths and create dictionary of control dataframes
    controls_dict = {plate: pd.read_csv(path,index_col=0) for plate, path in zip(plate_barcodes, paths_to_controls)}

    # add the split and compound columns using the controls_dict
    plate_data = merge_plate_and_controls(plate_data, controls_dict)

    #only keep rows where compound is in keep_compounds
    plate_data = plate_data[plate_data['compound'].isin(config["plot_compounds"])]

    #onlyt keep plates that are ood
    plate_data = plate_data[~plate_data['PLATE_BARCODE'].isin(id_barcodes)]

    #if compound is neg, change to neg-M2 if PLATE_BARCODE is in m2_plates 
    plate_data['compound'] = plate_data.apply(lambda row: row['compound'] if row['compound'] != 'neg' else row['compound'] + ('-M2' if row['PLATE_BARCODE'] == 'ELN27012_G212LFH' else '-M1'), axis=1)

    # read in the controls and compute the zprime values
    controls = config["controls"]
    controls = compute_zprime(plate_data, controls, config["last_layer"])

    # plot the zprime values as a heatmap
    plot_zprime_heatmap(controls, vmin, vmax, config["save_path"])

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
