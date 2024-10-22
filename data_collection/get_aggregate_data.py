###
import argparse
import os
from typing import List
import glob
import numpy as np 
import pandas as pd
import yaml

def get_aggregate_data(
    plates: pd.DataFrame,
    controls: pd.DataFrame,
    last_layer: str
    ):
    """
    Get Aggregate Data

    Given a pandas dataframe of featurisation for 
    single cell images from a plate, and the controls_csv
    for that plate - this function will aggregate all 
    embeddings for images derived from the same well. 

    A new dataframe is created with each row 
    including the aggregate embedding features, 
    the row, column and plate barcode, as well
    as the split the well was assigned and compound
    used to treat the well. 

    Arguments
    ---------
    plates: dataframe including row,column,platebarcode,
            and embedding features
    controls: dataframe including row, column, plate_barcode,
             ,and a final column with split_compoundname in that format
    control_ids: list of control_ids i.e compound names we want data for
    last_layer: string of last layer name 

    Returns
    -------
    pd.DataFrame

    """

    feature_cols = [col for col in plates.columns if col.startswith(last_layer)]

    plates = plates.groupby(['ROW','COLUMN','PLATE_BARCODE'])[feature_cols].mean().reset_index()

    plates['split'] = plates.apply(lambda row: controls.iloc[int(row['ROW']-1), int(row['COLUMN']-1)], axis=1)
    plates['compound'] = plates.apply(lambda row: row['split'].split('_JCP',1)[1] if pd.notnull(row['split']) else np.nan, axis=1)
    plates['split'] = plates.apply(lambda row: row['split'].split('_JCP',1)[0] if pd.notnull(row['split']) else np.nan, axis=1)

    held_out_splits = ['test_out_compound','test_out_all_']
    wells_held_out = plates.loc[plates['split'].isin(held_out_splits)]

    wells_held_out['well_coords'] = wells_held_out.apply(lambda row: str(row['ROW'])+','+str(row['COLUMN']), axis=1)

    feature_cols = [col for col in wells_held_out.columns if col.startswith(last_layer)] + ['compound','split']

    feature_df = wells_held_out[feature_cols]

    held_out_feat = feature_df

    return held_out_feat 


def main(config) -> None:
    """
    Main Function:

    Load subdirectories containing feature csvs 



    """
    seed = config["seed"]

    model_dir = config["model_dir"]
    feature_dirs = [name for name in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, name))]

    features_string = '/FEATURES/'

    num_samples = config["num_samples"]
    last_layer = config["last_layer"]
    controls_dir = config["controls_dir"]
    save_dir = config["eval_data_dir"]
    model_name = config["model_name"]

    print("Model Name: {}".format(model_name))

    probing_df = pd.DataFrame([])
    for i in range(len(feature_dirs)):
        print("Processing file: {}/{}: ".format(i+1,len(feature_dirs)),flush=True)

        with open(model_dir + feature_dirs[i] + "/user_config.yaml", "r") as f:
            user_config = yaml.safe_load(f)
        plate_id = user_config["barcode"][0].split("_")[1]
        print("Plate: {}: ".format(plate_id),flush=True)
        controls = pd.read_csv(controls_dir + plate_id + "_controls.csv", index_col=0)

        feature_path = glob.glob(model_dir+feature_dirs[i]+features_string + "/*.csv")[0]
        features = pd.read_csv(feature_path)

        new_probing_df= get_aggregate_data(features,controls,last_layer)

        if i >0:
            probing_df = pd.concat([probing_df,new_probing_df])
        
        else:
            probing_df = new_probing_df

    target_names = np.unique(probing_df['compound'].values)
    label_mapper = {target_names[i]: i for i in range(len(target_names))}

    probing_df["TARGET"] = probing_df["compound"].map(label_mapper)

    probing_df.to_csv(save_dir+'/held_out_aggregate_data_'+model_name+'.csv')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, default="config_run.yaml", help="Config file"
    )

    args = parser.parse_args()

    yaml_file = open(f"{args.config}")
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    main(config)
