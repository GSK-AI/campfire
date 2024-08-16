###
import argparse
import os
from typing import List

import numpy as np 
import pandas as pd
import yaml

def get_aggregate_data(
    plates: pd.DataFrame,
    controls: pd.DataFrame,
    control_ids: List[str],
    last_layer: str,
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


    if control_ids is not None: 
        print("None")
        plates = plates.loc[plates['compound'].isin(control_ids)]

    feature_cols = feature_cols + ['compound','split']

    feature_df = plates[feature_cols]
    feature_df = feature_df.dropna()

    target_names = np.unique(feature_df ['compound'].values)
    label_mapper = {target_names[i]: i for i in range(len(target_names))}

    feature_df ["TARGET"] = feature_df["compound"].map(label_mapper)

    feature_df = feature_df.rename(columns=lambda x: 'embedding_' + x.split("_")[-1] if x.startswith(last_layer) else x)

    return feature_df 


def main(config) -> None:
    """
    Main Function:

    Load subdirectories containing feature csv files 
    for each plate. 

    For each plate, create a dataframe which 
    has the aggregate embedding for each well, 
    and includes the split and compound assigned to it

    Finally, save csv file containing aggregate embeddings 
    and targets for each well in each plate, and split. 

    """
    seed = config["seed"]

    model_dir = config["model_dir"]
    feature_dirs = [name for name in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, name))]

    features_string = '/FEATURES/metadata_features_model0.csv'

    plate_ids = ['JCPQC016','JCPQC017','JCPQC018','JCPQC019','JCPQC020',
            'JCPQC021','JCPQC022','JCPQC023','JCPQC024','JCPQC025',
            'JCPQC028','JCPQC029','JCPQC030','JCPQC031','JCPQC032',
            'JCPQC033','JCPQC034','JCPQC035','JCPQC036','JCPQC037',
            'JCPQC038','JCPQC051','JCPQC052','JCPQC053','JCPQC054']

    control_ids = config['control_ids']

    num_samples = config["num_samples"]
    last_layer = config["last_layer"]
    controls_dir = config["controls_dir"]
    save_dir = config["eval_data_dir"]
    model_name = config["model_name"]

    probing_df = pd.DataFrame([])
    for i in range(len(feature_dirs)):

        features = pd.read_csv(model_dir+feature_dirs[i]+features_string)
        controls = controls = pd.read_csv(controls_dir+plate_ids[i]+'_controls.csv',index_col=0)
        new_probing_df= get_aggregate_data(features,controls,control_ids,last_layer)

        if i >0:
            probing_df = pd.concat([probing_df,new_probing_df])
        
        else:
            probing_df = new_probing_df

    probing_df.to_csv(save_dir+'aggregate_data_'+model_name+'.csv')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, default="config_run.yaml", help="Config file"
    )

    args = parser.parse_args()

    yaml_file = open(f"{args.config}")
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    main(config)
