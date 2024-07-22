###
import argparse
import os
import pathlib
from typing import Dict

import numpy as np 
import pandas as pd
import yaml

def get_probing_data(plates,controls,control_ids,num_samples,last_layer,seed):


    plates['split'] = plates.apply(lambda row: controls.iloc[int(row['ROW']-1), int(row['COLUMN']-1)], axis=1)
    plates['compound'] = plates.apply(lambda row: row['split'].split('_JCP',1)[1] if pd.notnull(row['split']) else np.nan, axis=1)
    plates['split'] = plates.apply(lambda row: row['split'].split('_JCP',1)[0] if pd.notnull(row['split']) else np.nan, axis=1)


    plates = plates.loc[plates['compound'].isin(control_ids)]


    feature_cols = [col for col in plates.columns if col.startswith(last_layer)]
    np.random.seed(seed)
    feature_df = plates.groupby(['compound','split'])[feature_cols].apply(lambda x: x.sample(n=num_samples, replace=False)).reset_index()

    held_out_feat = feature_df

    held_out_feat = held_out_feat.loc[held_out_feat['compound'].isin(control_ids)]

    target_names = np.unique(held_out_feat['compound'].values)
    label_mapper = {target_names[i]: i for i in range(len(target_names))}

    held_out_feat["TARGET"] = held_out_feat["compound"].map(label_mapper)

    return held_out_feat 


def main(config) -> None:
    """
    Main Function:

    Load subdirectories containing feature csvs 



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

    control_ids = ['2022_033924',
    '2022_085227',
    '2022_037716',
    '2022_025848',
    '2022_046054',
    '2022_035095',
    '2022_064022',
    '2022_050797',
    '2022_012818']

    num_samples = config["num_samples"]
    last_layer = config["last_layer"]
    controls_dir = config["controls_dir"]
    save_dir = config["linear_probe_data_dir"]
    model_name = config["model_name"]

    probing_df = pd.DataFrame([])
    for i in range(len(feature_dirs)):

        features = pd.read_csv(model_dir+feature_dirs[i]+features_string)
        controls = controls = pd.read_csv(controls_dir+plate_ids[i]+'_controls.csv',index_col=0)
        new_probing_df= get_probing_data(features,controls,control_ids,num_samples,last_layer,seed)

        if i >0:
            probing_df = pd.concat([probing_df,new_probing_df])
        
        else:
            probing_df = new_probing_df

    probing_df.to_csv(save_dir+'linear_probing_'+model_name+'.csv')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, default="config_run.yaml", help="Config file"
    )

    args = parser.parse_args()

    yaml_file = open(f"{args.config}")
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    main(config)
