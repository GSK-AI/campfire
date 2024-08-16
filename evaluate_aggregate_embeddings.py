###
###
import argparse
import pathlib

import numpy as np 
import pandas as pd
import yaml

import pickle
from sklearn.metrics import accuracy_score



from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder


def train_catboost(
    seed,
    log_dir,
    train_data,
    train_labels,
    test_data,
    test_labels,
    out_data,
    out_labels):
    """
    Train CatBoost 

    Given train features and labels, 
    train CatBoost with default parameters

    Compute F1 and Accuracy for test wells
    and for the test_out_plate wells i.e 
    plates not seen in training

    Save to pkl 

    Arguments
    ---------
    seed: random seed
    log_dir: directory to store catboost training logs
    train_data: features of training samples
    train_labels: labels of training samples
    test_data: features of test samples
    test_labels: labels of test samples
    out_data: features of test_out_plate samples
    out_labels: labels of test_out_plate samples
    """
    new_dir = pathlib.Path(log_dir)
    new_dir.mkdir(parents=True, exist_ok=True)

    model = CatBoostClassifier(eval_metric="MultiClass",train_dir=log_dir,random_seed=seed)

    model.fit(train_data, train_labels)

    y_pred = model.predict(test_data)
    y_pred_out = model.predict(out_data)

    f1 = f1_score(test_labels, y_pred, average="macro")
    accuracy = accuracy_score(test_labels, y_pred)

    result_dict = {"f1": [f1], "accuracy": [accuracy]}

    f1 = f1_score(out_labels, y_pred_out, average="macro")
    accuracy = accuracy_score(out_labels, y_pred_out)

    result_dict_out = {"f1": [f1], "accuracy": [accuracy]}
    

    return result_dict,result_dict_out


def main(config) -> None:
    """
    Main Function

    Read in aggregate embedding data frame
    Get train,val,test and test_out_plate samples

    Split train samples into K-splits

    For each split train CatBoost classifier

    Save test metrics (seen and unseen plates)
    """
    seed = config["seed"]
    model_name = config["model_name"]

    #load probing_df
    eval_data_dir = config["eval_data_dir"]
    eval_data_file_name =  config["eval_data_file_name"]
    probing_df = pd.read_csv(eval_data_dir+eval_data_file_name+'_'+str(model_name)+'.csv')

    #subset into train,val,test,test_out 
    train_df = probing_df.loc[probing_df['split']=='train']
    val_df = probing_df.loc[probing_df['split']=='val']
    test_df = probing_df.loc[probing_df['split']=='test']
    test_out_df = probing_df.loc[probing_df['split']=='test_out_plate_']

    #subset train into 10 train splits 
    Nruns = config["num_runs"]
    list_of_train_dfs = [pd.DataFrame() for _ in range(Nruns)]
    for target_value in train_df['TARGET'].unique():
        train_df_target_rows = train_df.loc[train_df['TARGET']==target_value]

        nrun_splits = np.array_split(train_df_target_rows,10)
        for i in range(Nruns):
            list_of_train_dfs[i] = pd.concat([list_of_train_dfs[i],nrun_splits[i]])


    #Create dataloaders for val,test and test_out 
    last_layer = "embedding"
    feature_cols = [col for col in train_df.columns if col.startswith(last_layer)]
    data_val = val_df[feature_cols].values
    label_val = val_df["TARGET"].values

    data_test = test_df[feature_cols].values
    label_test = test_df["TARGET"].values

    data_out = test_out_df[feature_cols].values
    label_out = test_out_df["TARGET"].values

    #run linear probe for a train split
    log_dir = config["log_dir"]
    test_metrics = []
    test_out_metrics = []
    for i in range(Nruns):
        train_data = list_of_train_dfs[i]
        data_train = train_data[feature_cols].values
        label_train =train_data["TARGET"].values

        tm,tom = train_catboost(seed,log_dir,
                    train_data=data_train,
                    train_labels=label_train,
                    test_data=data_test,
                    test_labels=label_test,
                    out_data=data_out,
                    out_labels=label_out)

        test_metrics.append(tm)
        test_out_metrics.append(tom)

    #Save
    metrics_dir = config["metrics_dir"]
    with open(metrics_dir+'agg_test_metrics_'+model_name+'.pkl', 'wb') as handle:
        pickle.dump(test_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(metrics_dir+'agg_test_out_metrics_'+model_name+'.pkl', 'wb') as handle:
        pickle.dump(test_out_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, default="config_run.yaml", help="Config file"
    )

    args = parser.parse_args()

    yaml_file = open(f"{args.config}")
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    main(config)
