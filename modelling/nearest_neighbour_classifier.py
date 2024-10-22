import numpy as np 
import pandas as pd 
import yaml
import argparse
import pickle

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier


def n_fold_nn_classifier(X_iid,y_iid,X_ood,y_ood,num_folds):
    """
    
    Build a NN classifier with manhattan distance metric, 
    using IID data. Evaluate the accuracy of this classifier
    on IID and OOD data, returning accuracy for both.

    Repeat this for num_folds in stratified k-fold cross
    validation.

    """
    skf = StratifiedKFold(n_splits=num_folds)

    iid_acc = []
    ood_acc = []
    for _, (train_index, test_index) in enumerate(skf.split(X_iid, y_iid)):

        X_train = X_iid[train_index]
        y_train = y_iid[train_index]

        X_test = X_iid[test_index]
        y_test = y_iid[test_index]
        

        neigh = KNeighborsClassifier(n_neighbors=1,metric='manhattan')
        neigh.fit(X_train, y_train)

        y_pred = neigh.predict(X_test)

        iid_acc.append(np.sum(y_pred == y_test)/len(y_test))

        y_pred_ood = neigh.predict(X_ood)

        ood_acc.append(np.sum(y_pred_ood == y_ood)/len(y_ood))


    return iid_acc,ood_acc

def main(config) -> None:
    """
    Main Function:
    """

    seed = config["seed"]
    model_name = config["model_name"]

    #load probing_df
    well_data_dir = config["eval_data_dir"]
    well_data_file_name =  config["eval_data_file_name"]
    well_df = pd.read_csv(well_data_dir+well_data_file_name+'_'+str(model_name)+'.csv')

    iid_split = config["iid_split"]
    ood_split = config["ood_split"]

    iid_df = well_df.loc[well_df['split']==iid_split]
    ood_df = well_df.loc[well_df['split']==ood_split]

    last_layer = config["last_layer"]
    feature_cols = [col for col in well_df.columns if col.startswith(last_layer)]


    X_iid = iid_df[feature_cols].values
    y_iid = iid_df['TARGET'].values

    X_ood = ood_df[feature_cols].values
    y_ood = ood_df['TARGET'].values

    num_folds = config["num_folds"]

    iid_acc,ood_acc = n_fold_nn_classifier(X_iid,y_iid,X_ood,y_ood,num_folds)


    #Save
    metrics_dir = config["metrics_dir"]
    with open(metrics_dir+'nn_iid_accuracy_'+model_name+'.pkl', 'wb') as handle:
        pickle.dump(iid_acc, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(metrics_dir+'nn_ood_accuracy_'+model_name+'.pkl', 'wb') as handle:
        pickle.dump(ood_acc, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, default="config_run.yaml", help="Config file"
    )

    args = parser.parse_args()

    yaml_file = open(f"{args.config}")
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    main(config)
