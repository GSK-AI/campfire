###
import argparse
import pickle
from typing import Dict

import numpy as np 
import pandas as pd
import yaml

import torch 
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import balanced_accuracy_score,accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

import pytorch_lightning as pl 
from  pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torch.optim.lr_scheduler import StepLR
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F

class LinearClassifier(pl.LightningModule):
    """ Linear Classifier"""
    def __init__(
        self,
        embed_dimension,
        num_outputs,):

        super().__init__()

        self.embed_dimension = embed_dimension
        self.num_outputs = num_outputs

        self.linear_layer = nn.Linear(self.embed_dimension,self.num_outputs)
        self.softmax = torch.softmax

    def configure_optimizers(self):
        """Configure Adam optimizer"""
        optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.5)
        return [optimizer], [scheduler]

    def forward(self,x):

        return self.linear_layer(x)

    def training_step(self, batch, batch_idx):

        inputs,targets = batch

        outputs = self.linear_layer(inputs)


        class_prob = self.softmax(outputs, dim=1)
        y_pred = torch.argmax(class_prob, dim=1)

        loss = F.cross_entropy(outputs, targets)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):

        inputs,targets = batch

        outputs = self.linear_layer(inputs)

        print(outputs.shape)

        class_prob = self.softmax(outputs, dim=1)
        y_pred = torch.argmax(class_prob, dim=1)

        loss = F.cross_entropy(outputs, targets)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        bal_acc = accuracy_score(targets.cpu(),y_pred.cpu())

        self.log("val_accuracy", bal_acc, prog_bar=True, on_step=False, on_epoch=True)


        return loss

    def test_step(self, batch, batch_idx):

        inputs,targets = batch

        outputs = self.linear_layer(inputs)

        class_prob = self.softmax(outputs, dim=1)
        y_pred = torch.argmax(class_prob, dim=1)
        
        loss = F.cross_entropy(outputs, targets)

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        bal_acc = accuracy_score(targets.cpu(),y_pred.cpu())

        self.log("test_accuracy", bal_acc, prog_bar=True, on_step=False, on_epoch=True)


        return loss

def get_metrics(model, device, dl_test):
    model.eval()

    targets = []
    predictions = []

    with torch.no_grad():
        for data, target in dl_test:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)  # get the index of the max log-probability

            targets.extend(target.cpu())
            predictions.extend(pred.cpu())

    bal_acc = balanced_accuracy_score(targets,predictions)
    acc = accuracy_score(targets,predictions)
    cm = confusion_matrix(targets,predictions,normalize='pred')

    metric_dict = {
        "balanced accuracy": bal_acc,
        "accuracy": acc,
        "confusion matrix": cm
    }
    return metric_dict 

def run_linear_probe(seed,log_dir,num_epochs,patience,embed_dim,num_classes,dl_train,dl_test,dl_out):

    torch.manual_seed(seed)

    model = LinearClassifier(embed_dim,num_classes)
    
    # training
    early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=patience, verbose=False, mode="max")
    trainer = pl.Trainer(gpus=1, max_epochs=num_epochs,default_root_dir=log_dir,
        enable_model_summary=False,callbacks=[early_stop_callback],logger=False)

    trainer.fit(model,dl_train,dl_test)

    test_metrics = get_metrics(model, 'cpu', dl_test)

    test_out_metrics = get_metrics(model, 'cpu', dl_out)

    return test_metrics,test_out_metrics


def main(config) -> None:
    """
    Main Function:
    """

    seed = config["seed"]
    model_name = config["model_name"]

    #load probing_df
    eval_data_dir = config["eval_data_dir"]
    eval_data_file_name =  config["eval_data_file_name"]
    probing_df = pd.read_csv(eval_data_dir+eval_data_file_name+'_'+str(model_name)+'.csv')

    #subset into train,val,test,test_out 
    held_out_compound_df = probing_df.loc[probing_df['split']=='test_out_compound']
    held_out_all_df = probing_df.loc[probing_df['split']=='test_out_all_']


    num_epochs = config["num_epochs"]
    log_dir = config["log_dir"]
    patience = config["patience"]

    num_folds = config["num_folds"]


    skf = StratifiedKFold(n_splits=num_folds)

    last_layer = config["last_layer"]
    feature_cols = [col for col in held_out_compound_df.columns if col.startswith(last_layer)]
    X = held_out_compound_df[feature_cols].values
    y = held_out_compound_df['TARGET'].values
    print("X shape: {}, y shape: {}".format(X.shape,y.shape),flush=True)

    X_out = held_out_all_df[feature_cols].values
    y_out = held_out_all_df['TARGET'].values
    embed_dim = X_out.shape[1]
    num_classes = len(np.unique(y))
    test_metrics = []
    test_out_metrics = []
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):

        print("Fold {}/{}".format(i+1,num_folds),flush=True)

        train_x = X[train_index]
        train_y = y[train_index]
        test_x = X[test_index]
        test_y = y[test_index]

        train_x = torch.tensor(train_x,dtype=torch.float32)
        train_y = torch.tensor(train_y,dtype=torch.int64)
        test_x = torch.tensor(test_x,dtype=torch.float32)
        test_y = torch.tensor(test_y,dtype=torch.int64)

        test_ood_x = torch.tensor(X_out,dtype=torch.float32)
        test_ood_y = torch.tensor(y_out,dtype=torch.int64)


        batch_size = config["batch_size"]
        ds_train, ds_test = TensorDataset(train_x, train_y), TensorDataset(test_x, test_y)
        dl_train, dl_test = DataLoader(ds_train, batch_size=batch_size, shuffle=True), DataLoader(ds_test, batch_size=batch_size, shuffle=False)
        ds_ood = TensorDataset(test_ood_x,test_ood_y)
        dl_ood = DataLoader(ds_ood,batch_size=batch_size, shuffle=False)

        tm,tom = run_linear_probe(seed,log_dir+'_'+str(i),num_epochs,patience,embed_dim,num_classes,
        dl_train,dl_test,dl_ood)

        test_metrics.append(tm)
        test_out_metrics.append(tom)

    #Save
    metrics_dir = config["metrics_dir"]
    with open(metrics_dir+'held_out_compound_test_metrics_'+model_name+'.pkl', 'wb') as handle:
        pickle.dump(test_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(metrics_dir+'held_out_compound_test_ood_metrics_'+model_name+'.pkl', 'wb') as handle:
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
