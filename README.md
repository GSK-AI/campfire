# channel_agnostic_vit
Repository for code used to run experiments in "Out-of-distribution evaluations of channel agnostic masked autoencoders in fluorescence microscopy"

## Generating Results

To generate the results of the experiments in our manuscript, we provide the code to run downstream evaluations of our model. Although we do not provide the code to train \textbf{Campfire}, by providing this code, it is possible, given a directory to the embeddings of data from the JUMP-CP dataset, to compare a model to ours using the same downstream criteria. 


### 1. Generating training, validation, and held-out splits. 

Run the following command 
```
python create_controls.py -c controls_config.yaml
```
This will generate a .csv file for each of the TARGET2 and COMPOUND plates in Source 3. These .csv files are plate maps, such that each cell will describe the position on the 384-well plate, the compound with which it was stimulated, and the split to which it has been assigned. Using the provided config will generate the same set of data splits used to train \textbf{Campfire}. 

### 2. Given a set of embeddings, run linear probing, predicting 1-of-9 controls 

In our manuscript, we evaluate models for fluorscence microscopy by training a linear layer with single cell embeddings to predict 1-of-9 controls. 
By running `python runners/run_linear_pipeline.py` will trigger a pipleline, that via a config file, set within `runners/run_linear_pipeline.py`, takes a .csv file containing all single cell embeddings for TARGET2 plates, samples single cell embeddings from each well, assigns them to data splits, and then trains several linear layers using different subsets of the training set. The output of this will be the performance metrics for the in-distribution test set and out-of-distribution test set, for the model specified by the config file. 

### 3. Given a set of embeddings, run linear probing, predicting 1-of-60 held-out compounds  
To evaluate models when dealing with images of cells subject to out-of-distribution compounds, we train a linear layer with single cell embeddings to predict 1-of-60 compounds held-out of model pretraining (as specified in the control csv files generated earlier). Within `runners/run_held_out_pipeline.py` set the config file for the model under evaluation (i.e \texbf{Campfire}, DinoViT-S8, etc). After setting config file, run `python runners/run_held_out_pipeline.py` to trigger a pipeline that will take single cell embeddings, sample 30 embeddings from each well in the TARGET2 plates, and train 5 linear layers, using 5 subsets of the training data via cross-fold validation. The output of this will be the performance metrics for the in-distribution test set and out-of-distribution test set, for the model specified by the config file. 
 
To generate results shown in Tab. 3.3, use the following:
```
# python modelling/macrophage_embeddings.py -c configs/finetuning/config_fvit_head.yaml
```


To generate results shown in figure 3.2, use the following:
```
# python modelling/macrophage_embeddings.py -c configs/finetuning/config_fvit_head.yaml
# python modelling/macrophage_embeddings.py -c configs/finetuning/config_dino_head.yaml
```

To generate results shown in figure 3.3, use the following:
```
python modelling/macrophage_zprime.py -c configs/finetuning/config_fvit_head.yaml
```