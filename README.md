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