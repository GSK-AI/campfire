#!/bin/bash
#SBATCH --partition=aiml
#SBATCH --mem=64GB
#SBATCH --job-name=fvit_eval
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH --error=/nvme_scratch1/cjh86475/code/channel_agnostic_vit/slurm_outs/%j.err
#SBATCH --output=/nvme_scratch1/cjh86475/code/channel_agnostic_vit/slurm_outs/%j.out
#SBATCH --time=10-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=christian.j.hurry@gsk.com

cd /nvme_scratch1/cjh86475/code/channel_agnostic_vit/

# CONDA_ENV=aiml-cell-imaging  
CONDA_ENV=aiml_active_learning_almo_all
module purge
module load miniconda/23.1.0-1

source activate aiml_active_learning_almo_all
# source activate aiml-cell-imaging   

export PYTHONPATH=/nvme_scratch1/cjh86475/translational_imaging/aiml-cell-imaging/aiml_cell_imaging/src/
export CELL_PAINTING_HOME=/nvme_scratch1/cjh86475/translational_imaging/aiml-cell-imaging/aiml_cell_imaging/src/
export MODEL_BUCKET=model-template
export ORACLE_SID="usprd843"
export ORACLE_USER="CM_SUPPORT"
export ORACLE_PASSWORD="metr0p0lis"


# python get_linear_probe_data.py -c config.yaml

# python linear_probing.py -c config.yaml


# python get_aggregate_data.py -c config.yaml

# python evaluate_aggregate_embeddings.py -c config.yaml


# python run_eval_pipeline.py 


python run_linear_pipeline.py 