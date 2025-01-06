#!/bin/bash
#SBATCH --partition=preempted-gpu
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
CONDA_ENV=aiml_cell_imaging_all
module purge
module load miniconda/23.1.0-1

source activate aiml_cell_imaging_all
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



# python run_linear_pipeline.py 

# python plotting/model_comparisons.py -c configs/config.yaml

# python plot_batch_generalisation.py -c config.yaml

# python plot_channel_integration.py -c config.yaml

# python run_held_out_linear_pipeline.py 

# python create_controls.py -c controls_config.yaml

# python runners/run_held_out_well_level_pipeline.py 

# python runners/run_held_out_linear_pipeline.py 

# python runners/run_control_well_level_pipeline.py 

# python runners/run_linear_pipeline.py 

# python plotting/model_comparisons.py -c configs/unseen_channel/held_out/dino_vits8.yaml

python plotting/plot_channel_integration.py -c configs/channel_integration/nucleus.yaml

# python plotting/plot_batch_generalisation.py -c configs/batch_generalisation/held_out/10_plates.yaml

# python modelling/macrophage_embeddings.py -c configs/finetuning/config_fvit_head.yaml


# python modelling/concat_vs_feedthrough.py -c configs/finetuning/channel_integration/config_fvit_feedthrough.yaml

# python modelling/macrophage_zprime.py -c configs/finetuning/config_fvit_head.yaml