import subprocess

scripts_in_order = [
    'data_collection/get_aggregate_data.py',
    'modelling/evaluate_aggregate_embeddings.py',
    'plotting/plot_pca_embeddings.py'
]

for script in scripts_in_order:
    subprocess.check_call(['python', script, '-c', 'config.yaml'])