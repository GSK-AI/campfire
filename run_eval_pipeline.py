import subprocess

scripts_in_order = [
    'get_aggregate_data.py',
    'evaluate_aggregate_embeddings.py',
    'plot_pca_embeddings.py'
]

for script in scripts_in_order:
    subprocess.check_call(['python', script, '-c', 'config.yaml'])