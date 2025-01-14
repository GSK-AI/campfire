import subprocess

scripts_in_order = [
    'data_collection/get_held_out_compound_data.py',
    'modelling/held_out_linear_probing.py',
    # 'plotting/plot_held_out_comparison.py'
]

for script in scripts_in_order:
    subprocess.check_call(['python', script, '-c', 'example_configs/held_out_compound_task_config.yaml'])