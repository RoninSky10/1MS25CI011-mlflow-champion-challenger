import os
import json
import mlflow
from mlflow.tracking import MlflowClient

def register_models_and_set_aliases():
    project_root = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(project_root, 'outputs', 'metrics_summary.json')

    if not os.path.exists(summary_path):
        print(f'Error: {summary_path} not found. Please run evaluate.py first.')
        return

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    # 1. Connect to Tracking Server / Backend
    db_path = os.path.join(project_root, 'mlflow.db').replace('\\', '/')
    sqlite_uri = f'sqlite:///{db_path}'
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', sqlite_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    print(f'Using MLflow Tracking URI: {tracking_uri}')

    exp = client.get_experiment_by_name('hf-sentiment-champion-challenger')
    if not exp:
        print('Error: Experiment hf-sentiment-champion-challenger not found!')
        return

    # Find the runs for Model A and Model B
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=['attributes.start_time DESC']
    )

    run_a = None
    run_b = None
    for r in runs:
        r_name = r.data.tags.get('mlflow.runName', '')
        if 'A_sentiment' in r_name and run_a is None:
            run_a = r
        elif 'B_sentiment' in r_name and run_b is None:
            run_b = r

    if not run_a or not run_b:
        print('Error: Could not locate runs for both Model A and Model B.')
        return

    model_name = 'sentiment-champion-challenger'
    print(f'Registering models under registry name: {model_name}')

    # Register Model A (Version 1 or new version)
    model_a_uri = f"runs:/{run_a.info.run_id}/model"
    v_a = mlflow.register_model(model_a_uri, model_name)
    print(f"Registered Model A ({run_a.data.params.get('hf_model_id')}) as Version {v_a.version}")

    # Register Model B (Version 2 or new version)
    model_b_uri = f"runs:/{run_b.info.run_id}/model"
    v_b = mlflow.register_model(model_b_uri, model_name)
    print(f"Registered Model B ({run_b.data.params.get('hf_model_id')}) as Version {v_b.version}")

    # Determine winner based on decision in metrics_summary
    winner = summary.get('decision', {}).get('winner', 'champion')
    print(f'\nApplying Champion-Challenger Decision: Winner is {winner.upper()}')

    if winner == 'challenger':
        champ_ver = v_b.version
        chall_ver = v_a.version
        champ_hf = run_b.data.params.get('hf_model_id')
        chall_hf = run_a.data.params.get('hf_model_id')
    else:
        champ_ver = v_a.version
        chall_ver = v_b.version
        champ_hf = run_a.data.params.get('hf_model_id')
        chall_hf = run_b.data.params.get('hf_model_id')

    # Set Aliases (Lab 11)
    client.set_registered_model_alias(model_name, 'champion', champ_ver)
    client.set_registered_model_alias(model_name, 'challenger', chall_ver)

    # Set descriptions and tags
    client.update_model_version(
        name=model_name,
        version=champ_ver,
        description=f'CHAMPION MODEL: {champ_hf}. Evaluated on curated SST-2 dataset by USN 1MS25CI011.'
    )
    client.set_model_version_tag(model_name, champ_ver, 'role', 'champion')
    client.set_model_version_tag(model_name, champ_ver, 'student_usn', '1MS25CI011')

    client.update_model_version(
        name=model_name,
        version=chall_ver,
        description=f'CHALLENGER MODEL: {chall_hf}. Evaluated on curated SST-2 dataset by USN 1MS25CI011.'
    )
    client.set_model_version_tag(model_name, chall_ver, 'role', 'challenger')
    client.set_model_version_tag(model_name, chall_ver, 'student_usn', '1MS25CI011')

    print('\n' + '='*70)
    print('MODEL REGISTRY CONFIGURATION SUMMARY')
    print('='*70)
    print(f'Registered Model: {model_name}')
    print(f'Alias @champion   -> Version {champ_ver} ({champ_hf})')
    print(f'Alias @challenger -> Version {chall_ver} ({chall_hf})')
    print('='*70)
    print('Successfully updated Model Registry and set aliases!')

if __name__ == '__main__':
    register_models_and_set_aliases()
