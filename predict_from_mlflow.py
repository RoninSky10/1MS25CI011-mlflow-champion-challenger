import os
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

def predict_new_data():
    project_root = os.path.dirname(os.path.abspath(__file__))
    new_data_path = os.path.join(project_root, 'data', 'new_data.csv')
    output_path = os.path.join(project_root, 'outputs', 'new_data_predictions.csv')

    print('='*75)
    print('PREDICT FROM MLFLOW USING CHAMPION ALIAS')
    print('='*75)

    # 1. Connect to MLflow Tracking Server / Backend
    db_path = os.path.join(project_root, 'mlflow.db').replace('\\', '/')
    sqlite_uri = f'sqlite:///{db_path}'
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', sqlite_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    print(f'Using MLflow Tracking URI: {tracking_uri}')

    model_name = 'sentiment-champion-challenger'
    champion_alias = 'champion'
    model_uri = f'models:/{model_name}@{champion_alias}'

    # 2. Inspect Model Metadata via MLflow Client (Section 5.6)
    try:
        model_version_info = client.get_model_version_by_alias(model_name, champion_alias)
        version_num = model_version_info.version
        run_id = model_version_info.run_id
        run_data = client.get_run(run_id)
        hf_model_id = run_data.data.params.get('hf_model_id', 'Unknown HF ID')
        model_size = run_data.data.params.get('model_size', 'N/A')

        print(f'Target Registry Model : {model_name}')
        print(f'Requested Alias       : @{champion_alias}')
        print(f'Resolved Model Version: Version {version_num}')
        print(f'Hugging Face Model ID : {hf_model_id} ({model_size})')
        print(f'Source Run ID         : {run_id}')
        print(f'Loading URI           : {model_uri}')
    except Exception as e:
        print(f'Notice: Could not query registry alias ({e}). Attempting direct load...')

    # 3. Load Model STRICTLY through MLflow (No direct Hugging Face imports!)
    print(f'\nLoading model strictly via MLflow URI: {model_uri} ...')
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    print('Model successfully loaded into memory via MLflow!')

    # 4. Load New Data (Section 5.6)
    df_new = pd.read_csv(new_data_path)
    print(f'\nLoaded {len(df_new)} unseen test samples from {new_data_path}')

    # 5. Predict through loaded MLflow Model
    predictions_df = loaded_model.predict(df_new)

    # 6. Format and Display Clean Terminal Output
    print('\n' + '='*85)
    print(f'PREDICTION RESULTS ON NEW DATA (Model: {hf_model_id} [v{version_num}])')
    print('='*85)
    print(f'{"ID":<5} | {"Predicted Label":<15} | {"Confidence":<10} | {"Input Text"}')
    print('-'*85)

    display_rows = []
    for idx, row in predictions_df.iterrows():
        sample_id = df_new.loc[idx, 'id'] if 'id' in df_new.columns else idx + 1
        text_snippet = (row['text'][:55] + '...') if len(row['text']) > 55 else row['text']
        label = row['predicted_label']
        conf = row['confidence']
        print(f'{sample_id:<5} | {label:<15} | {conf:<10.4f} | {text_snippet}')
        display_rows.append({
            'id': sample_id,
            'text': row['text'],
            'predicted_label': label,
            'confidence': conf,
            'model_version': version_num,
            'hf_model_id': hf_model_id
        })
    print('='*85)

    # 7. Save output to new_data_predictions.csv (Section 5.6)
    out_csv = pd.DataFrame(display_rows)
    out_csv.to_csv(output_path, index=False)
    print(f'\nSaved predictions table to: {output_path}')

if __name__ == '__main__':
    predict_new_data()
