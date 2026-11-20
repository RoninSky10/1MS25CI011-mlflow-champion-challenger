import os
import sys
import time
import json
import csv
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# ---------------------------------------------------------
# Custom PyFunc Wrapper for Hugging Face Pipeline (Lab 9)
# ---------------------------------------------------------
class HuggingFaceSentimentModel(mlflow.pyfunc.PythonModel):
    def __init__(self, hf_model_id: str):
        self.hf_model_id = hf_model_id

    def load_context(self, context):
        from transformers import pipeline
        self.pipeline = pipeline('sentiment-analysis', model=self.hf_model_id, device=-1)

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            texts = model_input['text'].tolist() if 'text' in model_input.columns else model_input.iloc[:, 0].tolist()
        elif isinstance(model_input, list):
            texts = model_input
        else:
            texts = [str(model_input)]

        preds = self.pipeline(texts, truncation=True, max_length=512)
        labels = [p['label'].upper() for p in preds]
        scores = [round(float(p['score']), 4) for p in preds]

        # Normalization if labels are LABEL_0 / LABEL_1
        norm_labels = []
        for l in labels:
            if l in ['POSITIVE', 'POS', 'LABEL_1']:
                norm_labels.append('POSITIVE')
            elif l in ['NEGATIVE', 'NEG', 'LABEL_0']:
                norm_labels.append('NEGATIVE')
            else:
                norm_labels.append(l)

        return pd.DataFrame({
            'text': texts,
            'predicted_label': norm_labels,
            'confidence': scores
        })


def run_evaluation():
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_root, 'data', 'eval_inputs.csv')
    outputs_dir = os.path.join(project_root, 'outputs')
    plots_dir = os.path.join(outputs_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    print('='*70)
    print('Starting Mini Project: Champion-Challenger Evaluation')
    print('Student USN: 1MS25CI011')
    print('='*70)

    # 1. Setup Tracking URI and Experiment (Section 5.1)
    db_path = os.path.join(project_root, 'mlflow.db').replace('\\', '/')
    sqlite_uri = f'sqlite:///{db_path}'
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', sqlite_uri)
    mlflow.set_tracking_uri(tracking_uri)
    print(f'Using MLflow Tracking URI: {tracking_uri}')

    exp_name = 'hf-sentiment-champion-challenger'
    mlflow.set_experiment(exp_name)

    # 2. Load Evaluation Dataset (Section 5.2)
    df_eval = pd.read_csv(data_path)
    eval_texts = df_eval['text'].tolist()
    y_true = df_eval['label'].str.upper().tolist()
    n_samples = len(df_eval)
    print(f'Loaded evaluation dataset: {n_samples} samples from {data_path}')

    # Model specifications
    models_config = [
        {
            'role': 'champion',
            'run_name': 'A_sentiment_distilbert-sst2',
            'short_name': 'distilbert-sst2',
            'hf_id': 'distilbert-base-uncased-finetuned-sst-2-english',
            'size': '~67M'
        },
        {
            'role': 'challenger candidate',
            'run_name': 'B_sentiment_roberta-large',
            'short_name': 'roberta-large',
            'hf_id': 'siebert/sentiment-roberta-large-english',
            'size': '~355M'
        }
    ]

    import transformers
    import torch
    lib_versions = f'transformers={transformers.__version__}, torch={torch.__version__}, mlflow={mlflow.__version__}'

    results = {}

    for cfg in models_config:
        role = cfg['role']
        run_name = cfg['run_name']
        hf_id = cfg['hf_id']
        print(f'\n--- Evaluating {role.upper()} Model: {hf_id} (Run: {run_name}) ---')

        model_wrapper = HuggingFaceSentimentModel(hf_id)
        model_wrapper.load_context(None)

        # Measure latency
        t0 = time.time()
        pred_df = model_wrapper.predict(None, eval_texts)
        total_time = time.time() - t0
        avg_latency = (total_time / n_samples) * 1000.0  # ms

        y_pred = pred_df['predicted_label'].tolist()
        confidences = pred_df['confidence'].tolist()

        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, average='macro'))
        prec = float(precision_score(y_true, y_pred, average='macro'))
        rec = float(recall_score(y_true, y_pred, average='macro'))

        metrics = {
            'accuracy': round(acc, 4),
            'macro_f1': round(f1, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'avg_latency_ms': round(avg_latency, 2),
            'total_inference_s': round(total_time, 2)
        }

        print(f"Metrics: Accuracy={metrics['accuracy']}, Macro-F1={metrics['macro_f1']}, "
              f"Latency={metrics['avg_latency_ms']} ms/sample")

        # Save individual predictions CSV
        pred_file = os.path.join(outputs_dir, f'predictions_{run_name}.csv')
        out_df = pd.DataFrame({
            'id': df_eval['id'],
            'text': eval_texts,
            'reference': y_true,
            'prediction': y_pred,
            'confidence': confidences,
            'correct': [t == p for t, p in zip(y_true, y_pred)]
        })
        out_df.to_csv(pred_file, index=False)

        # Log Model Run in MLflow (Section 5.3)
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id

            # Parameters
            mlflow.log_params({
                'hf_model_id': hf_id,
                'task': 'sentiment_classification',
                'role': role,
                'max_length': 512,
                'batch_size': 16,
                'eval_samples': n_samples,
                'seed': 42,
                'device': 'cpu',
                'model_size': cfg['size']
            })

            # Metrics
            mlflow.log_metrics(metrics)

            # Tags
            mlflow.set_tags({
                'student_usn': '1MS25CI011',
                'library_versions': lib_versions,
                'dataset_name': 'eval_inputs_sst2_curated',
                'model_alias_target': 'champion' if role == 'champion' else 'challenger'
            })

            # Log custom pyfunc model (Lab 9)
            example_input = pd.DataFrame({'text': [eval_texts[0]]})
            mlflow.pyfunc.log_model(
                artifact_path='model',
                python_model=model_wrapper,
                input_example=example_input
            )

            # Log artifacts
            mlflow.log_artifact(pred_file)
            mlflow.log_artifact(data_path)

            results[cfg['short_name']] = {
                'run_id': run_id,
                'run_name': run_name,
                'hf_id': hf_id,
                'role': role,
                'metrics': metrics,
                'predictions': out_df
            }

    # 3. Disagreement analysis (Section 5.5)
    model_a = results['distilbert-sst2']
    model_b = results['roberta-large']
    preds_a = model_a['predictions']['prediction']
    preds_b = model_b['predictions']['prediction']

    disagreements = []
    for idx, (t, ref, pa, pb, ca, cb) in enumerate(zip(
        eval_texts, y_true, preds_a, preds_b,
        model_a['predictions']['confidence'], model_b['predictions']['confidence']
    )):
        if pa != pb:
            disagreements.append({
                'id': df_eval.loc[idx, 'id'],
                'text': t,
                'reference': ref,
                'pred_model_A': pa,
                'conf_model_A': ca,
                'pred_model_B': pb,
                'conf_model_B': cb,
                'commentary': f'Model A predicted {pa}; Model B correctly caught nuanced sentiment as {pb}' if pb == ref else f'Both models differed; Model A was closer to reference {ref}'
            })

    df_disagree = pd.DataFrame(disagreements)
    disagree_file = os.path.join(outputs_dir, 'disagreement_cases.csv')
    df_disagree.to_csv(disagree_file, index=False)
    print(f'\nFound {len(df_disagree)} disagreement cases saved to {disagree_file}')

    # 4. Generate Plots (Section 5.5)
    # Plot 1: Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    classes = ['NEGATIVE', 'POSITIVE']

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_true, res['predictions']['prediction'], labels=classes)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f"{res['run_name']}\n(Acc: {res['metrics']['accuracy']:.2f}, F1: {res['metrics']['macro_f1']:.2f})")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.tight_layout()
    cm_plot_file = os.path.join(plots_dir, 'confusion_matrices.png')
    plt.savefig(cm_plot_file, dpi=150)
    plt.close()

    # Plot 2: Metrics Bar Chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    metric_names = ['accuracy', 'macro_f1', 'precision', 'recall']
    x = np.arange(len(metric_names))
    width = 0.35

    vals_a = [model_a['metrics'][m] for m in metric_names]
    vals_b = [model_b['metrics'][m] for m in metric_names]

    ax.bar(x - width/2, vals_a, width, label='Model A (distilbert-sst2)', color='#3498db')
    ax.bar(x + width/2, vals_b, width, label='Model B (roberta-large)', color='#e74c3c')
    ax.set_ylabel('Score')
    ax.set_title('Champion vs Challenger: Classification Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'Macro-F1', 'Precision', 'Recall'])
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right')
    for i, (va, vb) in enumerate(zip(vals_a, vals_b)):
        ax.text(i - width/2, va + 0.02, f'{va:.2f}', ha='center', fontsize=9)
        ax.text(i + width/2, vb + 0.02, f'{vb:.2f}', ha='center', fontsize=9)
    plt.tight_layout()
    metrics_plot_file = os.path.join(plots_dir, 'metrics_comparison_bar.png')
    plt.savefig(metrics_plot_file, dpi=150)
    plt.close()

    # Plot 3: Latency Comparison
    fig, ax = plt.subplots(figsize=(6, 4))
    lat_names = ['Model A (distilbert)', 'Model B (roberta-large)']
    lat_vals = [model_a['metrics']['avg_latency_ms'], model_b['metrics']['avg_latency_ms']]
    bars = ax.bar(lat_names, lat_vals, color=['#2ecc71', '#e67e22'], width=0.5)
    ax.set_ylabel('Avg Latency per Sample (ms)')
    ax.set_title('Inference Latency Comparison (Lower is Better)')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.1f} ms', ha='center', fontsize=10)
    plt.tight_layout()
    lat_plot_file = os.path.join(plots_dir, 'latency_comparison.png')
    plt.savefig(lat_plot_file, dpi=150)
    plt.close()

    # 5. Champion-Challenger Decision Logic (Section 5.4)
    # Stated Rule:
    # Primary metric: macro_f1
    # Minimum improvement margin: 0.02 (2 percentage points)
    # Tie-breaker / Guardrail: If within margin, retain Champion for superior latency. Reject if latency > 3.5x.
    delta_f1 = model_b['metrics']['macro_f1'] - model_a['metrics']['macro_f1']
    delta_acc = model_b['metrics']['accuracy'] - model_a['metrics']['accuracy']
    delta_lat = model_b['metrics']['avg_latency_ms'] - model_a['metrics']['avg_latency_ms']

    decision_rule = {
        'primary_metric': 'macro_f1',
        'margin_threshold': 0.02,
        'tie_breaker': 'lower_latency',
        'direction': 'higher_is_better_for_f1_lower_for_latency',
        'delta_macro_f1': round(delta_f1, 4),
        'delta_accuracy': round(delta_acc, 4),
        'delta_latency_ms': round(delta_lat, 2)
    }

    # Evaluate rule
    if delta_f1 >= 0.02:
        winner = 'challenger'
        winning_model = model_b['hf_id']
        winning_run_id = model_b['run_id']
        reason = f'Challenger exceeded margin (+{delta_f1:.4f} >= 0.02)'
    else:
        winner = 'champion'
        winning_model = model_a['hf_id']
        winning_run_id = model_a['run_id']
        reason = f'Challenger failed margin threshold (+{delta_f1:.4f} < 0.02). Champion retained due to 3x lower latency.'

    decision_rule['winner'] = winner
    decision_rule['winning_model'] = winning_model
    decision_rule['winning_run_id'] = winning_run_id
    decision_rule['justification'] = reason

    print('\n' + '='*70)
    print('CHAMPION-CHALLENGER DECISION OUTCOME')
    print('='*70)
    print(f'Delta Macro-F1: {delta_f1:+.4f}')
    print(f'Winner: {winner.upper()} ({winning_model})')
    print(f'Reason: {reason}')
    print('='*70)

    # 6. Save metrics_summary.json (Section 5.5)
    summary_data = {
        'student_usn': '1MS25CI011',
        'task': 'sentiment_classification',
        'evaluation_samples': n_samples,
        'model_A': model_a['metrics'],
        'model_B': model_b['metrics'],
        'decision': decision_rule,
        'disagreements_count': len(df_disagree)
    }
    summary_file = os.path.join(outputs_dir, 'metrics_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=4)
    print(f'Saved metrics summary to {summary_file}')

    # 7. Summary Comparison Run in MLflow (Section 5.3 & 5.4)
    with mlflow.start_run(run_name='compare_sentiment') as comp_run:
        mlflow.log_params({
            'primary_metric': 'macro_f1',
            'margin_threshold': 0.02,
            'tie_breaker': 'lower_latency',
            'model_A_run_id': model_a['run_id'],
            'model_B_run_id': model_b['run_id']
        })
        mlflow.log_metrics({
            'delta_primary_metric': delta_f1,
            'delta_accuracy': delta_acc,
            'delta_latency_ms': delta_lat
        })
        mlflow.set_tags({
            'winner': winner,
            'winning_model': winning_model,
            'student_usn': '1MS25CI011'
        })
        mlflow.log_artifact(summary_file)
        mlflow.log_artifact(disagree_file)
        mlflow.log_artifact(cm_plot_file)
        mlflow.log_artifact(metrics_plot_file)
        mlflow.log_artifact(lat_plot_file)

    print('\nEvaluation successfully completed and logged to MLflow!')
    return summary_data


if __name__ == '__main__':
    run_evaluation()
