# Required Screenshots (Submission Section 6)

According to Section 6 of the assignment, the screenshots/ folder must contain three images:

1. mlflow_compare_view.png:
   - Open MLflow UI at http://127.0.0.1:5000.
   - Go to experiment 'hf-sentiment-champion-challenger'.
   - Select runs 'A_sentiment_distilbert-sst2' and 'B_sentiment_roberta-large'.
   - Click the 'Compare' button.
   - Capture a screenshot showing side-by-side metric comparisons (Accuracy, Macro-F1, Latency).

2. registry_aliases.png:
   - In MLflow UI, go to the 'Models' tab.
   - Select registered model 'sentiment-champion-challenger'.
   - Capture a screenshot showing the versions and assigned aliases (@champion and @challenger).

3. terminal_predict_output.png:
   - Run 'python predict_from_mlflow.py' in the terminal.
   - Capture a screenshot of the output showing resolved alias metadata and the prediction table.
