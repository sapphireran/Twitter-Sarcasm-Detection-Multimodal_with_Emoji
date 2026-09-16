# Recorded 2023 scores

Source: `get_metrics_of_models.ipynb` and `evaluate_loaded_dl_models.ipynb`. W = word only, WE = word + emoji.

| Model | Split | Modality | Accuracy | F1 | Precision | Recall |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| SVM | test | W | 0.7690 | 0.7722 | 0.7617 | 0.7830 |
| SVM | test | WE | 0.7630 | 0.7663 | 0.7558 | 0.7770 |
| SVM | subtest | W | 0.8129 | 0.8523 | 0.8333 | 0.8721 |
| SVM | subtest | WE | 0.8237 | 0.8529 | 0.8820 | 0.8256 |
| DecisionTree | test | W | 0.7265 | 0.7557 | 0.6828 | 0.8460 |
| DecisionTree | test | WE | 0.7295 | 0.7566 | 0.6877 | 0.8410 |
| DecisionTree | subtest | W | 0.7770 | 0.8342 | 0.7723 | 0.9070 |
| DecisionTree | subtest | WE | 0.7986 | 0.8462 | 0.8021 | 0.8953 |
| RandomForest | test | W | 0.8145 | 0.8232 | 0.7862 | 0.8640 |
| RandomForest | test | WE | 0.8180 | 0.8255 | 0.7928 | 0.8610 |
| RandomForest | subtest | W | 0.8058 | 0.8525 | 0.8041 | 0.9070 |
| RandomForest | subtest | WE | 0.8525 | 0.8839 | 0.8619 | 0.9070 |
| GradientBoosting | test | W | 0.7460 | 0.7515 | — | — |
| GradientBoosting | test | WE | 0.7475 | 0.7528 | — | — |
| GradientBoosting | subtest | W | 0.7950 | 0.8376 | — | — |
| GradientBoosting | subtest | WE | 0.7950 | 0.8357 | — | — |
| BiLSTM+Attn | test | W | 0.8635 | 0.8656 | — | — |
| BiLSTM+Attn | test | WE | 0.8735 | 0.8686 | — | — |
| BiLSTM+Attn | subtest | W | 0.8669 | 0.8940 | — | — |
| BiLSTM+Attn | subtest | WE | 0.8921 | 0.9107 | — | — |
