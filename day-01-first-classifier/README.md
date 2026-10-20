# Day 01 · My first classifier

▶ Recap video: [YouTube Short](https://youtube.com/shorts/CzLfIm7CbMI)

I trained a model that tells 3 species of iris flowers apart (setosa, versicolor, virginica) from 4 measurements per flower: sepal length, sepal width, petal length and petal width.

Notebook: [Classifier_Irisdataset.ipynb](Classifier_Irisdataset.ipynb)

## What I did

| Step | What happens |
|---|---|
| 1. Load | `load_iris()`: 150 flowers, 4 features, 3 species |
| 2. Split | 80% train / 20% test. `stratify=y` keeps the species balanced in both sets |
| 3. Build | `make_pipeline(StandardScaler(), LogisticRegression())`: scale every feature, then learn the class boundaries |
| 4. Evaluate | Accuracy and confusion matrix on the 30 test flowers |
| 5. Compare | Same data with `RandomForestClassifier()` |
| 6. Predict | A new flower `[5.1, 3.5, 1.4, 0.2]` |

## Results

| Model | Test accuracy |
|---|---|
| Logistic regression (scaled) | **93.3%** (28 / 30) |
| Random forest | 93.3% on my run* |

Confusion matrix (rows = real species, columns = predicted):

                  setosa  versicolor  virginica
    setosa          10        0           0
    versicolor       0        9           1
    virginica        0        1           9

Setosa is always right. The 2 mistakes are versicolor and virginica being swapped once each, because those 2 species have overlapping measurements.

The new flower `[5.1, 3.5, 1.4, 0.2]` is predicted as **setosa**.

\* `RandomForestClassifier()` has no `random_state`, so its score can change between runs. I got 93.3%; other runs can give 90% or 96.7%.

## What I learned

- **Split before you train.** The test set must stay unseen, or the score means nothing.
- **Scale your features.** `StandardScaler` puts every measurement on the same scale, which helps linear models like logistic regression.
- **Read the confusion matrix, not just accuracy.** 93.3% doesn't tell you which species get mixed up. The matrix does.
