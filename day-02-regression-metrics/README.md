# Day 02 · Regression and evaluation metrics

▶ Recap video: [YouTube Short](https://youtube.com/shorts/AazKTwSB940)

Day 1 predicted a label (which iris species). Today I predicted a number: how much electricity a day uses, in kWh. The data is a simulated dataset of 500 days with 3 features: operating hours, device count and average temperature.

Notebook: [Power_Regression.ipynb](Power_Regression.ipynb)

## What I did

| Step | What happens |
|---|---|
| 1. Simulate | 500 days: `operating_hours` from 1 to 12, `device_count` from 1 to 20, `avg_temp_c` from 18 to 35 °C |
| 2. Target | `energy_kwh = 1.8 × hours + 0.35 × devices + 0.08 × temp + noise`, with noise from `rng.normal(0, 8.0, n)` |
| 3. Split | 80% train / 20% test (`random_state=42`), so 100 test days |
| 4. Build | `LinearRegression()` |
| 5. Baseline | `DummyRegressor(strategy="mean")`: always predicts the average of the training set |
| 6. Evaluate | MAE, RMSE and R² for both models on the test set |
| 7. Predict | A new day: 8 hours, 10 devices, 30 °C |

## Results

I ran the same code twice, changing only the noise level (`rng.normal(0, 2.5, n)`, then `rng.normal(0, 8.0, n)`). The notebook shows the 8.0 run.

| Noise | Model | MAE (kWh) | RMSE (kWh) | R² |
|---|---|---|---|---|
| 2.5 | Baseline (predict mean) | 5.227 | 6.549 | -0.000 |
| 2.5 | Linear regression | **2.411** | **2.941** | **0.798** |
| 8.0 | Baseline (predict mean) | 8.280 | 10.398 | -0.000 |
| 8.0 | Linear regression | **7.715** | **9.410** | **0.181** |

The new day (8 hours, 10 devices, 30 °C) is predicted at **20.21 kWh** with noise 2.5 and **20.02 kWh** with noise 8.0.

Same model, same code. With more noise, MAE and RMSE go up and R² falls from 0.80 to 0.18. The model still beats the baseline, but by much less: the pattern is still there, just buried under more noise.

## The 3 metrics

- **MAE** (mean absolute error): how many kWh the prediction is off, on average.
- **RMSE** (root mean squared error): errors are squared before averaging, so big misses count more. RMSE well above MAE means there are some large errors.
- **R²**: `1 − (model's squared error) / (squared error of always predicting the mean)`. Near 1 is good, near 0 means no better than the mean, and below 0 means worse than the mean. The baseline predicts the *training* mean, so its test R² can be slightly different from 0 (here −0.000).

## What I learned

- **Beat a baseline first.** A model only means something if it does better than always guessing the average.
- **Watch RMSE for big misses.** Two models with the same MAE can make very different large errors.
- **Noise limits every model.** When the data gets noisier, the same model scores worse, because there is less signal to learn.
