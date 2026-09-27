# 🚗 Used Car Price Prediction (XGBoost)

A machine learning regression project that predicts a fair estimated price for a used car
based on its brand, exact model, model year, mileage, engine specs, accident history, and
other attributes — trained with **XGBoost** and deployed through an interactive **Gradio** app.

**B.Tech AI & Data Science — Machine Learning Capstone Project**

## Overview

Pricing a used car is often based on guesswork and negotiation, which can lead buyers to
overpay and sellers to under- or over-price their vehicle. This project builds a supervised
regression model to estimate a fair market price from the vehicle's listed attributes.

**Type of ML problem:** Supervised Learning → Regression (target variable: `price`)

### Highlights (v2)

- `model` (the exact car model, not just brand) is used as a real input feature to XGBoost,
  encoded with `OneHotEncoder(min_frequency=5)` to control the high cardinality (~1,898 unique
  values) without discarding the signal.
- The Gradio app has a **Brand → Model cascading dropdown** built directly from the training
  data, plus a **USD → INR** conversion button that reuses the already-predicted price instead
  of re-running the model.

## Pipeline

1. **Load data** — upload/read the used-cars CSV.
2. **Inspect** — shape, dtypes, missing values, and a Brand→Model cardinality check that
   motivates the `min_frequency` choice used later.
3. **Clean** — parse `price`/`milage` out of text, drop unparseable rows, cap extreme price
   outliers (top 1%), and fill missing categorical values with sensible defaults.
4. **Feature engineering** — derive `car_age`, `mileage_per_year`, `engine_horsepower`,
   `engine_liters`, `transmission_simplified`, `is_accident`, `has_clean_title`.
5. **EDA** — price distribution, correlations, and other exploratory plots.
6. **Feature selection** — categorical (`brand`, `model`, `fuel_type`, `transmission_simplified`,
   `ext_col`, `int_col`) and numerical features.
7. **Train/test split** — 80/20.
8. **Preprocessing** — one-hot encoding via `ColumnTransformer`.
9. **Model** — `XGBRegressor` (300 estimators, learning rate 0.08, max depth 6, subsample /
   colsample 0.8) inside a `sklearn` `Pipeline`.
10. **Evaluation** — MAE, RMSE, R².
11. **Feature importance** — top contributing features.
12. **Deployment** — a Gradio web app for interactive price prediction.

## Getting started

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
jupyter notebook Used_Car_Price_Prediction_XGBoost_v2.ipynb
```

The notebook was written for Google Colab (it uses `google.colab.files.upload()` to pick up
the dataset), but falls back to reading a local `used_cars.csv` if one is already present in
the working directory — so it also runs fine in a local Jupyter environment.

### Dataset

The dataset (`used_cars.csv`) is **not included** in this repository (see `.gitignore`).
Place your used-cars CSV in the project root before running the notebook, or upload it when
prompted if running in Colab.

## Tech stack

- pandas, numpy — data handling
- matplotlib, seaborn — visualization
- scikit-learn — preprocessing, pipeline, train/test split, metrics
- XGBoost — regression model
- Gradio — interactive demo app

## Project structure

```
.
├── Used_Car_Price_Prediction_XGBoost_v2.ipynb   # main notebook
├── requirements.txt
└── README.md
```

## License

This project is for educational purposes as part of a B.Tech AI & Data Science capstone.
