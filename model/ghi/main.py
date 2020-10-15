"""Main file for ML process"""

# Skelarn
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

# Xgboost
from xgboost import XGBRegressor

# Utilities
import pandas as pd
import numpy as np
import time
from ai_tools import load, tools, plot


# Final models (from previous analisys).
rfr = RandomForestRegressor(
    bootstrap=True,
    min_samples_leaf=2,
    n_estimators=120,
    min_samples_split=2,
    max_depth=12,
    max_leaf_nodes=600,
)
hgbr = BestHGBR = HistGradientBoostingRegressor(
    l2_regularization=0.001,
    learning_rate=0.05,
    max_iter=1000,
    max_leaf_nodes=500
)
ann = MLPRegressor(
    activation='relu',
    alpha=0.0001,
    learning_rate_init=0.01,
    beta_1=0.9,
    beta_2=0.9
)
xgbr = XGBRegressor(
    booster='gbtree',
    colsample_bytree=1,
    learning_rate=0.1,
    max_depth=8,
    min_child_weight=1,
    n_estimators=100,
    reg_alpha=1,
    reg_lambda=1,
    subsample=1,
    tree_method='gpu_hist'
)

estimators = {
    'XGBR': xgbr,
    'HGBR': hgbr,
    'ANN': ann,
    'RFR': rfr,
}


# Process
@tools.chronometer
def main():
    """All main process."""

    # Data for final train
    X, y = load.load_csv_xy(
        filename='./data/rad.csv',
        target='GHI'
    )

    # Data for final test
    tests = {
        'Torreon': './data/test_torreon.csv',
        'SantaAna': './data/test_santaana.csv',
        'Madrid': './data/test_madrid.csv'
    }
    for name, filename in tests.items():
        tests[name] = load.load_csv_xy(filename, 'GHI')

    # Fit final models
    times_for_fit = {}
    scores = {}
    for name, estimator in estimators.items():
        model, time_fit = _fit_model(estimator, X, y)
        times_for_fit[name] = time_fit
        tools.save_model(model, name)

        scores[name] = {}
        for place, dataset in tests.items():
            pred_control = model.predict(dataset[0])
            pred_train = model.predict(X)
            scores[name][place] = {
                'Train': {
                    'R2': r2_score(y, pred_train),
                    'RMSE': tools.rmse_score(y, pred_train),
                    'kWaño/m2': dataset[1].sum()/(24*365)
                },
                'Test': {
                    'R2': r2_score(dataset[1], pred_control),
                    'RMSE': tools.rmse_score(dataset[1], pred_control),
                    'kWaño/m2': pred_control.sum()/(24*365)
                },
            }

            # Save metadata for results
            _save_metada(
                model_name=name,
                data=scores[name][place],
                time=times_for_fit[name],
                place=place
            )

            # Plotting results.
            plot.scattering_performance_plot(
                y_true=dataset[1],
                y_pred=pred_control,
                model_name=name,
                title=place,
                save=True
            )
            plot._ghi_comparation_plot(
                y_true=dataset[1],
                y_pred=pred_control,
                title='Comparación para ' + place,
                model_name=name,
                place=place,
                save=True
            )
            plot._ghi_interactive_plot(
                y_true=dataset[1],
                y_pred=pred_control,
                place=place,
                model=name,
                save=True
            )

        # To monitorize process.
        tools._print_result(
            title=f'Time ({name})',
            res='Time [seconds]',
            value=times_for_fit[name]
        )


def _fit_model(model, X, y):
    """Fit model and return it."""

    initial_time = time.time()
    fited_model = model.fit(X, y)
    final_time = time.time()
    elapsed_time = final_time - initial_time

    return fited_model, elapsed_time


def _save_metada(model_name, data, time, place):
    """Save data for model performance."""

    from io import open

    performance_data = f"""
    --------- Modelo: {model_name} ---------

    * Locación: {place}

    * Entrenamiento:
        R2 = {data['Train']['R2']}
        RMSE = {data['Train']['RMSE']}
        kWaño/m2 = {data['Train']['kWaño/m2']}
    
    * Test:
        R2 = {data['Test']['R2']}
        RMSE = {data['Test']['RMSE']}
        kWaño/m2 = {data['Test']['kWaño/m2']}

    - Tiempo de entrenamiento: {time}
    """

    with open(f'./meta/{model_name}_{place}.txt', 'a+', encoding='utf-8') as f:
        f.write(performance_data)


if __name__ == "__main__":

    main()
