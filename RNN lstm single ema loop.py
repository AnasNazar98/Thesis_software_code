

# lstm single ema

import numpy as np
import pandas as pd
import itertools as itr
from skimpy import skim
from scipy.stats import iqr
from sklearn.model_selection import train_test_split
from feature_engine.timeseries.forecasting import LagFeatures
from feature_engine.timeseries.forecasting import WindowFeatures
from feature_engine.timeseries.forecasting import ExpandingWindowFeatures
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import time

import tensorflow as tf
from tensorflow import keras
from functools import reduce

import tensorflow as tf
import random



SEED = 99
tf.random.set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

print(tf.__version__)

results = []

days = list(range(1, 8))

for day_number in days:


    SEED = 99
    tf.random.set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)


    garmin = pd.read_excel('C:/Users/anasn/Desktop/E/Semester 4/Thesis/files/Garmin_days_EMA_Anas.xlsx',
                        index_col=0)
    ema = pd.read_csv('C:/Users/anasn/Desktop/E/Semester 4/Thesis/files/EMA_days_Answered_Final.csv'
                    , sep=';'
                    , decimal=',')

    garmin_valid_ids = garmin[garmin['day'] == 14]['participant_id'].unique()

    garmin = (garmin
            .query("day <= 14 and participant_id in @garmin_valid_ids"))


    garmin = (garmin
            .groupby(['participant_id', 'day', 'date', 'hours_cat'])
            .agg(Steps = ("Steps", lambda x: np.sum(x)))
            .sort_values(['participant_id', 'date', 'hours_cat'])
            .reset_index(drop=False))

    garmin['hours_cat'] = pd.Categorical(garmin['hours_cat']
        , categories=['Morning', 'Noon', 'Afternoon', 'Evening'])

    garmin = (garmin
            .sort_values(['participant_id', 'day', 'date', 'hours_cat']))

    participant_id = garmin['participant_id'].unique()
    day = np.arange(1, 15)
    hours_cat = garmin['hours_cat'].unique()

    template = pd.DataFrame(list(itr.product(participant_id, day, hours_cat)), 
                            columns=['participant_id', 'day', 'hours_cat'])

    template['timestep'] = (template
                            .groupby('participant_id')
                            .cumcount() + 1)

    template = pd.merge(template, garmin, on=["participant_id", "day", "hours_cat"]
                        , how='left')

    garmin = template.copy()

    ema["Time_cat"] = pd.Categorical(ema['Time_cat'],
        categories=['Morning', 'Noon', 'Afternoon', 'Evening'])

    ema = (ema
        .rename(columns = {"Time_cat": "hours_cat"}))

    garmin = pd.merge(garmin, ema, how='left',
                    on=["participant_id", "day", "hours_cat"])


    garmin['date'] = (garmin
                    .groupby(["participant_id", "day"])['date']
                    .transform(lambda x: x.ffill().bfill()))

    garmin.columns


    garmin = (garmin
            .get(['participant_id', 'day', 'hours_cat', 'timestep', 'date',
                    'PHYSICAL_NORM', 'MENTAL_NORM', 'MOTIVATION_NORM', 'EFFICACY_NORM',
                    'CONTEXT_NORM', 'Steps']))



    np.random.seed(SEED)
    shuffled_ids = np.random.permutation(participant_id)  
    n = len(shuffled_ids)

    train_size = int(np.floor(0.7 * n))
    val_size = int(np.floor(0.1 * n))

    train_ids = shuffled_ids[:train_size]
    val_ids = shuffled_ids[train_size:train_size + val_size]
    test_ids = shuffled_ids[train_size + val_size:]

    print(len(train_ids), len(val_ids), len(test_ids))
    print(sorted(train_ids))
    print(sorted(val_ids))
    print(sorted(test_ids))
    ###############################################################################
    #  Yeo-Johnson 
    from feature_engine.transformation import YeoJohnsonTransformer


    steps_train_df = garmin[garmin['participant_id'].isin(train_ids)][['Steps']].dropna()
    step_transformer = YeoJohnsonTransformer(variables=['Steps'])
    step_transformer.fit(steps_train_df)


    garmin['Steps_original'] = garmin['Steps']

    steps_non_null = garmin.loc[garmin['Steps'].notna(), ['Steps']]
    transformed_steps = step_transformer.transform(steps_non_null)

    garmin['Steps_transformed'] = np.nan
    garmin.loc[steps_non_null.index, 'Steps_transformed'] = transformed_steps['Steps']

    garmin['Steps'] = garmin['Steps_transformed']





    mask = -999
    garmin = garmin.fillna(mask)

    lag_vars = ['Steps'
                    , "PHYSICAL_NORM", "MENTAL_NORM", "MOTIVATION_NORM", "EFFICACY_NORM", "CONTEXT_NORM"
    ]

    length = 4*day_number
    lag_range = np.arange(1, length+1).tolist()


    hours_map = {'Morning': 0, 'Noon': 1, 'Afternoon': 2, 'Evening': 3}
    garmin['hours_idx'] = garmin['hours_cat'].map(hours_map)


    garmin = pd.concat([garmin, pd.get_dummies(garmin['hours_cat'])], axis=1)
    garmin[['Morning', 'Noon', 'Afternoon', 'Evening']] = garmin[['Morning', 'Noon', 'Afternoon', 'Evening']].astype(int)







    def make_lag(df):
        lf = LagFeatures(periods=lag_range 
                        #list(range(1, length+1))
                        , variables=lag_vars
                        , missing_values='ignore')
        return lf.fit_transform(df)




    garmin = (
        garmin
        .groupby(['participant_id'])
        .apply(make_lag)
        .reset_index(drop=True)
        )

    garmin.columns




    data_train = (garmin
                .query("participant_id in @train_ids and timestep > @length")
                .query("Steps != @mask"))

    data_val = (garmin
                .query("participant_id in @val_ids and timestep > @length")
                .query("Steps != @mask"))

    data_test = (garmin
                .query("participant_id in @test_ids and timestep > @length")
                .query("Steps != @mask"))




    lagged_features = garmin.filter(regex=r"_lag_\d+$").columns.tolist()



    other_features = ['hours_cat']
    time_of_day_features = ['Noon', 'Afternoon', 'Evening']


    features = (time_of_day_features
                + 
                lagged_features
                )


    sorted_lagged_columns = sorted(
        [col for col in data_train.columns if 'Steps_lag_' in col],
        key=lambda x: int(x.split('_')[-1]),
        reverse=True
    )



    X_train = (data_train
            .get(features))
    y_train = data_train.loc[:, 'Steps']



    X_val = (data_val
            .get(features))
    y_val = data_val.loc[:, 'Steps']


    X_test = (data_test
            .get(features))
    y_test = data_test.loc[:, 'Steps']

    

    step_cols = [f"Steps_lag_{i}" for i in range(length, 0, -1)]
    ema_vars = ["PHYSICAL_NORM", "MENTAL_NORM", "MOTIVATION_NORM", "EFFICACY_NORM", "CONTEXT_NORM"]
    ema_cols = [[f"{var}_lag_{i}" for i in range(length, 0, -1)] for var in ema_vars]
    time_cols = ["Noon", "Afternoon", "Evening"]

    steps = X_train[step_cols].values.reshape(-1, length, 1)
    ema_0 = X_train[ema_cols[0]].values.reshape(-1, length, 1)
    ema_1 = X_train[ema_cols[1]].values.reshape(-1, length, 1)
    ema_2 = X_train[ema_cols[2]].values.reshape(-1, length, 1)
    ema_3 = X_train[ema_cols[3]].values.reshape(-1, length, 1)
    ema_4 = X_train[ema_cols[4]].values.reshape(-1, length, 1)
    time = X_train[time_cols].values.reshape(-1, 1, 3)
    time_repeated = np.repeat(time, length, axis=1)
    X_train_seq = np.concatenate([steps
                                , ema_0, ema_1, ema_2, ema_3, ema_4
                                , time_repeated], axis=2)

    steps = X_val[step_cols].values.reshape(-1, length, 1)
    ema_0 = X_val[ema_cols[0]].values.reshape(-1, length, 1)
    ema_1 = X_val[ema_cols[1]].values.reshape(-1, length, 1)
    ema_2 = X_val[ema_cols[2]].values.reshape(-1, length, 1)
    ema_3 = X_val[ema_cols[3]].values.reshape(-1, length, 1)
    ema_4 = X_val[ema_cols[4]].values.reshape(-1, length, 1)
    time = X_val[time_cols].values.reshape(-1, 1, 3)
    time_repeated = np.repeat(time, length, axis=1)
    X_val_seq = np.concatenate([steps
                                , ema_0, ema_1, ema_2, ema_3, ema_4
                                , time_repeated], axis=2)

    steps = X_test[step_cols].values.reshape(-1, length, 1)
    ema_0 = X_test[ema_cols[0]].values.reshape(-1, length, 1)
    ema_1 = X_test[ema_cols[1]].values.reshape(-1, length, 1)
    ema_2 = X_test[ema_cols[2]].values.reshape(-1, length, 1)
    ema_3 = X_test[ema_cols[3]].values.reshape(-1, length, 1)
    ema_4 = X_test[ema_cols[4]].values.reshape(-1, length, 1)
    time = X_test[time_cols].values.reshape(-1, 1, 3)
    time_repeated = np.repeat(time, length, axis=1)
    X_test_seq = np.concatenate([steps
                                , ema_0, ema_1, ema_2, ema_3, ema_4
                                , time_repeated], axis=2)

    X_train = X_train_seq
    X_val = X_val_seq
    X_test = X_test_seq





    from sklearn.utils import shuffle

    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    X_val, y_val = shuffle(X_val, y_val, random_state=42)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)






    train_2d = X_train.reshape(-1, X_train.shape[-1])  
    medians = np.median(train_2d, axis=0)
    iqrs = np.subtract(*np.percentile(train_2d, [75, 25], axis=0))
    iqrs[-4:] = 1.0

    iqrs[iqrs == 0] = 1e-8


    def robust_scale_ignore_mask(X, medians, iqrs, mask_value=-999):
        mask = (X == mask_value)
        X_masked = np.where(mask, np.nan, X)
        X_scaled = (X_masked - medians) / iqrs
        X_scaled[mask] = mask_value

        return X_scaled


    X_train = robust_scale_ignore_mask(X_train, medians, iqrs, mask_value=-999)
    X_val = robust_scale_ignore_mask(X_val, medians, iqrs, mask_value=-999)
    X_test = robust_scale_ignore_mask(X_test, medians, iqrs, mask_value=-999)




    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.metrics import r2_score
    from tensorflow.keras.layers import Masking, GRU, Dense

    X_train = np.array(X_train, dtype=np.float16)
    X_val = np.array(X_val, dtype=np.float16)
    X_test = np.array(X_test, dtype=np.float16)





    model = Sequential([
        Masking(mask_value=mask, input_shape=(X_train.shape[1], X_train.shape[2])),
        
        LSTM(128, return_sequences=True),
        LSTM(64, return_sequences=False),

        Dense(16, activation='relu'),
        Dense(1)
    ])

    from tensorflow.keras.optimizers import Adam

    optimizer = Adam(learning_rate=0.005)

    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])

    early_stop = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )




    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)



    y_pred_train = y_pred_train.flatten()
    y_pred_val = y_pred_val.flatten()
    y_pred_test = y_pred_test.flatten()


    y_train_inv = step_transformer.inverse_transform(pd.DataFrame({'Steps': y_train}))['Steps']
    y_pred_train_inv = step_transformer.inverse_transform(pd.DataFrame({'Steps': y_pred_train}))['Steps']

    y_val_inv = step_transformer.inverse_transform(pd.DataFrame({'Steps': y_val}))['Steps']
    y_pred_val_inv = step_transformer.inverse_transform(pd.DataFrame({'Steps': y_pred_val}))['Steps']

    y_test_inv = step_transformer.inverse_transform(pd.DataFrame({'Steps': y_test}))['Steps']
    y_pred_test_inv = step_transformer.inverse_transform(pd.DataFrame({'Steps': y_pred_test}))['Steps']




    from sktime.performance_metrics.forecasting import MedianAbsolutePercentageError
    from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score

    def evaluate(y_true, y_pred, name=""):
        y_true = pd.Series(y_true).reset_index(drop=True)
        y_pred = pd.Series(y_pred).reset_index(drop=True)

        mae = mean_absolute_error(y_true, y_pred)
        medae = median_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mean_val = np.mean(y_true)
        median_val = np.median(y_true)


        print(f"\n{name} Set Evaluation:")
        print(f"MAE:            {mae:.2f}")
        print(f"MedAE:          {medae:.2f}")
        print(f"R2:             {r2:.2f}")
        print(f"Mean:           {mean_val:.2f}")
        print(f"Median:         {median_val:.2f}")
        print(f"MAE / Mean:     {mae / mean_val:.3f}")
        print(f"MedAE / Median: {medae / median_val:.3f}")
        
        
        return {
            'MAE': round(mae, 2),
            'MedAE': round(medae, 2),
            'R2': round(r2, 2),
            'Mean': round(mean_val, 2),
            'Median': round(median_val, 2),
            'MAE/Mean': round(mae / mean_val, 3),
            'MedAE/Median': round(medae / median_val, 3)
        }


    #evaluate(y_train_inv, y_pred_train_inv, name="Train")
    #evaluate(y_val_inv, y_pred_val_inv, name="Validation")
    #evaluate(y_test_inv, y_pred_test_inv, name="Test")

    metrics = evaluate(y_test_inv, y_pred_test_inv, name="Test")

    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    
    results.append({
        'day': day_number,
        'Set': 'Test',
        **metrics
    })
    
    results
    
    
    output_path = r"C:\Users\anasn\Desktop\E\Semester 4\Thesis\R code\xgboost timeseries\prediction plots xgboost\results_single_rnn_lstm_ema.xlsx"
    
    results_df = pd.DataFrame(results)
    
    results_df.to_excel(output_path.replace("\\", "/"), index=False)

