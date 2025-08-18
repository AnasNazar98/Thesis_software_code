

# lgbm combination alone

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
from sklearn.metrics import median_absolute_error

import os
import time
import tensorflow as tf
import random

day_number = 6

results = []

time_of_day_order = ['Morning', 'Noon', 'Afternoon', 'Evening']

for source in time_of_day_order:
    for target in time_of_day_order:


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
        # Yeo-Johnson 

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





        
        lag_vars = ['Steps'
                        #, "PHYSICAL_NORM", "MENTAL_NORM", "MOTIVATION_NORM", "EFFICACY_NORM", "CONTEXT_NORM"
                        
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





        def make_lags(base, step=4, max_lag=length):
            return [i for i in range(base, max_lag + 1, step)]

        shift_mapping = {
            ('Morning', 'Noon'): make_lags(1),
            ('Morning', 'Afternoon'): make_lags(2),
            ('Morning', 'Evening'): make_lags(3),
            ('Morning', 'Morning'): make_lags(4),

            ('Noon', 'Morning'): make_lags(3),
            ('Noon', 'Afternoon'): make_lags(1),
            ('Noon', 'Evening'): make_lags(2),
            ('Noon', 'Noon'): make_lags(4),

            ('Afternoon', 'Morning'): make_lags(2),
            ('Afternoon', 'Noon'): make_lags(3),
            ('Afternoon', 'Evening'): make_lags(1),
            ('Afternoon', 'Afternoon'): make_lags(4),

            ('Evening', 'Morning'): make_lags(1),
            ('Evening', 'Noon'): make_lags(2),
            ('Evening', 'Afternoon'): make_lags(3),
            ('Evening', 'Evening'): make_lags(4),
        }

        relevant_lags = shift_mapping.get((source, target), [])

        all_lagged_cols = garmin.filter(regex=r'_lag_\d+$').columns.tolist()

        lag_cols = [col for col in all_lagged_cols 
                    if int(col.split('_')[-1]) in relevant_lags]
        lag_cols


        selected_columns = ['participant_id', 'day', 'hours_cat', 'timestep', 'Steps', 'Steps_original'
                            , 'Morning', 'Noon', 'Afternoon', 'Evening'] + lag_cols

        garmin_filtered = garmin[selected_columns]

        garmin_filtered = garmin_filtered.query("hours_cat == @target")

        

        time_of_day_mapping = {
            'Morning': 0,
            'Noon': 1,
            'Afternoon': 2,
            'Evening': 3
        }



        time_of_day_order = ['Morning', 'Noon', 'Afternoon', 'Evening']
        time_of_day_mapping = {v: i for i, v in enumerate(time_of_day_order)}

        
        length_base = length 

        if time_of_day_mapping[source] == time_of_day_mapping[target] or time_of_day_mapping[source] > time_of_day_mapping[target]:
            length = length_base
        else:
            length = length_base - 3

        garmin = garmin_filtered



        data_train = (garmin
                    .query("participant_id in @train_ids and timestep > @length")
                    .dropna(subset=['Steps']))
        data_val = (garmin
                    .query("participant_id in @val_ids and timestep > @length")
                    .dropna(subset=['Steps']))
        data_test = (garmin
                    .query("participant_id in @test_ids and timestep > @length")
                    .dropna(subset=['Steps']))

        



        (data_train
        .agg(mean = ('Steps', 'max')))
        (data_val
        .agg(mean = ('Steps', 'max')))
        (data_test
        .agg(mean = ('Steps', 'max')))


        


        lagged_features = garmin.filter(regex=r"_lag_\d+$").columns.tolist()


        other_features = ['hours_cat']
        time_of_day_features = ['Noon', 'Afternoon', 'Evening']

        cycle = ['cycle_day_sin', 'cycle_day_cos']

        features = (
                    lagged_features
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




        model = lgb.LGBMRegressor(
            n_estimators=3000,
            num_leaves=1000,
            max_depth=100,
            min_child_samples=1,  
            min_split_gain=0,
            #subsample=1,
            learning_rate=0.005,
            reg_alpha=0.01,
            reg_lambda=0.01,

            objective='regression_l1',

            random_state=SEED,
            n_jobs=-1,
            verbosity=-1   
        )

        start_time = time.time()

        model.fit(X_train, y_train
                , eval_set=[ (X_val, y_val)

                            ]
                )

        end_time = time.time()

        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)



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

        #evaluate(y_train, y_pred_train)
        #evaluate(y_val, y_pred_val)
        #evaluate(y_test, y_pred_test)

        #lgb.plot_metric(model)

        #lgb.plot_importance(model)

        y_train_inv = step_transformer.inverse_transform(pd.DataFrame({'Steps': y_train}))['Steps']
        y_pred_train_inv = step_transformer.inverse_transform(pd.DataFrame({'Steps': y_pred_train}))['Steps']

        y_val_inv = step_transformer.inverse_transform(pd.DataFrame({'Steps': y_val}))['Steps']
        y_pred_val_inv = step_transformer.inverse_transform(pd.DataFrame({'Steps': y_pred_val}))['Steps']

        y_test_inv = step_transformer.inverse_transform(pd.DataFrame({'Steps': y_test}))['Steps']
        y_pred_test_inv = step_transformer.inverse_transform(pd.DataFrame({'Steps': y_pred_test}))['Steps']


        # Run evaluations
        #evaluate(y_train_inv, y_pred_train_inv, name="Train")
        #evaluate(y_val_inv, y_pred_val_inv, name="Validation")
        #evaluate(y_test_inv, y_pred_test_inv, name="Test")

        metrics = evaluate(y_test_inv, y_pred_test_inv, name="Test")
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ###############################################################################



        
        results.append({
            'source': source,
            'target': target,
            'Set': 'Test',
            **metrics
        })
        
        results
        
        
        output_path = r"C:\Users\anasn\Desktop\E\Semester 4\Thesis\R code\xgboost timeseries\prediction plots xgboost\results_combination_lgbm_alone.xlsx"
        
        results_df = pd.DataFrame(results)
        
        results_df.to_excel(output_path.replace("\\", "/"), index=False)

