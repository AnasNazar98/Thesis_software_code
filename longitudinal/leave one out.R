


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





day_number = 6

all_participant_results = []


    

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
garmin_old = garmin.copy()


#########


###############################################################################
lag_vars = ['Steps'
                #, "PHYSICAL_NORM", "MENTAL_NORM", "MOTIVATION_NORM", "EFFICACY_NORM", "CONTEXT_NORM"
]

length = 4*day_number
lag_range = np.arange(1, length+1).tolist()


hours_map = {'Morning': 0, 'Noon': 1, 'Afternoon': 2, 'Evening': 3}
garmin['hours_idx'] = garmin['hours_cat'].map(hours_map)



garmin = pd.concat([garmin, pd.get_dummies(garmin['hours_cat'])], axis=1)
garmin[['Morning', 'Noon', 'Afternoon', 'Evening']] = garmin[['Morning', 'Noon', 'Afternoon', 'Evening']].astype(int)



#########
all_participant_ids = garmin['participant_id'].unique()

for test_pid in all_participant_ids:
    test_ids = np.array([test_pid])
    
    train_ids = all_participant_ids[all_participant_ids != test_pid]

        
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
    garmin_old = garmin.copy()
    
    
    #########
    
    
    ###############################################################################
    lag_vars = ['Steps'
                    #, "PHYSICAL_NORM", "MENTAL_NORM", "MOTIVATION_NORM", "EFFICACY_NORM", "CONTEXT_NORM"
    ]
    
    length = 4*day_number
    lag_range = np.arange(1, length+1).tolist()
    
    
    hours_map = {'Morning': 0, 'Noon': 1, 'Afternoon': 2, 'Evening': 3}
    garmin['hours_idx'] = garmin['hours_cat'].map(hours_map)
    
    
    
    garmin = pd.concat([garmin, pd.get_dummies(garmin['hours_cat'])], axis=1)
    garmin[['Morning', 'Noon', 'Afternoon', 'Evening']] = garmin[['Morning', 'Noon', 'Afternoon', 'Evening']].astype(int)

    
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
                  .dropna(subset=['Steps']))

    data_test = (garmin
                 .query("participant_id in @test_ids and timestep > @length")
                 .dropna(subset=['Steps']))
    
    
    
    

    lagged_features = garmin.filter(regex=r"_lag_\d+$").columns.tolist()
    
    
  
    other_features = ['hours_cat']
    time_of_day_features = ['Noon', 'Afternoon', 'Evening']
    
    
    features = (time_of_day_features
    
                + lagged_features
                )
    
    
    X_train = (data_train
               .get(features))
    y_train = data_train.loc[:, 'Steps']
    X_train.shape
    
    

    
    
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
    
        random_state=123,
        n_jobs=-1,
        verbosity=-1   
    )
    
    start_time = time.time()
    
    model.fit(X_train, y_train
              
              )
    
    end_time = time.time()
    
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    
    
    
    #lgb.plot_metric(model)
    
    #lgb.plot_importance(model)
    
    
    y_train_inv = step_transformer.inverse_transform(pd.DataFrame({'Steps': y_train}))['Steps']
    y_pred_train_inv = step_transformer.inverse_transform(pd.DataFrame({'Steps': y_pred_train}))['Steps']
    
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

    
    
    # Run evaluations
    #evaluate(y_train_inv, y_pred_train_inv, name="Train")
    evaluate(y_test_inv, y_pred_test_inv, name="Test")
    
    

    import numpy as np
    import pandas as pd

    data_test_eval = data_test.copy()
    data_test_eval['y_true'] = y_test_inv.values
    data_test_eval['y_pred'] = y_pred_test_inv.values

    def safe_error(row):
        actual = row['y_true']
        pred = row['y_pred']
        denom = actual if actual != 0 else 1
        return abs(pred - actual) / denom

    
    data_test_eval['error'] = data_test_eval.apply(safe_error, axis=1)

    participant_success = (
        data_test_eval
        .groupby('participant_id')
        .apply(lambda df: pd.Series({
            'total_predictions': len(df),
            'low_error_count': np.sum(df['error'] <= 0.1),
            'success_rate': np.mean(df['error'] <= 0.1),
            'successful': np.mean(df['error'] <= 0.1) >= 0.8,
            
            'MedAE': median_absolute_error(df['y_true'], df['y_pred']),
            'Median_Steps': np.median(df['y_true']),
            'MedAE_over_Median': median_absolute_error(df['y_true'], df['y_pred']) / np.median(df['y_true']),
        }))
        .reset_index()
    )

    print(participant_success)
    
    participant_success['test_pid'] = test_pid
    all_participant_results.append(participant_success)
    
    final_results = pd.concat(all_participant_results, ignore_index=True)

    output_path = r"C:\Users\anasn\Desktop\E\Semester 4\Thesis\R code\xgboost timeseries\prediction plots xgboost\participant_success_loocv_single.xlsx"
    final_results.to_excel(output_path, index=False)



    
    
