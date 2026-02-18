import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from base_model import BaseModel

class LinearPredictionModel(BaseModel):
    
    def __init__(self):
        self.models = {}
    
    def create_cumulative_features(self, df):
        df = df.copy()
        
        stats_to_accumulate = [
             'minutes', 'goals_scored', 'assists',
            'clean_sheets',  'bps', 'saves', 'penalties_saved',
            'goals_conceded', 'influence', 'creativity', 'threat',
            'expected_goals', 'expected_assists', 
             
            'tackles', 'defensive_contribution'
        ]
        
        for stat in stats_to_accumulate:
            df[f'season_avg_{stat}'] = df.groupby('id')[stat].transform(
                lambda x: x.expanding().mean().fillna(0)
            )
        df = df[df['season_avg_minutes']>0]
        return df

    def create_rolling_features(self, df):
        df = df.copy()
        
        stats_to_roll = [
             'minutes', 'goals_scored', 'assists',
            'clean_sheets',  'bps', 'saves', 'penalties_saved',
            'goals_conceded', 'influence', 'creativity', 'threat',
            'expected_goals', 'expected_assists',
            
            'tackles', 'defensive_contribution'
        ]
        
        for stat in stats_to_roll:
            df[f'{stat}_last3'] = df.groupby('id')[stat].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
            
            df[f'{stat}_last5'] = df.groupby('id')[stat].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )
        
        return df
        
    def get_relevant_features(self, position_type):
        common = ['minutes', 'bps', 'yellow_cards', 'red_cards']
        
        if position_type == 1:
            specific = ['saves', 'penalties_saved', 'clean_sheets', 
                       'goals_conceded', 'expected_goals_conceded', 'defensive_contribution']
        
        elif position_type == 2:
            specific = ['clean_sheets', 'goals_conceded', 
                       'defensive_contribution', 'expected_goals_conceded',
                       'goals_scored', 'assists', 'expected_goals', 
                       'expected_assists',  
                       'influence', 'creativity', 'threat']
        
        elif position_type == 3:
            specific = ['goals_scored', 'assists', 'expected_goals',
                       'expected_assists', 
                       'clean_sheets', 'goals_conceded',
                       'defensive_contribution',
                       'influence', 'creativity', 'threat']
        
        elif position_type == 4:
            specific = ['goals_scored', 'assists', 
                       'expected_goals', 'expected_assists', 
                       'penalties_missed', 
                       'influence', 'creativity', 'threat']
        
        all_features = common + specific
        
        feature_patterns = []
        for stat in all_features:
            feature_patterns.extend([
                f'season_avg_{stat}',
                f'{stat}_last3',
                f'{stat}_last5'
            ])
        
        feature_patterns.append('now_cost')
        
        return feature_patterns

    def train_linear_regression_smart(self, df, position_type):
        df_position = df[df['element_type'] == position_type].copy()
        df_position['target_points'] = df_position.groupby('id')['total_points'].shift(-1)
        df_position = df_position.dropna(subset=['target_points'])
        
        df_position = df_position.sort_values(['id', 'fixture'])
        df_position = df_position[df_position.groupby('id').cumcount() >= 3].copy()
            
        relevant_features = self.get_relevant_features(position_type)
        feature_cols = [col for col in relevant_features if col in df_position.columns]
        
        df_model = df_position[feature_cols + ['target_points']].dropna()
        
        X = df_model[feature_cols]
        y = df_model['target_points']
        
        print(f"\nTraining for Position {position_type}")
        print(f"Number of samples: {len(X)}")
        print(f"Number of features: {len(feature_cols)}")
        
        model = Ridge(alpha=50.0)
        model.fit(X, y)

        coefficients = pd.DataFrame({
            'feature': feature_cols+["intercept"],
            'coefficient': np.append(model.coef_, model.intercept_)
        }).sort_values('coefficient', ascending=False, key=abs)
        
        print(f"\nIntercept: {model.intercept_:.4f}")
        
        return model, coefficients

    def train(self, df: pd.DataFrame, target_gw=None):
        # Filter data if target_gw specified
        if target_gw is not None:
            df = df[df['gameweek'] < target_gw].copy()
        # Create features
        df_with_season = self.create_cumulative_features(df)
        df_features = self.create_rolling_features(df_with_season)
        
        positions = {1: 'Goalkeeper', 2: 'Defender', 3: 'Midfielder', 4: 'Forward'}
        
        for pos_id, pos_name in positions.items():
            print("\n" + "="*80)
            print(f"TRAINING: {pos_name.upper()} (Position {pos_id})")
            print("="*80)
            
            model, coef = self.train_linear_regression_smart(df_features, position_type=pos_id)
            self.models[pos_id] = model
            
            print(f"\nTop 20 Most Important Features for {pos_name}:")
            print(coef.head(20))

    def predict(self, df: pd.DataFrame, target_gw=None) -> pd.DataFrame:
        if target_gw is not None:
            df = df[df['gameweek'] < target_gw].copy()

        df_with_season = self.create_cumulative_features(df)
        df_features = self.create_rolling_features(df_with_season)
        
        all_predictions = []
        
        for pos_id in [1, 2, 3, 4]:
            df_position = df_features[df_features['element_type'] == pos_id].copy()
            
            relevant_features = self.get_relevant_features(pos_id)
            feature_cols = [col for col in relevant_features if col in df_position.columns]
            
            # ✅ Add 'team' to the columns pulled into df_model
            extra_cols = ['web_name', 'fixture', 'id', 'element_type', 'team']
            df_model = df_position[feature_cols + extra_cols].dropna()
            df_latest = df_model.loc[df_model.groupby('id')['fixture'].idxmax()]
            
            X = df_latest[feature_cols]
            df_latest = df_latest.copy()
            df_latest['predicted_points'] = self.models[pos_id].predict(X)
            
            # ✅ Include 'now_cost' (already in feature_cols) and 'team' in output
            output_cols = ['id', 'web_name', 'element_type', 'predicted_points', 'now_cost', 'team']
            all_predictions.append(df_latest[output_cols])
        
        return pd.concat(all_predictions, ignore_index=True)