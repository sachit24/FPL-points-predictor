#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 18:16:45 2025

@author: sachitsapra
"""

import pandas as pd
import numpy as np
import requests
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import json



def get_full_data():
    # API endpoint
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"

    # Make the GET request and convert to dictionary
    response = requests.get(url)
    data = response.json()
    
    return data

def get_gameweek_data(full_data:dict):
    # Game weeks data
    events_df = pd.DataFrame(full_data['events'])
    finished_gws = events_df[events_df['finished']==True]
    all_gameweeks_data = []
    
    for gw_id in finished_gws["id"]:
        gw_url = f"https://fantasy.premierleague.com/api/event/{gw_id}/live/"
        gw_response = requests.get(gw_url)
        gw_data = gw_response.json()
        all_gameweeks_data.append(gw_data["elements"])
    
    # ADD these lines after the loop that appends finished_gws to all_gameweeks_data:
    # next_gw = int(finished_gws["id"].max()) + 1
    # gw_url = f"https://fantasy.premierleague.com/api/event/{next_gw}/live/"
    # gw_response = requests.get(gw_url)
    # gw_data = gw_response.json()
    # all_gameweeks_data.append(gw_data["elements"])
    # print(f"Temporarily included GW{next_gw} live data")

    
    return all_gameweeks_data

def make_gw_df(gameweeks_data, full_data):
    all_players_data = []
    for gw in gameweeks_data:
        for player in gw:
            player_id = player["id"]
            player_stats = player["stats"]
            player_stats['id'] = player_id
            player_stats['fixture'] = player["explain"][0]["fixture"]
            all_players_data.append(player_stats)
    
    df = pd.DataFrame(all_players_data)
    new_column_order = ["id"] + [col for col in df.columns if col != "id"]
    df = df[new_column_order]
    players_info = pd.DataFrame(full_data["elements"])
    df = df.merge(players_info[['id', 'element_type', 'web_name', 'status']], 
                  on='id', how='left')
    
    # Convert object columns to numeric
    object_cols = ['influence', 'creativity', 'threat', 'ict_index', 
                   'expected_goals', 'expected_assists', 
                   'expected_goal_involvements', 'expected_goals_conceded']
    
    for col in object_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.sort_values(['id', 'fixture'])
        
    return df

def create_cumulative_features(df):
    """Calculate season AVERAGES up to each gameweek (not including current)"""
    df = df.copy()
    
    # All stats to track cumulatively
    stats_to_accumulate = [
         'minutes', 'goals_scored', 'assists',
        'clean_sheets',  'bps', 'saves', 'penalties_saved',
        'goals_conceded', 'influence', 'creativity', 'threat',
        'expected_goals', 'expected_assists', 
         
        'tackles', 'defensive_contribution'
    ]
    
    for stat in stats_to_accumulate:
        # Expanding mean (average up to this point) shifted by 1
        df[f'season_avg_{stat}'] = df.groupby('id')[stat].transform(
            lambda x: x.expanding().mean().fillna(0)
        )
    df = df[df['season_avg_minutes']>0]
    return df

def create_rolling_features(df):
    """Create rolling average features for last 3 and 5 gameweeks"""
    df = df.copy()
    
    # Same stats for rolling averages
    stats_to_roll = [
         'minutes', 'goals_scored', 'assists',
        'clean_sheets',  'bps', 'saves', 'penalties_saved',
        'goals_conceded', 'influence', 'creativity', 'threat',
        'expected_goals', 'expected_assists',
        
        'tackles', 'defensive_contribution'
    ]
    
    for stat in stats_to_roll:
        # Last 3 gameweeks average
        df[f'{stat}_last3'] = df.groupby('id')[stat].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        # Last 5 gameweeks average
        df[f'{stat}_last5'] = df.groupby('id')[stat].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
    
    return df


def get_top_players_latest_gameweek(df, model, position_type, top_n=20):
    """Get top N players by predicted points from the LATEST gameweek only"""
    df_position = df[df['element_type'] == position_type].copy()
    # df_position = df_position[df_position['status']=='a']
    df_position['target_points'] = df_position.groupby('id')['total_points'].shift(-1)
    df_position = df_position.dropna(subset=['target_points'])
    
    # Use position-specific features (same as training!)
    relevant_features = get_relevant_features(position_type)
    feature_cols = [col for col in relevant_features if col in df_position.columns]
    
    df_model = df_position[feature_cols + ['target_points', 'web_name', 'fixture', 'id']].dropna()
    
    # Get only the latest fixture for each player
    df_latest = df_model.loc[df_model.groupby('id')['fixture'].idxmax()]
    
    # Make predictions
    X = df_latest[feature_cols]
    df_latest['predicted_points'] = model.predict(X)
    
    # Get top N
    top_players = df_latest.nlargest(top_n, 'predicted_points')
    
    comparison = top_players[['web_name', 'fixture', 'predicted_points', 'target_points']].copy()
    comparison['error'] = comparison['predicted_points'] - comparison['target_points']
    comparison['abs_error'] = abs(comparison['error'])
    
    return comparison

def get_relevant_features(position_type):
    """Get features that actually matter for each position"""
    
    # Common features for all positions (including cards)
    common = [ 'minutes', 'bps',
              'yellow_cards', 'red_cards']
    
    if position_type == 1:  # Goalkeeper
        specific = ['saves', 'penalties_saved', 'clean_sheets', 
                   'goals_conceded', 'expected_goals_conceded',  'defensive_contribution']
    
    elif position_type == 2:  # Defender
        # Defenders: defensive stats + attacking potential
        specific = ['clean_sheets', 'goals_conceded', 
                    
                   'defensive_contribution', 'expected_goals_conceded',
                   'goals_scored', 'assists', 'expected_goals', 
                   'expected_assists',  
                             'influence', 'creativity', 'threat', ]
    
    elif position_type == 3:  # Midfielder
        # Midfielders: attacking + defensive (box-to-box)
        specific = ['goals_scored', 'assists', 'expected_goals',
                   'expected_assists', 
                   'clean_sheets', 'goals_conceded',
                    'defensive_contribution',
                   
                             'influence', 'creativity', 'threat', ]
    
    elif position_type == 4:  # Forward
        # Forwards: attacking stats without defensive work
        specific = ['goals_scored', 'assists', 
                   'expected_goals', 'expected_assists', 
                   
                   'penalties_missed', 
                             'influence', 'creativity', 'threat', ]  # Important for penalty takers
    
    # Build full feature list with all variants
    all_features = common + specific
    
    # Create patterns for season_avg, last3, last5
    feature_patterns = []
    for stat in all_features:
        feature_patterns.extend([
            f'season_avg_{stat}',
            f'{stat}_last3',
            f'{stat}_last5'
        ])
    
    feature_patterns.append('now_cost')  # Add price once (not rolled)
    
    return feature_patterns

def train_linear_regression_smart(df, position_type):
    """Train with position-relevant features only"""
    # Remove last 2 rows for each player (removes GW10 and GW11)
    # df = df.groupby('id').apply(lambda x: x.iloc[:-1]).reset_index(drop=True)
    
    df_position = df[df['element_type'] == position_type].copy()
    df_position['target_points'] = df_position.groupby('id')['total_points'].shift(-1)
    df_position = df_position.dropna(subset=['target_points'])
    
    # drop each player's first 3 appearances (only for training)
    df_position = df_position.sort_values(['id', 'fixture'])
    df_position = df_position[df_position.groupby('id').cumcount() >= 3].copy()
        
    # Get relevant features for this position
    relevant_features = get_relevant_features(position_type)
    
    # Only keep features that exist in the dataframe
    feature_cols = [col for col in relevant_features if col in df_position.columns]
    
    df_model = df_position[feature_cols + ['target_points']].dropna()
    
    X = df_model[feature_cols]
    y = df_model['target_points']
    
    print(f"\nTraining for Position {position_type}")
    print(f"Number of samples: {len(X)}")
    print(f"Number of features: {len(feature_cols)}")
    
    
    model = Ridge(alpha=50.0)  # Instead of LinearRegression()
    model.fit(X, y)

    coefficients = pd.DataFrame({
        'feature': feature_cols+["intercept"],
        'coefficient': np.append(model.coef_,model.intercept_)
    }).sort_values('coefficient', ascending=False, key=abs)
    
    print(f"\nIntercept: {model.intercept_:.4f}")
    
    return model, coefficients

def get_top_players_next_gameweek(df, model, position_type, top_n=20):
    """Get top N players by predicted points for the NEXT gameweek (no actual results yet)"""
    df_position = df[df['element_type'] == position_type].copy()
    
    # Use position-specific features (same as training!)
    relevant_features = get_relevant_features(position_type)
    feature_cols = [col for col in relevant_features if col in df_position.columns]
    
    df_model = df_position[feature_cols + ['web_name', 'fixture', 'id']].dropna()
    
    # Get only the latest fixture for each player
    df_latest = df_model.loc[df_model.groupby('id')['fixture'].idxmax()]
    
    # Make predictions
    X = df_latest[feature_cols]
    df_latest['predicted_points'] = model.predict(X)
    
    # Get top N
    top_players = df_latest.nlargest(top_n, 'predicted_points')
    
    return top_players[['web_name', 'fixture', 'predicted_points']]



# Modify the main section:
if __name__ == "__main__":
    data = get_full_data()
    gameweeks_data = get_gameweek_data(data)
    players_gw_stats_df = make_gw_df(gameweeks_data, data)
    df_with_season = create_cumulative_features(players_gw_stats_df)
    df_features = create_rolling_features(df_with_season)
    
    positions = {1: 'Goalkeeper', 2: 'Defender', 3: 'Midfielder', 4: 'Forward'}
    
    # Determine the next gameweek number
    events_df = pd.DataFrame(data['events'])
    finished_gws = events_df[events_df['finished']==True]
    next_gw = finished_gws['id'].max() + 1 if not finished_gws.empty else 1
    
    # Dictionary to store all coefficients
    all_coefficients = {}
    
    for pos_id, pos_name in positions.items():
        print("\n" + "="*80)
        print(f"TRAINING: {pos_name.upper()} (Position {pos_id})")
        print("="*80)
        
        # Train with position-specific features
        model, coef = train_linear_regression_smart(df_features, position_type=pos_id)
        
        # Show top coefficients
        print(f"\nTop 20 Most Important Features for {pos_name}:")
        print(coef.head(20))
        
        # Convert coefficients DataFrame to dictionary
        coef_dict = dict(zip(coef['feature'], coef['coefficient']))
        
        # Store in main dictionary with position as key
        all_coefficients[str(pos_id)] = {
            'position_name': pos_name,
            'coefficients': coef_dict
        }
        
        # Show top predicted players for the next gameweek (dynamic)
        print("\n" + "-"*80)
        print(f"TOP 20 {pos_name.upper()}S PREDICTED FOR GW{next_gw}")
        print("-"*80)
        
        top_players = get_top_players_next_gameweek(df_features, model, 
                                                      position_type=pos_id, top_n=20)
        print(top_players.to_string(index=False))
    
    # Save all coefficients to JSON
    with open('model_coefficients.json', 'w') as f:
        json.dump(all_coefficients, f, indent=2)
    
    # print("\n" + "="*80)
    # print("✓ Coefficients saved to 'model_coefficients.json'")
    # print("="*80)
    

    
    # for pos_id, pos_name in positions.items():
    #     print("\n" + "="*80)
    #     print(f"TRAINING: {pos_name.upper()} (Position {pos_id})")
    #     print("="*80)
        
    #     # Train with position-specific features (automatically uses GW1-9)
    #     model, coef = train_linear_regression_smart(df_features, position_type=pos_id)
        
    #     # Show top coefficients
    #     print(f"\nTop 20 Most Important Features for {pos_name}:")
    #     print(coef.head(20))
        
    #     # Show top predicted players using GW10 → GW11
    #     print("\n" + "-"*80)
    #     print(f"TOP 20 {pos_name.upper()}S BY PREDICTED POINTS (GW10 → GW11)")
    #     print("-"*80)
        
    #     top_players = get_top_players_latest_gameweek(df_features, model, 
    #                                                     position_type=pos_id, top_n=20)
    #     print(top_players.to_string(index=False))
    #     print(f"\nAverage Absolute Error for Top 20: {top_players['abs_error'].mean():.2f} points")
 #%%
# After all training is done, add this:
print("\n" + "="*80)
print("BRUNO GUIMARÃES FEATURE ANALYSIS")
print("="*80)

# Find Bruno in the dataframe
bruno_df = df_features[df_features['web_name'].str.contains('B.Fernandes', case=False, na=False)]

if not bruno_df.empty:
    # Get his latest row (GW15)
    bruno_latest = bruno_df.iloc[-1]
    
    # Get midfielder features
    relevant_features = get_relevant_features(3)  # 3 = midfielder
    feature_cols = [col for col in relevant_features if col in bruno_latest.index]
    
    print(f"\nBruno's features for GW16 prediction:")
    bruno_features = {}
    for feat in feature_cols:
        bruno_features[feat] = float(bruno_latest[feat])
        print(f"{feat}: {bruno_latest[feat]:.3f}")
    
    # Save to JSON
    with open('bruno_features.json', 'w') as f:
        json.dump(bruno_features, f, indent=2)
    print("\n✓ Bruno's features saved to 'bruno_features.json'")