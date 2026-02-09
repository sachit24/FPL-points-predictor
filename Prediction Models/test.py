#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 02:39:53 2026

@author: sachitsapra
"""

from fpl_data_loader import FPLDataLoader
from linear_prediction_model import LinearPredictionModel


if __name__ == "__main__":
    # Load raw data
    loader = FPLDataLoader()
    raw_df = loader.load_data()
    
    # Train model (handles feature engineering internally)
    model = LinearPredictionModel()
    model.train(raw_df)
    
    # Get predictions (handles feature engineering internally)
    predictions = model.predict(raw_df)
    
    # Display top 20 for each position
    positions = {1: 'Goalkeeper', 2: 'Defender', 3: 'Midfielder', 4: 'Forward'}
    
    for pos_id, pos_name in positions.items():
        print("\n" + "-"*80)
        print(f"TOP 20 {pos_name.upper()}S PREDICTED FOR NEXT GAMEWEEK")
        print("-"*80)
        
        top_players = predictions[predictions['element_type'] == pos_id].nlargest(20, 'predicted_points')
        print(top_players[['web_name', 'predicted_points']].to_string(index=False))