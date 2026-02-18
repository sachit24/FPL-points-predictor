import sys
sys.path.append('../Prediction Models')
sys.path.append('../Selection Models')

from fpl_data_loader import FPLDataLoader
from linear_prediction_model import LinearPredictionModel
from IterativeDeepeningTeamSelector import IterativeDeepeningTeamSelector

if __name__ == "__main__":
    # Configuration
    target_gw = None  # None = predict next GW, or set to specific GW for backtesting
    budget = 100.0    # FPL budget (£100m)
    
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load data
    loader = FPLDataLoader()
    raw_df = loader.load_data()
    print(f"Loaded {len(raw_df)} player-gameweek records")
    print("Sample data:")
    print(raw_df.head())
    print("Columns:", raw_df.columns.tolist())

    print("\n" + "="*80)
    print("TRAINING MODEL")
    print("="*80)
    
    # Train prediction model
    model = LinearPredictionModel()
    model.train(raw_df, target_gw)
    
    print("\n" + "="*80)
    print("GETTING PREDICTIONS")
    print("="*80)
    
    # Get predictions
    predictions = model.predict(raw_df, target_gw)
    print(f"Generated predictions for {len(predictions)} players")
    
    print("\n" + "="*80)
    print("BUILDING OPTIMAL TEAM")
    print("="*80)
    
    # Build team using iterative deepening selector
    selector = IterativeDeepeningTeamSelector()
    result = selector.rebuild_team(predictions, budget)
    
    team = result['team']
    remaining_budget = result['budget']
    
    print(f"\nTeam built successfully!")
    print(f"Remaining budget: £{remaining_budget:.1f}m")
    print(f"Total predicted points: {team['predicted_points'].sum():.2f}")
    
    print("\n" + "-"*80)
    print("TEAM BREAKDOWN")
    print("-"*80)
    
    positions = {1: 'Goalkeepers', 2: 'Defenders', 3: 'Midfielders', 4: 'Forwards'}
    
    for pos_id, pos_name in positions.items():
        pos_players = team[team['element_type'] == pos_id].sort_values('predicted_points', ascending=False)
        print(f"\n{pos_name}:")
        for _, player in pos_players.iterrows():
            print(f"  {player['web_name']:20s} | £{player['now_cost']:.1f}m | {player['predicted_points']:.2f} pts")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total cost: £{team['now_cost'].sum():.1f}m")
    print(f"Budget remaining: £{remaining_budget:.1f}m")
    print(f"Expected points: {team['predicted_points'].sum():.2f}")