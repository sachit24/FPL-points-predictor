import sys
sys.path.append("../Prediction Models")
sys.path.append("../Selection Models")

# ── Concrete strategy imports ─────────────────────────────────────────────────
# To test a different strategy, swap these imports for another class:
#   e.g. from xgboost_model import XGBoostModel
#        from GreedyTeamSelector import GreedyTeamSelector

from linear_prediction_model       import LinearPredictionModel
from IterativeDeepeningTeamSelector import IterativeDeepeningTeamSelector

# ── Other imports ─────────────────────────────────────────────────────────────
from fpl_data_loader import FPLDataLoader
from BestTeamBacktest import Backtester

# =============================================================================
# CONFIGURATION
# =============================================================================

PREDICTION_MODEL = LinearPredictionModel           # Must inherit BaseModel
TEAM_SELECTOR    = IterativeDeepeningTeamSelector  # Must inherit TeamSelector

START_GW         = 10       # First GW to predict (trains on GW1 .. START_GW-1)
BUDGET           = 100.0    # FPL squad budget (£m)
OUTPUT_CSV       = "backtest_results.csv"

# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    # Load raw data ONCE — all GWs, all players
    print("Loading FPL data (this runs once)...")
    loader = FPLDataLoader()
    raw_df = loader.load_data()
#     print(f"Loaded {len(raw_df)} player-gameweek records  "
#           f"(GW1 – GW{raw_df['gameweek'].max()})\n")

    # Instantiate chosen strategies
    model    = PREDICTION_MODEL()
    selector = TEAM_SELECTOR()

    # Run backtest
    backtester = Backtester(model, selector, start_gw=START_GW, budget=BUDGET)
    results    = backtester.run(raw_df)

    # Save and display results
    results.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to '{OUTPUT_CSV}'")
    print(f"\n{results.to_string(index=False)}")