import sys
import json
import pandas as pd

sys.path.append('../Prediction Models')
sys.path.append('../Selection Models')

from fpl_data_loader import FPLDataLoader
from linear_prediction_model import LinearPredictionModel
from IterativeDeepeningTeamSelector import IterativeDeepeningTeamSelector

# =============================================================================
# CONFIGURATION
# =============================================================================

target_gw      = None        # None = predict next GW, or set to a specific GW for backtesting
budget         = 100.0       # FPL total budget (£100m) — used when rebuilding from scratch

# ── Transfer suggestion config ────────────────────────────────────────────────
NUM_TRANSFERS  = 2           # How many transfers to suggest (0 = no transfers, max 15)
TEAM_JSON_PATH = "team.json" # Path to the saved team JSON file


# =============================================================================
# HELPERS
# =============================================================================

def validate_num_transfers(n: int, team_size: int) -> None:
    """Raise ValueError if NUM_TRANSFERS is out of range."""
    if not (0 <= n <= 15):
        raise ValueError(f"NUM_TRANSFERS must be between 0 and 15 (got {n}).")
    if n > team_size:
        raise ValueError(
            f"NUM_TRANSFERS ({n}) cannot exceed the number of players in the team ({team_size})."
        )


def enrich_team_from_predictions(team_json: dict, predictions: pd.DataFrame) -> tuple:
    """
    Build a DataFrame for the saved team by matching each player's id against
    the predictions DataFrame, which carries live now_cost, element_type, team, etc.

    Players whose id cannot be found in predictions are kept with placeholder
    values so nothing silently disappears — a warning is printed instead.

    Returns
    -------
    team_df : pd.DataFrame  — enriched team ready for the selector
    bank    : float         — remaining budget from the JSON
    """
    position_map = {"goalkeeper": 1, "defender": 2, "midfielder": 3, "forward": 4}
    pred_index   = predictions.set_index("id")
    bank         = float(team_json.get("budget", 0.0))

    rows = []
    for p in team_json["players"]:
        pid = p["id"]

        if pid in pred_index.index:
            live = pred_index.loc[pid]
            rows.append({
                "id":               pid,
                "web_name":         live["web_name"],
                "element_type":     int(live["element_type"]),
                "now_cost":         float(live["now_cost"]),
                "predicted_points": float(live["predicted_points"]),
                "team":             live["team"],
            })
        else:
            print(f"  ⚠  Player id={pid} ({p['webName']}) not found in predictions — using placeholder values.")
            rows.append({
                "id":               pid,
                "web_name":         p["webName"],
                "element_type":     position_map.get(p["position"], -1),
                "now_cost":         0.0,
                "predicted_points": 0.0,
                "team":             p.get("team", "Unknown"),
            })

    return pd.DataFrame(rows), bank


def print_team(team_df: pd.DataFrame, label: str = "TEAM") -> None:
    positions = {1: "Goalkeepers", 2: "Defenders", 3: "Midfielders", 4: "Forwards"}
    print(f"\n{'='*80}")
    print(label)
    print("="*80)
    for pos_id, pos_name in positions.items():
        players = team_df[team_df["element_type"] == pos_id].sort_values(
            "predicted_points", ascending=False
        )
        if players.empty:
            continue
        print(f"\n{pos_name}:")
        for _, player in players.iterrows():
            print(f"  {player['web_name']:20s} | £{player['now_cost']:.1f}m | {player['predicted_points']:.2f} pts")
    print(f"\nTotal cost    : £{team_df['now_cost'].sum():.1f}m")
    print(f"Expected pts  : {team_df['predicted_points'].sum():.2f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # ── 0. Load team JSON and validate transfer count ────────────────────────
    print("="*80)
    print("LOADING SAVED TEAM")
    print("="*80)
    with open(TEAM_JSON_PATH, "r") as f:
        team_json = json.load(f)
    print(f"Loaded team from '{TEAM_JSON_PATH}' ({len(team_json['players'])} players)")

    validate_num_transfers(NUM_TRANSFERS, len(team_json["players"]))

    # ── 1. Load FPL data ─────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("LOADING FPL DATA")
    print("="*80)
    loader = FPLDataLoader()
    raw_df = loader.load_data()
    print(f"Loaded {len(raw_df)} player-gameweek records")

    # ── 2. Train model ───────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("TRAINING MODEL")
    print("="*80)
    model = LinearPredictionModel()
    model.train(raw_df, target_gw)

    # ── 3. Get predictions ───────────────────────────────────────────────────
    print("\n" + "="*80)
    print("GETTING PREDICTIONS")
    print("="*80)
    predictions = model.predict(raw_df, target_gw)
    print(f"Generated predictions for {len(predictions)} players")

    # ── 4. Enrich the saved team with live data ──────────────────────────────
    current_team_df, bank = enrich_team_from_predictions(team_json, predictions)
    print(f"\nBank: £{bank:.1f}m")
    print_team(current_team_df, label="CURRENT TEAM")

    # ── 5. Transfer suggestions ──────────────────────────────────────────────
    selector = IterativeDeepeningTeamSelector()

    if NUM_TRANSFERS == 0:
        print("\n" + "="*80)
        print("NUM_TRANSFERS = 0 — no transfers requested.")
        print("="*80)

    else:
        print("\n" + "="*80)
        print(f"SUGGESTING {NUM_TRANSFERS} TRANSFER(S)")
        print("="*80)

        result = selector.make_transfers(
            current_team      = current_team_df,
            available_players = predictions,
            num_transfers     = NUM_TRANSFERS,
            budget            = bank,
        )

        new_team         = result["team"]
        transfers        = result["transfers"]
        remaining_budget = result["budget"]

        # ── Print suggested transfers ────────────────────────────────────────
        print(f"\n{'─'*80}")
        print("SUGGESTED TRANSFERS")
        print(f"{'─'*80}")
        for i, t in enumerate(transfers, 1):
            direction = "↑ gain" if t["points_gain"] >= 0 else "↓ lose"
            print(
                f"  Transfer {i}: OUT {t['out']:20s}  →  IN {t['in']:20s} | "
                f"Cost Δ £{t['cost_change']:+.1f}m | "
                f"Pts {direction} {abs(t['points_gain']):.2f}"
            )

        total_pts_gain = sum(t["points_gain"] for t in transfers)
        print(f"\n  Total predicted points improvement : {total_pts_gain:+.2f}")
        print(f"  Remaining bank after transfers     : £{remaining_budget:.1f}m")

        print_team(new_team, label="SUGGESTED TEAM")

        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"  Transfers made  : {NUM_TRANSFERS}")
        print(f"  Points before   : {current_team_df['predicted_points'].sum():.2f}")
        print(f"  Points after    : {new_team['predicted_points'].sum():.2f}")
        print(f"  Improvement     : {new_team['predicted_points'].sum() - current_team_df['predicted_points'].sum():+.2f}")
        print(f"  Cost before     : £{current_team_df['now_cost'].sum():.1f}m")
        print(f"  Cost after      : £{new_team['now_cost'].sum():.1f}m")
        print(f"  Bank remaining  : £{remaining_budget:.1f}m")
        print(f"Latest GW in data: GW{raw_df['gameweek'].max()} — predicting for GW{raw_df['gameweek'].max() + 1}")