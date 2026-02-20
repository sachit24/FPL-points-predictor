import sys
import requests
import pandas as pd

sys.path.append("../Selection Models")   # for BestElevenSelector

from base_model    import BaseModel
from TeamSelector  import TeamSelector
from BestElevenSelector import BestElevenSelector


class Backtester:

    def __init__(
        self,
        model:     BaseModel,
        selector:  TeamSelector,
        start_gw:  int   = 10,
        budget:    float = 100.0,
    ):
        self.model           = model
        self.selector        = selector
        self.start_gw        = start_gw
        self.budget          = budget
        self.eleven_selector = BestElevenSelector()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_avg_entry_scores(self) -> pd.DataFrame:
        """Pull average_entry_score per finished GW from the bootstrap API."""
        url  = "https://fantasy.premierleague.com/api/bootstrap-static/"
        data = requests.get(url).json()

        events_df = pd.DataFrame(data["events"])
        finished  = events_df[events_df["finished"] == True][["id", "average_entry_score"]].copy()
        finished.rename(columns={"id": "gameweek", "average_entry_score": "avg_score"}, inplace=True)
        return finished.reset_index(drop=True)

    def _actual_gw_points(
        self,
        eleven_df:  pd.DataFrame,
        captain_id: int,
        raw_df:     pd.DataFrame,
        gw:         int,
    ) -> int:
        """
        Sum actual total_points for the starting 11 in the given GW.
        Captain's points are doubled.
        Handles double-GW fixtures by summing across all fixtures in the week.
        """
        gw_data  = raw_df[raw_df["gameweek"] == gw]
        gw_pts   = gw_data.groupby("id")["total_points"].sum()

        total = 0
        for pid in eleven_df["id"]:
            pts = int(gw_pts.get(pid, 0))
            if pid == captain_id:
                pts *= 2
            total += pts
        return total

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full backtest.

        Parameters
        ----------
        raw_df : pd.DataFrame
            The full dataset returned by FPLDataLoader.load_data().
            Must contain columns: id, gameweek, total_points, element_type,
            web_name, now_cost, team, predicted_points (after model runs).

        Returns
        -------
        pd.DataFrame  with columns: gameweek, avg_score, our_score, vs_avg
        """
        last_gw = int(raw_df["gameweek"].max())

        print(f"Backtesting GW{self.start_gw} → GW{last_gw}")
        # print("Fetching average entry scores from FPL API...")
        avg_df = self._fetch_avg_entry_scores()

        # Restrict to the GWs we are testing
        results = avg_df[
            (avg_df["gameweek"] >= self.start_gw) &
            (avg_df["gameweek"] <= last_gw)
        ].copy().reset_index(drop=True)

        results["our_score"] = 0

        for gw in range(self.start_gw, last_gw + 1):
            # print(f"\n{'='*60}")
            print(f"GW{gw}  |  training on GW1–{gw - 1}, predicting GW{gw}")
            # print(f"{'='*60}")

            # 1. Train on data strictly before this GW
            self.model.train(raw_df, target_gw=gw)

            # 2. Predict for this GW
            predictions = self.model.predict(raw_df, target_gw=gw)
            # print(f"  Predictions generated for {len(predictions)} players")
            # Zero out predicted points for players who didn't play in this GW
            # Note: this uses future information — known limitation for now
            actual_gw_minutes = raw_df[raw_df["gameweek"] == gw].groupby("id")["minutes"].sum()
            predictions["predicted_points"] = predictions.apply(
                lambda row: 0.0 if actual_gw_minutes.get(row["id"], 1) == 0 else row["predicted_points"],
                axis=1
            )

            # 3. Build best 15-player squad from scratch
            result = self.selector.rebuild_team(predictions, self.budget)
            squad  = result["team"]

            # 4. Pick best starting 11 + captain
            eleven, captain = self.eleven_selector.select(squad)
            captain_id      = int(captain["id"])

            gw_pts = raw_df[raw_df["gameweek"] == gw].groupby("id")["total_points"].sum()

            positions = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
            for _, p in eleven.sort_values("element_type").iterrows():
                flag = " ©" if p["id"] == captain_id else ""
                actual = int(gw_pts.get(p["id"], 0))
                print(f"  {positions[p['element_type']]}  {p['web_name']:20s} pred: {p['predicted_points']:.2f}  actual: {actual}{flag}")

            # print(f"  Captain : {captain['web_name']} "
            #       f"(predicted {captain['predicted_points']:.2f} pts)")

            # 5. Actual points for this GW
            our_score = self._actual_gw_points(eleven, captain_id, raw_df, gw)
            avg_score = int(results.loc[results["gameweek"] == gw, "avg_score"].values[0])

            print(f"  Our score : {our_score}")
            print(f"  Avg score : {avg_score}")
            print(f"  vs avg    : {our_score - avg_score:+d}")

            results.loc[results["gameweek"] == gw, "our_score"] = our_score

        # Final column
        results["vs_avg"] = results["our_score"] - results["avg_score"]

        print(f"\n{'='*60}")
        print("BACKTEST COMPLETE")
        print(f"  GWs tested       : {self.start_gw}–{last_gw}")
        print(f"  Total our score  : {results['our_score'].sum()}")
        print(f"  Total avg score  : {results['avg_score'].sum()}")
        print(f"  Total vs avg     : {results['vs_avg'].sum():+d}")
        print(f"{'='*60}")

        return results