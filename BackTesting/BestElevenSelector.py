"""
BestElevenSelector.py
---------------------
Place this file in the Selection Models folder.

Picks the best starting 11 from a 15-player squad and identifies the captain.
Rules (standard FPL):
    - Exactly 1 goalkeeper
    - At least 3 defenders
    - At least 2 midfielders
    - At least 1 forward
    - Total = 11 players
"""

from itertools import combinations
import pandas as pd


class BestElevenSelector:

    def select(self, squad_df: pd.DataFrame) -> tuple:
        """
        From a 15-player squad DataFrame, select the best starting 11
        and the captain (highest predicted points in the 11).

        Parameters
        ----------
        squad_df : pd.DataFrame
            Must contain columns: id, element_type, predicted_points

        Returns
        -------
        (eleven_df, captain) : (pd.DataFrame, pd.Series)
        """
        best_score  = -float("inf")
        best_eleven = None

        squad_reset = squad_df.reset_index(drop=True)
        indices     = list(range(len(squad_reset)))

        for combo in combinations(indices, 11):
            eleven = squad_reset.iloc[list(combo)]

            gk_count  = (eleven["element_type"] == 1).sum()
            def_count = (eleven["element_type"] == 2).sum()
            mid_count = (eleven["element_type"] == 3).sum()
            fwd_count = (eleven["element_type"] == 4).sum()

            if gk_count == 1 and def_count >= 3 and mid_count >= 2 and fwd_count >= 1:
                score = eleven["predicted_points"].sum()
                if score > best_score:
                    best_score  = score
                    best_eleven = eleven.copy()

        if best_eleven is None:
            raise ValueError("Could not form a valid starting 11 from the squad.")

        captain = best_eleven.loc[best_eleven["predicted_points"].idxmax()]
        return best_eleven, captain