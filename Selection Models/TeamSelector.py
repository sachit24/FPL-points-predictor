from abc import ABC, abstractmethod
import pandas as pd

class TeamSelector(ABC):
    """Abstract base class for team selection strategies"""
    
    @abstractmethod
    def make_transfers(self, current_team: pd.DataFrame, available_players: pd.DataFrame, 
                      num_transfers: int, budget: float) -> dict:
        """
        Make transfers to existing team
        
        Args:
            current_team: DataFrame with current team players
            available_players: DataFrame with all available players (predictions)
            num_transfers: Number of transfers to make
            budget: Available budget
            
        Returns:
            dict with keys: 'team' (new team DataFrame), 'transfers' (list of swaps), 'budget' (remaining)
        """
        pass
    
    @abstractmethod
    def rebuild_team(self, available_players: pd.DataFrame, budget: float) -> dict:
        """
        Build a completely new team from scratch
        
        Args:
            available_players: DataFrame with all available players (predictions)
            budget: Total budget
            
        Returns:
            dict with keys: 'team' (team DataFrame), 'budget' (remaining)
        """
        pass