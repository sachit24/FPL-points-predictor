from abc import ABC, abstractmethod
import pandas as pd

class BaseModel(ABC):
    """Abstract base class for any FPL model"""
    
    @abstractmethod
    def train(self, df: pd.DataFrame, target_gw=None):
        """Train the model on historical data"""
        pass
    
    @abstractmethod
    def predict(self, df: pd.DataFrame, target_gw=None) -> pd.DataFrame:
        """
        Return predictions for next gameweek
        
        Returns:
            pd.DataFrame with columns: id, web_name, element_type, predicted_points
        """
        pass