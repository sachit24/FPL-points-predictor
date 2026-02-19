import pandas as pd
import requests

class FPLDataLoader:
    
    def get_full_data(self):
        url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        response = requests.get(url)
        data = response.json()
        return data

    def get_gameweek_data(self, full_data: dict):
        events_df = pd.DataFrame(full_data['events'])
        finished_gws = events_df[events_df['finished']==True]
        all_gameweeks_data = []
        
        for gw_id in finished_gws["id"]:
            gw_url = f"https://fantasy.premierleague.com/api/event/{gw_id}/live/"
            gw_response = requests.get(gw_url)
            gw_data = gw_response.json()
            all_gameweeks_data.append(gw_data["elements"])
        
        return all_gameweeks_data

    def make_gw_df(self, gameweeks_data, full_data):
        all_players_data = []
        for gw_idx, gw in enumerate(gameweeks_data):
            for player in gw:
                player_id = player["id"]
                player_stats = player["stats"]
                player_stats['id'] = player_id
                player_stats['fixture'] = player["explain"][0]["fixture"]
                player_stats['gameweek'] = gw_idx + 1  # ADD THIS LINE
                all_players_data.append(player_stats)
        
        df = pd.DataFrame(all_players_data)
        new_column_order = ["id"] + [col for col in df.columns if col != "id"]
        df = df[new_column_order]
        players_info = pd.DataFrame(full_data["elements"])
        df = df.merge(players_info[['id', 'element_type', 'web_name', 'status', 'now_cost', 'team']], 
                      on='id', how='left')
        
        # Convert now_cost from tenths to actual millions (60 -> 6.0)
        df['now_cost'] = df['now_cost'] / 10.0
        
        object_cols = ['influence', 'creativity', 'threat', 'ict_index', 
                       'expected_goals', 'expected_assists', 
                       'expected_goal_involvements', 'expected_goals_conceded']
        
        for col in object_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.sort_values(['id', 'fixture'])
            
        return df
    
    def load_data(self):
        """Load raw FPL data"""
        data = self.get_full_data()
        gameweeks_data = self.get_gameweek_data(data)
        df = self.make_gw_df(gameweeks_data, data)
        return df