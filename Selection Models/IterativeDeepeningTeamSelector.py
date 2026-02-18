import pandas as pd
import numpy as np
from itertools import combinations, product
from TeamSelector import TeamSelector

class IterativeDeepeningTeamSelector(TeamSelector):
    
    def __init__(self, max_offset=5, max_players_per_club=3, price_penalty_weight=0.3):
        self.max_offset = max_offset
        self.max_players_per_club = max_players_per_club
        self.price_penalty_weight = price_penalty_weight
        
        # Different max_k per position: {position: max_k}
        self.position_max_k = {
            1: 6,   # Goalkeepers (2 in team)
            2: 12,  # Defenders (5 in team)
            3: 15,  # Midfielders (5 in team)
            4: 10   # Forwards (3 in team)
        }
        self.initial_k = 4  # Start with 4 replacements per player
    
    def calculate_performance_percentiles(self, all_players: pd.DataFrame, team_player_ids: set) -> dict:
        """Calculate performance percentile for each player in their position"""
        percentiles = {}
        
        for position in [1, 2, 3, 4]:
            players_in_position = all_players[all_players['element_type'] == position].copy()
            
            # Include if: has non-zero points OR is in the team
            eligible = players_in_position[
                (players_in_position['predicted_points'] > 0.0) | 
                (players_in_position['id'].isin(team_player_ids))
            ].copy()
            
            # Sort by predicted points (ascending: worst to best)
            eligible = eligible.sort_values('predicted_points')
            
            # Assign percentiles ONLY for players in the current team
            for i, row in enumerate(eligible.itertuples()):
                if row.id in team_player_ids:
                    percentile = (i / (len(eligible) - 1)) * 100.0 if len(eligible) > 1 else 50.0
                    percentiles[row.id] = percentile
        
        return percentiles
    
    def calculate_price_percentiles(self, all_players: pd.DataFrame, team_player_ids: set) -> dict:
        """Calculate price percentile for each player in their position"""
        percentiles = {}
        
        for position in [1, 2, 3, 4]:
            players_in_position = all_players[all_players['element_type'] == position].copy()
            
            # Sort by price (ascending: cheapest to most expensive)
            players_in_position = players_in_position.sort_values('now_cost')
            
            # Assign percentiles ONLY for players in the current team
            for i, row in enumerate(players_in_position.itertuples()):
                if row.id in team_player_ids:
                    percentile = (i / (len(players_in_position) - 1)) * 100.0 if len(players_in_position) > 1 else 50.0
                    percentiles[row.id] = percentile
        
        return percentiles
    
    def find_worst_players(self, current_team: pd.DataFrame, num_transfers: int, 
                          perf_percentiles: dict, price_percentiles: dict) -> pd.DataFrame:
        """Find worst N players using combined score"""
        scores = []
        
        for _, player in current_team.iterrows():
            perf_pct = perf_percentiles.get(player['id'], 50.0)
            price_pct = price_percentiles.get(player['id'], 50.0)
            
            # Lower is worse: low performance + high price = bad value
            combined_score = perf_pct - (self.price_penalty_weight * price_pct)
            scores.append(combined_score)
        
        current_team = current_team.copy()
        current_team['combined_score'] = scores
        worst = current_team.nsmallest(num_transfers, 'combined_score')
        
        return worst.drop(columns=['combined_score'])
    
    def find_best_replacements(self, players_to_replace: pd.DataFrame, 
                              current_team: pd.DataFrame, 
                              available_players: pd.DataFrame, 
                              budget: float) -> list:
        """Find best replacement combinations using iterative deepening"""
        
        # Group players to replace by position
        players_by_position = {}
        for position in [1, 2, 3, 4]:
            pos_players = players_to_replace[players_to_replace['element_type'] == position]
            if len(pos_players) > 0:
                players_by_position[position] = pos_players
        
        # Iterative deepening: start with initial_k, increase if no solution found
        for offset in range(self.max_offset + 1):
            current_k = self.initial_k + offset
            
            choices_per_position = []
            
            for position, to_replace in players_by_position.items():
                num_to_replace = len(to_replace)
                max_candidates = min(num_to_replace * current_k, self.position_max_k[position])
                
                # Players we want to KEEP (not being replaced)
                kept_players = current_team[~current_team['id'].isin(players_to_replace['id'])]
                
                # Get top candidates - exclude only the players we're keeping
                candidates = available_players[
                    (available_players['element_type'] == position) &
                    (~available_players['id'].isin(kept_players['id']))
                ].nlargest(max_candidates, 'predicted_points')
                
                if len(candidates) < num_to_replace:
                    # Not enough candidates, skip this iteration
                    choices_per_position = None
                    break
                
                # Generate combinations for this position
                position_combos = list(combinations(range(len(candidates)), num_to_replace))
                position_combos = [
                    [candidates.iloc[i] for i in combo] 
                    for combo in position_combos
                ]
                
                choices_per_position.append(position_combos)
            
            if choices_per_position is None:
                continue  # Try next offset
            
            # Generate cartesian product across positions
            all_combinations = list(product(*choices_per_position))
            
            # Flatten combinations
            all_combinations = [
                [player for choice in combo for player in choice]
                for combo in all_combinations
            ]
            
            # Score each combination
            best_swaps = None
            best_score = -float('inf')
            
            for combination in all_combinations:
                swaps = self.map_combination_to_swaps(combination, players_to_replace)
                
                # Check constraints
                if not self.fits_within_budget(swaps, budget):
                    continue
                
                if self.has_club_violation(swaps, current_team):
                    continue
                
                # Calculate score
                score = sum(player['predicted_points'] for player in combination)
                
                if score > best_score:
                    best_score = score
                    best_swaps = swaps
            
            # If we found valid swaps, return immediately
            if best_swaps:
                return best_swaps
        
        # No valid combination found after all offsets
        return []
    
    def map_combination_to_swaps(self, combination: list, players_to_replace: pd.DataFrame) -> list:
        """Map incoming players to outgoing players by position"""
        swaps = []
        
        # Group by position
        outgoing_by_pos = {}
        for _, player in players_to_replace.iterrows():
            pos = player['element_type']
            if pos not in outgoing_by_pos:
                outgoing_by_pos[pos] = []
            outgoing_by_pos[pos].append(player)
        
        incoming_by_pos = {}
        for player in combination:
            pos = player['element_type']
            if pos not in incoming_by_pos:
                incoming_by_pos[pos] = []
            incoming_by_pos[pos].append(player)
        
        # Pair them up
        for pos in outgoing_by_pos:
            outgoing = outgoing_by_pos[pos]
            incoming = incoming_by_pos[pos]
            
            for i in range(len(outgoing)):
                swaps.append({
                    'out': outgoing[i],
                    'in': incoming[i],
                    'points_gain': incoming[i]['predicted_points'] - outgoing[i]['predicted_points']
                })
        
        return swaps
    
    def fits_within_budget(self, swaps: list, budget: float) -> bool:
        """Check if swaps fit within budget"""
        remaining = budget
        for swap in swaps:
            remaining += swap['out']['now_cost']
            remaining -= swap['in']['now_cost']
        return remaining >= 0
    
    def has_club_violation(self, swaps: list, current_team: pd.DataFrame) -> bool:
        """Check if swaps would violate club constraint"""
        club_counts = current_team['team'].value_counts().to_dict()
        
        # Remove players being transferred out
        for swap in swaps:
            club = swap['out']['team']
            club_counts[club] = club_counts.get(club, 0) - 1
            if club_counts[club] == 0:
                del club_counts[club]
        
        # Add players being transferred in
        for swap in swaps:
            club = swap['in']['team']
            club_counts[club] = club_counts.get(club, 0) + 1
        
        # Check if any club exceeds limit
        return any(count > self.max_players_per_club for count in club_counts.values())
    
    def make_transfers(self, current_team: pd.DataFrame, available_players: pd.DataFrame, 
                      num_transfers: int, budget: float) -> dict:
        """Main transfer logic - exact Java conversion"""
        
        if num_transfers == 0:
            return {'team': current_team.copy(), 'transfers': [], 'budget': budget}
        
        team_player_ids = set(current_team['id'])
        
        # Calculate percentiles
        perf_percentiles = self.calculate_performance_percentiles(available_players, team_player_ids)
        print(available_players.head()) # Debug: Check available players data
        price_percentiles = self.calculate_price_percentiles(available_players, team_player_ids)
        
        # Find worst players
        players_to_replace = self.find_worst_players(current_team, num_transfers, 
                                                     perf_percentiles, price_percentiles)
        
        # Find best replacements
        best_swaps = self.find_best_replacements(players_to_replace, current_team, 
                                                 available_players, budget)
        
        if not best_swaps:
            raise ValueError("No valid transfers found within budget and club constraints")
        
        # Apply swaps
        new_team = current_team[~current_team['id'].isin(players_to_replace['id'])].copy()
        new_budget = budget
        
        transfers = []
        for swap in best_swaps:
            new_team = pd.concat([new_team, pd.DataFrame([swap['in']])], ignore_index=True)
            new_budget += swap['out']['now_cost']
            new_budget -= swap['in']['now_cost']
            
            transfers.append({
                'out': swap['out']['web_name'],
                'in': swap['in']['web_name'],
                'position': swap['in']['element_type'],
                'cost_change': swap['in']['now_cost'] - swap['out']['now_cost'],
                'points_gain': swap['points_gain']
            })
        
        return {'team': new_team, 'transfers': transfers, 'budget': new_budget}
    
    def rebuild_team(self, available_players: pd.DataFrame, budget: float) -> dict:
        """Build team from scratch - call make_transfers with 15 transfers from dummy team"""
        formation = {1: 2, 2: 5, 3: 5, 4: 3}
        dummy_team = []
        
        for position, count in formation.items():
            worst = available_players[available_players['element_type'] == position].nsmallest(count, 'predicted_points')
            dummy_team.append(worst)
        
        dummy_team = pd.concat(dummy_team, ignore_index=True)
        
        # Make 15 transfers (rebuild entire team)
        return self.make_transfers(dummy_team, available_players, 15, budget)
    
