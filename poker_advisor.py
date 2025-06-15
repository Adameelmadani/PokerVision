import random
from poker_evaluator import get_hand_analysis, CARD_VALUES

class PokerAdvisor:
    def __init__(self):
        # Pre-flop hand rankings (pairs)
        self.pocket_pair_rankings = {
            'A': 1, 'K': 2, 'Q': 3, 'J': 4, '10': 5, '9': 6, '8': 7,
            '7': 8, '6': 9, '5': 10, '4': 11, '3': 12, '2': 13
        }
        
        # Pre-flop hand rankings (suited)
        self.suited_rankings = {
            'AK': 1, 'AQ': 2, 'AJ': 3, 'KQ': 4, 'A10': 5, 'KJ': 6,
            'QJ': 7, 'K10': 8, 'Q10': 9, 'J10': 10
        }

    def get_preflop_hand_strength(self, player_cards):
        """Calculate pre-flop hand strength"""
        if not player_cards or len(player_cards) != 2:
            return 0
        
        card1, card2 = player_cards
        if card1.get('empty', True) or card2.get('empty', True):
            return 0

        rank1, rank2 = card1['rank'], card2['rank']
        suit1, suit2 = card1['suit'], card2['suit']
        
        # Check for pocket pairs
        if rank1 == rank2:
            return 1.0 - (self.pocket_pair_rankings.get(rank1, 13) / 13)
        
        # Sort ranks for consistent checking
        ranks = sorted([rank1, rank2], key=lambda x: CARD_VALUES.get(x, 0), reverse=True)
        hand_code = ranks[0] + ranks[1]
        
        # Check suited hands
        if suit1 == suit2:
            suited_rank = self.suited_rankings.get(hand_code, 15)
            return 0.8 - (suited_rank / 20)
            
        # Offsuit hands are generally weaker
        suited_rank = self.suited_rankings.get(hand_code, 20)
        return 0.6 - (suited_rank / 25)

    def get_position_strength(self, position):
        """Calculate position strength (0-1)"""
        # Position strength increases from early to late position
        position_values = {
            'early': 0.3,    # First to act
            'middle': 0.5,   # Middle position
            'late': 0.8,     # Button or cut-off
            'blind': 0.2     # Blinds
        }
        return position_values.get(position, 0.5)

    def analyze_situation(self, player_cards, table_cards, position='middle', pot_size=0, stack_size=100):
        """
        Analyze the current poker situation and provide strategic advice.
        
        Args:
            player_cards (list): Player's hole cards
            table_cards (list): Community cards
            position (str): Player's position ('early', 'middle', 'late', 'blind')
            pot_size (float): Current pot size in big blinds
            stack_size (float): Player's stack size in big blinds
            
        Returns:
            dict: Analysis and recommendations
        """
        # Get basic hand analysis
        hand_analysis = get_hand_analysis(player_cards, table_cards)
        
        # Calculate pre-flop strength
        preflop_strength = self.get_preflop_hand_strength(player_cards)
        position_strength = self.get_position_strength(position)
        
        # Determine game state
        num_table_cards = len([c for c in table_cards if not c.get('empty', False)])
        
        if num_table_cards == 0:
            # Pre-flop advice
            if preflop_strength > 0.8:
                action = "Raise 3-4x BB"
                reasoning = "Very strong starting hand"
            elif preflop_strength > 0.6:
                action = "Raise 2.5-3x BB" if position_strength > 0.5 else "Call"
                reasoning = "Strong hand, good position" if position_strength > 0.5 else "Strong hand but poor position"
            elif preflop_strength > 0.4:
                action = "Call" if position_strength > 0.5 else "Fold"
                reasoning = "Speculative hand, use position" if position_strength > 0.5 else "Marginal hand in early position"
            else:
                action = "Fold"
                reasoning = "Weak starting hand"
                
        else:
            # Post-flop advice based on hand strength and win probability
            win_prob = hand_analysis['win_probability']
            
            if win_prob > 80:
                action = "Bet for value, 1/2 to 2/3 pot"
                reasoning = f"Very strong hand ({hand_analysis['hand_name']})"
            elif win_prob > 60:
                action = "Bet for value, 1/3 to 1/2 pot"
                reasoning = f"Strong hand ({hand_analysis['hand_name']})"
            elif win_prob > 40:
                action = "Check/Call" if position_strength < 0.6 else "Small bet, 1/4 pot"
                reasoning = f"Medium strength hand ({hand_analysis['hand_name']})"
            elif win_prob > 20:
                action = "Check/Fold"
                reasoning = f"Weak hand ({hand_analysis['hand_name']})"
            else:
                action = "Fold"
                reasoning = f"Very weak hand ({hand_analysis['hand_name']})"
        
        return {
            'action': action,
            'reasoning': reasoning,
            'hand_strength': preflop_strength if num_table_cards == 0 else win_prob/100,
            'position_strength': position_strength,
            'hand_name': hand_analysis['hand_name'],
            'win_probability': hand_analysis['win_probability']
        }
