import random
import itertools
from collections import Counter

# Define card values for comparison
CARD_VALUES = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
    'J': 11, 'Q': 12, 'K': 13, 'A': 14
}

# Define hand rankings
HAND_RANKINGS = {
    9: "Royal Flush",
    8: "Straight Flush",
    7: "Four of a Kind",
    6: "Full House",
    5: "Flush",
    4: "Straight",
    3: "Three of a Kind",
    2: "Two Pair",
    1: "Pair",
    0: "High Card"
}

def card_to_numeric(card):
    """Convert card dictionary to numeric value and suit"""
    if card.get('empty', False):
        return None, None
    rank = card['rank']
    suit = card['suit']
    return CARD_VALUES.get(rank, 0), suit

def evaluate_hand(cards):
    """
    Evaluate a poker hand and return the hand type and strength.
    
    Args:
        cards (list): List of card dictionaries
        
    Returns:
        tuple: (hand_rank, hand_name, best_five_cards)
    """
    # Filter out empty cards
    valid_cards = [c for c in cards if not c.get('empty', False)]
    if len(valid_cards) < 2:
        return 0, "Incomplete Hand", []
    
    # Convert to numeric values
    numeric_cards = [(CARD_VALUES.get(c['rank'], 0), c['suit']) for c in valid_cards]
    
    # Sort by value (descending)
    sorted_cards = sorted(numeric_cards, key=lambda x: x[0], reverse=True)
    
    # Check for each hand type
    values = [v for v, _ in sorted_cards]
    suits = [s for _, s in sorted_cards]
    
    # Check for flush
    flush = False
    if len(suits) >= 5:
        suit_counts = Counter(suits)
        most_common_suit = suit_counts.most_common(1)[0]
        if most_common_suit[1] >= 5:
            flush = True
            flush_suit = most_common_suit[0]
            flush_cards = [c for c in sorted_cards if c[1] == flush_suit][:5]
    
    # Check for straight
    straight = False
    straight_cards = []
    if len(values) >= 5:
        # Handle Ace as 1 for A-5-4-3-2 straight
        if 14 in values:
            values_with_low_ace = sorted(set(values + [1]))
        else:
            values_with_low_ace = sorted(set(values))
            
        for i in range(len(values_with_low_ace) - 4):
            if values_with_low_ace[i:i+5] == list(range(values_with_low_ace[i], values_with_low_ace[i] + 5)):
                straight = True
                if 14 in values and values_with_low_ace[i] == 1:  # A-5-4-3-2 straight
                    straight_high = 5
                    # Find the cards for this straight
                    straight_values = set([1, 2, 3, 4, 5])
                    straight_cards = [c for c in sorted_cards if c[0] in straight_values or (c[0] == 14 and 1 in straight_values)][:5]
                else:
                    straight_high = values_with_low_ace[i+4]
                    straight_values = set(range(straight_high-4, straight_high+1))
                    straight_cards = [c for c in sorted_cards if c[0] in straight_values][:5]
                break
    
    # Count occurrences of each value
    value_counts = Counter(values)
    
    # Check hand types from highest to lowest
    
    # Royal Flush: A-K-Q-J-10 of same suit
    if flush and straight and sorted_cards[0][0] == 14 and sorted_cards[4][0] == 10 and len(set(s for _, s in sorted_cards[:5])) == 1:
        return 9, "Royal Flush", sorted_cards[:5]
    
    # Straight Flush: Five consecutive cards of the same suit
    if flush and straight:
        # Find cards that belong to both straight and flush
        straight_flush_cards = [c for c in straight_cards if c[1] == flush_suit]
        if len(straight_flush_cards) >= 5:
            return 8, "Straight Flush", straight_flush_cards[:5]
    
    # Four of a Kind: Four cards of the same value
    four_kind = [v for v, count in value_counts.items() if count == 4]
    if four_kind:
        quads = four_kind[0]
        kicker = next((v for v in values if v != quads), None)
        four_kind_cards = [c for c in sorted_cards if c[0] == quads]
        kicker_card = next((c for c in sorted_cards if c[0] == kicker), None)
        return 7, "Four of a Kind", four_kind_cards + ([kicker_card] if kicker_card else [])
    
    # Full House: Three cards of one value and two of another
    three_kind = [v for v, count in value_counts.items() if count >= 3]
    pairs = [v for v, count in value_counts.items() if count >= 2 and v not in three_kind]
    
    if three_kind and (pairs or len([v for v, count in value_counts.items() if count >= 3]) > 1):
        # If we have multiple three of a kinds, use the highest as trips and second highest for the pair
        if len(three_kind) > 1:
            trips = max(three_kind)
            pair = sorted(three_kind, reverse=True)[1]
        else:
            trips = three_kind[0]
            pair = max(pairs) if pairs else None
            
        if pair:
            trips_cards = [c for c in sorted_cards if c[0] == trips][:3]
            pair_cards = [c for c in sorted_cards if c[0] == pair][:2]
            return 6, "Full House", trips_cards + pair_cards
    
    # Flush: Five cards of the same suit (already computed above)
    if flush:
        return 5, "Flush", flush_cards
    
    # Straight: Five consecutive cards (already computed above)
    if straight:
        return 4, "Straight", straight_cards
    
    # Three of a Kind: Three cards of the same value
    if three_kind:
        trips = max(three_kind)
        trips_cards = [c for c in sorted_cards if c[0] == trips]
        kickers = [c for c in sorted_cards if c[0] != trips][:2]
        return 3, "Three of a Kind", trips_cards + kickers
    
    # Two Pair: Two cards of one value and two cards of another value
    if len(pairs) >= 2:
        top_pairs = sorted(pairs, reverse=True)[:2]
        pair1_cards = [c for c in sorted_cards if c[0] == top_pairs[0]][:2]
        pair2_cards = [c for c in sorted_cards if c[0] == top_pairs[1]][:2]
        kicker = [c for c in sorted_cards if c[0] not in top_pairs][:1]
        return 2, "Two Pair", pair1_cards + pair2_cards + kicker
    
    # One Pair: Two cards of the same value
    if pairs:
        pair = max(pairs)
        pair_cards = [c for c in sorted_cards if c[0] == pair][:2]
        kickers = [c for c in sorted_cards if c[0] != pair][:3]
        return 1, "Pair", pair_cards + kickers
    
    # High Card: Highest value card
    return 0, "High Card", sorted_cards[:5]

def calculate_win_probability(player_cards, table_cards, num_simulations=10000):
    """
    Calculate the probability of winning using Monte Carlo simulation.
    
    Args:
        player_cards (list): Player's hole cards
        table_cards (list): Community cards on the table
        num_simulations (int): Number of simulations to run
    
    Returns:
        float: Probability of winning as a percentage
    """
    # Filter out empty cards
    player_hand = [c for c in player_cards if not c.get('empty', False)]
    community_cards = [c for c in table_cards if not c.get('empty', False)]
    
    if len(player_hand) < 2:
        return 0.0  # Not enough player cards
    
    # Create a full deck of cards
    suits = ['Clubs', 'Diamonds', 'Hearts', 'Spades']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    
    full_deck = [{'rank': r, 'suit': s} for r in ranks for s in suits]
    
    # Remove cards that are already in play
    cards_in_play = player_hand + community_cards
    remaining_deck = [card for card in full_deck if not any(
        card['rank'] == c['rank'] and card['suit'] == c['suit'] for c in cards_in_play
    )]
    
    wins = 0
    
    # Monte Carlo simulation
    for _ in range(num_simulations):
        # Copy the remaining deck for this simulation
        sim_deck = list(remaining_deck)
        
        # Shuffle the deck
        random.shuffle(sim_deck)
        
        # Deal community cards if needed
        sim_community_cards = list(community_cards)
        while len(sim_community_cards) < 5:
            sim_community_cards.append(sim_deck.pop())
        
        # Deal opponent cards (assuming 1 opponent for simplicity)
        opponent_hand = [sim_deck.pop(), sim_deck.pop()]
        
        # Evaluate player's hand
        player_rank, _, _ = evaluate_hand(player_hand + sim_community_cards)
        
        # Evaluate opponent's hand
        opponent_rank, _, _ = evaluate_hand(opponent_hand + sim_community_cards)
        
        # Compare hands
        if player_rank > opponent_rank:
            wins += 1
    
    # Calculate probability
    return (wins / num_simulations) * 100

def get_hand_analysis(player_cards, table_cards):
    """
    Get a complete analysis of the current hand.
    
    Args:
        player_cards (list): Player's hole cards
        table_cards (list): Community cards on the table
    
    Returns:
        dict: Hand analysis including hand type and win probability
    """
    # Evaluate current hand
    hand_rank, hand_name, _ = evaluate_hand(player_cards + table_cards)
    
    # Calculate win probability
    win_probability = calculate_win_probability(player_cards, table_cards)
    
    return {
        'hand_rank': hand_rank,
        'hand_name': hand_name,
        'win_probability': win_probability
    }