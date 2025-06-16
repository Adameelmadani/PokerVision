import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimpleNLP:
    def __init__(self):
        # Add new dynamic gameplay queries
        self.gameplay_queries = {
            "should i fold": self._get_fold_advice,
            "should i call": self._get_call_advice,
            "should i raise": self._get_raise_advice,
            "what should i do": self._get_general_advice,
            "is my hand good": self._get_hand_strength_advice
        }

        # Static knowledge base for general queries
        self.knowledge_base = {
            "what are the poker hand rankings": 
                "Poker hand rankings from highest to lowest: Royal Flush, Straight Flush, Four of a Kind, " 
                "Full House, Flush, Straight, Three of a Kind, Two Pair, One Pair, High Card.",
            
            "what is a good starting hand": 
                "Premium starting hands include AA, KK, QQ, JJ, AK suited. Strong hands include TT, AQ, AJ suited, " 
                "KQ suited. Position and game context also matter a lot.",
            
            "how do I calculate pot odds": 
                "Pot odds are calculated by dividing the current pot size by the cost of your call. " 
                "If your odds of winning are better than the pot odds, the call is mathematically profitable.",
            
            "what is position in poker": 
                "Position refers to where you sit relative to the dealer button. Late positions (dealer, cutoff) " 
                "are advantageous since you act after most players and have more information.",
            
            "when should I fold": 
                "You should fold when the expected value of continuing is negative, which happens when your " 
                "hand strength and pot odds don't justify continuing. Don't get attached to mediocre hands.",
            
            "what is a cooler in poker": 
                "A cooler is when two very strong hands clash, and there's almost no way to avoid losing a lot " 
                "of chips. Examples include AA vs KK or set over set situations.",
            
            "what is card recognition": 
                "Card recognition in PokerVision uses computer vision and neural networks to automatically detect " 
                "and identify cards from a screen capture of your poker table.",
            
            "how accurate is the model": 
                "The model's accuracy depends on the quality and quantity of training data. Run test_models.py " 
                "to see specific accuracy metrics for card rank, suit and empty position recognition."
        }
        
        # Initialize vectorizer and vectors
        self.vectorizer = TfidfVectorizer()
        self.vectors = self.vectorizer.fit_transform(list(self.knowledge_base.keys()) + list(self.gameplay_queries.keys()))
        
        self.fallback_responses = [
            "I don't have information about that. Try asking about poker hands, odds, or card recognition.",
            "I'm not sure how to answer that. You can ask about poker strategy or the PokerVision system.",
            "That's beyond my knowledge. I can help with basic poker concepts or the card recognition system.",
            "Try asking about hand rankings, starting hands, pot odds, or the card recognition features."
        ]
        
        # Current game state (to be updated externally)
        self.current_game_state = {
            'hand_analysis': None,
            'strategy_advice': None,
            'game_state': None
        }

    def update_game_state(self, hand_analysis, strategy_advice, game_state):
        """Update the current game state for contextual responses"""
        self.current_game_state = {
            'hand_analysis': hand_analysis,
            'strategy_advice': strategy_advice,
            'game_state': game_state
        }

    def get_response(self, query):
        """Enhanced response generation with gameplay advice"""
        query = query.lower().strip()
        
        if len(query) < 3:
            return "Please ask a question about poker or the current hand."
            
        # Check for gameplay advice queries first
        for key, handler in self.gameplay_queries.items():
            if key in query:
                return handler()
                
        # Fall back to static knowledge base
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        best_match_index = np.argmax(similarities)
        best_match_score = similarities[best_match_index]
        
        if best_match_score > 0.3:
            best_question = list(self.knowledge_base.keys())[best_match_index]
            return self.knowledge_base[best_question]
        
        return random.choice(self.fallback_responses)

    def get_available_prompts(self):
        """Return all available prompts for the dropdown"""
        # Combine static and dynamic prompts
        static_prompts = list(self.knowledge_base.keys())
        gameplay_prompts = [
            "Should I fold?", 
            "Should I call?", 
            "Should I raise?", 
            "What should I do now?",
            "Is my hand good?"
        ]
        return static_prompts + gameplay_prompts

    def _get_fold_advice(self):
        """Generate fold advice based on current game state"""
        if not self._has_valid_game_state():
            return "I need to see your cards to give fold advice."
            
        hand_analysis = self.current_game_state['hand_analysis']
        strategy = self.current_game_state['strategy_advice']
        
        win_prob = hand_analysis['win_probability']
        if win_prob < 30:
            return f"Yes, folding is recommended. You have only {win_prob:.1f}% chance to win."
        elif win_prob < 45:
            return f"It's marginal ({win_prob:.1f}% win chance). {strategy['action']} - {strategy['reasoning']}"
        else:
            return f"No need to fold. You have a {win_prob:.1f}% chance to win. {strategy['action']}"

    def _get_call_advice(self):
        """Generate call advice based on current game state"""
        if not self._has_valid_game_state():
            return "I need to see your cards to give calling advice."
            
        strategy = self.current_game_state['strategy_advice']
        if 'Call' in strategy['action']:
            return f"Yes, calling is reasonable. {strategy['reasoning']}"
        return f"Better to {strategy['action']}. {strategy['reasoning']}"

    def _get_raise_advice(self):
        """Generate raise advice based on current game state"""
        if not self._has_valid_game_state():
            return "I need to see your cards to give raising advice."
            
        hand_analysis = self.current_game_state['hand_analysis']
        strategy = self.current_game_state['strategy_advice']
        
        if hand_analysis['win_probability'] > 70:
            return f"Yes, raising is strong here with {hand_analysis['hand_name']}. {strategy['reasoning']}"
        return f"{strategy['action']} might be better. {strategy['reasoning']}"

    def _get_general_advice(self):
        """Generate general advice based on current game state"""
        if not self._has_valid_game_state():
            return "I need to see your cards to give advice."
            
        strategy = self.current_game_state['strategy_advice']
        return f"Recommended action: {strategy['action']} - {strategy['reasoning']}"

    def _get_hand_strength_advice(self):
        """Generate hand strength advice"""
        if not self._has_valid_game_state():
            return "I need to see your cards to evaluate hand strength."
            
        hand_analysis = self.current_game_state['hand_analysis']
        win_prob = hand_analysis['win_probability']
        
        if win_prob > 80:
            strength = "very strong"
        elif win_prob > 60:
            strength = "strong"
        elif win_prob > 40:
            strength = "medium"
        else:
            strength = "weak"
            
        return f"Your hand is {strength} ({hand_analysis['hand_name']}) with {win_prob:.1f}% win probability."

    def _has_valid_game_state(self):
        """Check if we have valid game state data"""
        return (self.current_game_state['hand_analysis'] is not None and 
                self.current_game_state['strategy_advice'] is not None)