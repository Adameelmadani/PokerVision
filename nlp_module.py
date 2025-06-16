import re
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Download necessary NLTK data (uncomment first time)
# nltk.download('punkt')
# nltk.download('stopwords')

class SimpleNLP:
    def __init__(self):
        # Initialize NLP components
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            # Fallback if NLTK data isn't downloaded
            self.stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])
        
        # Add new dynamic gameplay queries with their triggers
        self.gameplay_query_triggers = {
            "fold": self._get_fold_advice,
            "call": self._get_call_advice,
            "raise": self._get_raise_advice,
            "check": self._get_call_advice,
            "do now": self._get_general_advice,
            "should i do": self._get_general_advice,
            "my hand": self._get_hand_strength_advice,
            "win chance": self._get_hand_strength_advice,
            "odds": self._get_hand_strength_advice
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
        
        # Initialize vectorizer with simpler parameters to avoid validation errors
        self.vectorizer = TfidfVectorizer(
            stop_words='english',  # Use built-in English stopwords instead of custom tokenizer
            ngram_range=(1, 2)     # Keep n-grams
        )
        
        # Preprocess knowledge base topics for vectorization
        self.knowledge_topics = list(self.knowledge_base.keys())
        
        # Fit vectorizer to knowledge base topics
        try:
            self.vectors = self.vectorizer.fit_transform(self.knowledge_topics)
        except Exception as e:
            print(f"Warning: TF-IDF vectorization failed: {str(e)}")
            # Fallback to simple string matching
            self.vectorizer = None
        
        # Fallback responses
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

    def _tokenize_and_stem(self, text):
        """Tokenize and stem the text"""
        # Clean text
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and stem
        return [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]
    
    def _preprocess_query(self, query):
        """Clean and preprocess the user query"""
        # Convert to lowercase
        query = query.lower().strip()
        
        # Remove punctuation
        query = query.translate(str.maketrans('', '', string.punctuation))
        
        return query

    def update_game_state(self, hand_analysis, strategy_advice, game_state):
        """Update the current game state for contextual responses"""
        self.current_game_state = {
            'hand_analysis': hand_analysis,
            'strategy_advice': strategy_advice,
            'game_state': game_state
        }

    def get_response(self, query):
        """Enhanced response generation with gameplay advice"""
        # Clean and preprocess the query
        query = self._preprocess_query(query)
        
        if len(query) < 3:
            return "Please ask a question about poker or your current hand."
        
        # Check for gameplay advice triggers
        for trigger, handler in self.gameplay_query_triggers.items():
            if trigger in query:
                return handler()
        
        # Use vectorization if available, otherwise fall back to string matching
        if self.vectorizer:
            try:
                query_vector = self.vectorizer.transform([query])
                similarities = cosine_similarity(query_vector, self.vectors).flatten()
                best_match_index = np.argmax(similarities)
                best_match_score = similarities[best_match_index]
                
                # If similarity is above threshold, return the corresponding answer
                if best_match_score > 0.3:
                    best_topic = self.knowledge_topics[best_match_index]
                    return self.knowledge_base[best_topic]
            except Exception as e:
                print(f"Warning: TF-IDF matching failed: {str(e)}")
                # Continue to fallback methods below
        else:
            # Simple string matching as fallback
            for topic, answer in self.knowledge_base.items():
                if query in topic or topic in query:
                    return answer
        
        # If we have game state but no direct query match, try to give helpful advice
        if self._has_valid_game_state():
            if any(word in query for word in ['good', 'bad', 'strong', 'weak', 'win']):
                return self._get_hand_strength_advice()
            if any(word in query for word in ['do', 'play', 'action', 'move']):
                return self._get_general_advice()
                
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