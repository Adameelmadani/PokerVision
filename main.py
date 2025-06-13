import time
import threading
import pygame
import os
import numpy as np
import cv2
from screenshot import take_screenshot
from card_detector import detect_cards
from card_recognizer import recognize_cards, preprocess_card_image, predict_card_with_models, simple_card_recognition, rank_model_loaded, suit_model_loaded, empty_model_loaded, predict_empty_position
from poker_evaluator import get_hand_analysis

# Game states
GAME_STATES = {
    0: "Pre-flop",
    3: "Flop",
    4: "Turn",
    5: "River"
}

class PokerCV:
    def __init__(self):
        self.running = False
        self.player_cards = []
        self.table_cards = []
        self.game_state = "Pre-flop"
        self.hand_analysis = None
        # Not loading empty images directly, will use model predictions
        
        # Initialize pygame for the interface
        pygame.init()
        self.screen = pygame.display.set_mode((300, 500))
        pygame.display.set_caption("PokerCV - Poker Card Recognition")
        self.font = pygame.font.SysFont("Arial", 16)
        self.clock = pygame.time.Clock()
        self.bg_color = (0, 100, 0)  # Poker table green
    
    def is_position_empty(self, image, position, region_type):
        """Check if a card position is empty using the trained model"""
        # Preprocess the image for the model
        preprocessed_image = preprocess_card_image(image)
        
        # Use the model to predict if the position is empty
        if 'empty_model_loaded' in globals() and empty_model_loaded:
            return predict_empty_position(preprocessed_image)
        else:
            # Fallback to a simple brightness check
            brightness = np.mean(image)
            threshold = 100  # Adjust threshold as needed
            return brightness < threshold
        
    def start_detection(self):
        """Start the card detection thread"""
        self.running = True
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
    def process_card_regions(self, screenshot_rank, screenshot_suit, position):
        """Process a card's rank and suit screenshots to identify the card"""
        # Check if position is empty using the model
        if self.is_position_empty(screenshot_rank, position, "rank") and \
           self.is_position_empty(screenshot_suit, position, "suit"):
            return {"rank": "Empty", "suit": "Empty", "empty": True}
        
        # Preprocess images
        preprocessed_rank = preprocess_card_image(screenshot_rank)
        preprocessed_suit = preprocess_card_image(screenshot_suit)
        
        # Get card rank and suit
        if rank_model_loaded and suit_model_loaded:
            # Use rank model for rank image and suit model for suit image
            rank = predict_card_with_models(preprocessed_rank)[0]  # Get only rank
            suit = predict_card_with_models(preprocessed_suit)[1]  # Get only suit
        else:
            # Fallback using simple recognition
            rank, _ = simple_card_recognition(screenshot_rank)
            _, suit = simple_card_recognition(screenshot_suit)
            
        return {"rank": rank, "suit": suit, "empty": False}
        
    def _detection_loop(self):
        """Main loop for card detection"""
        # Define card regions (rank and suit for each card)
        player_card_regions = [
            {"rank": (621, 463, 17, 23), "suit": (622, 487, 14, 17)},  # Player Card 1
            {"rank": (686, 463, 17, 23), "suit": (687, 487, 14, 17)}   # Player Card 2
        ]
        
        table_card_regions = [
            {"rank": (516, 257, 17, 21), "suit": (517, 280, 14, 17)},  # Table Card 1
            {"rank": (585, 257, 17, 21), "suit": (586, 280, 14, 17)},  # Table Card 2
            {"rank": (653, 257, 17, 21), "suit": (654, 280, 14, 17)},  # Table Card 3
            {"rank": (722, 257, 17, 21), "suit": (723, 280, 14, 17)},  # Table Card 4
            {"rank": (791, 257, 17, 21), "suit": (792, 280, 14, 17)}   # Table Card 5
        ]
        
        while self.running:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            # Process player cards
            player_cards = []
            for i, regions in enumerate(player_card_regions):
                # Take screenshots of rank and suit regions
                screenshot_rank = take_screenshot(region=regions["rank"])
                screenshot_suit = take_screenshot(region=regions["suit"])
                
                # Process the card
                card = self.process_card_regions(screenshot_rank, screenshot_suit, i)
                player_cards.append(card)
            
            # Process table cards
            table_cards = []
            non_empty_count = 0
            
            for i, regions in enumerate(table_card_regions):
                # Take screenshots of rank and suit regions
                screenshot_rank = take_screenshot(region=regions["rank"])
                screenshot_suit = take_screenshot(region=regions["suit"])
                
                # Process the card
                card = self.process_card_regions(screenshot_rank, screenshot_suit, i)
                table_cards.append(card)
                
                # Count non-empty cards
                if not card.get('empty', False):
                    non_empty_count += 1
            
            # Update game state based on number of non-empty table cards
            self.game_state = GAME_STATES.get(non_empty_count, "Unknown")
            
            # Update the card lists
            self.player_cards = player_cards
            self.table_cards = table_cards
            
            # Calculate hand analysis if we have player cards
            if any(not card.get('empty', False) for card in player_cards):
                self.hand_analysis = get_hand_analysis(player_cards, table_cards)
            else:
                self.hand_analysis = None
            
            # Sleep to avoid high CPU usage
            time.sleep(2)
        
    def run_interface(self):
        """Run the main interface loop"""
        self.start_detection()
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Clear the screen
            self.screen.fill(self.bg_color)
            
            # Display game state
            self.screen.blit(self.font.render(f"Game State: {self.game_state}", True, (255, 255, 255)), (20, 20))
            
            # Display player cards
            y_position = 50
            self.screen.blit(self.font.render("Player Cards:", True, (255, 255, 255)), (20, y_position))
            y_position += 30
            
            if not self.player_cards:
                self.screen.blit(self.font.render("No player cards detected", True, (255, 255, 255)), (20, y_position))
                y_position += 25
            else:
                for i, card in enumerate(self.player_cards):
                    if card.get('empty', False):
                        card_text = f"Card {i+1}: Empty"
                    else:
                        card_text = f"Card {i+1}: {card['rank']} of {card['suit']}"
                    self.screen.blit(self.font.render(card_text, True, (255, 255, 255)), (20, y_position))
                    y_position += 25
            
            # Display table cards
            y_position += 20
            self.screen.blit(self.font.render("Table Cards:", True, (255, 255, 255)), (20, y_position))
            y_position += 30
            
            if not self.table_cards:
                self.screen.blit(self.font.render("No table cards detected", True, (255, 255, 255)), (20, y_position))
            else:
                for i, card in enumerate(self.table_cards):
                    if card.get('empty', False):
                        card_text = f"Card {i+1}: Empty"
                    else:
                        card_text = f"Card {i+1}: {card['rank']} of {card['suit']}"
                    self.screen.blit(self.font.render(card_text, True, (255, 255, 255)), (20, y_position))
                    y_position += 25
            
            # Display hand analysis
            y_position += 40
            self.screen.blit(self.font.render("Hand Analysis:", True, (255, 255, 255)), (20, y_position))
            y_position += 30
            
            if self.hand_analysis:
                self.screen.blit(self.font.render(f"Hand: {self.hand_analysis['hand_name']}", True, (255, 255, 255)), (20, y_position))
                y_position += 25
                self.screen.blit(self.font.render(f"Win Probability: {self.hand_analysis['win_probability']:.2f}%", True, (255, 255, 255)), (20, y_position))
            else:
                self.screen.blit(self.font.render("No valid hand detected", True, (255, 255, 255)), (20, y_position))
            
            # Update the display
            pygame.display.flip()
            self.clock.tick(30)
        
        self.running = False
        pygame.quit()

if __name__ == "__main__":
    # Create data directories if they don't exist
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    suits_dir = os.path.join(data_dir, "cards_suits")
    numbers_dir = os.path.join(data_dir, "cards_numbers")
    empty_dir = os.path.join(data_dir, "empty_positions")
    
    os.makedirs(suits_dir, exist_ok=True)
    os.makedirs(numbers_dir, exist_ok=True)
    os.makedirs(empty_dir, exist_ok=True)
    
    app = PokerCV()
    app.run_interface()