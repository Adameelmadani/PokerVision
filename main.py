import time
import threading
import pygame
import os
import sys
import numpy as np
import cv2
import argparse
from screenshot import take_screenshot
from card_detector import detect_cards
from card_recognizer import recognize_cards, preprocess_card_image, predict_card_with_models, simple_card_recognition, rank_model_loaded, suit_model_loaded, empty_model_loaded, predict_empty_position, recognize_card_template_matching, load_card_templates
from poker_evaluator import get_hand_analysis
from poker_advisor import PokerAdvisor
from position_detector import PositionDetector

# Game states
GAME_STATES = {
    0: "Pre-flop",
    3: "Flop",
    4: "Turn",
    5: "River"
}

# Recognition methods
RECOGNITION_METHODS = ["neural", "template"]

class PokerCV:
    def __init__(self, recognition_method="neural"):
        self.running = False
        self.player_cards = []
        self.table_cards = []
        self.game_state = "Pre-flop"
        self.hand_analysis = None
        self.strategy_advice = None
        self.recognition_method = recognition_method
        self.advisor = PokerAdvisor()
        self.position_detector = PositionDetector()
        
        # Not loading empty images directly, will use model predictions
        
# Load templates if using template matching
        if self.recognition_method == "template":
            print("Using template matching for card recognition")
            self.rank_templates, self.suit_templates = load_card_templates()
        else:
            print("Using neural networks for card recognition")        # Initialize pygame for the interface
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("PokerVision Pro")
        
        # Initialize fonts
        self.font = pygame.font.SysFont("Segoe UI", 16)
        self.font_bold = pygame.font.SysFont("Segoe UI", 16, bold=True)
        self.title_font = pygame.font.SysFont("Segoe UI", 24, bold=True)
        self.card_font = pygame.font.SysFont("Segoe UI", 20)
        
        self.clock = pygame.time.Clock()
        
        # Define colors
        self.colors = {
            'bg': (27, 38, 44),         # Dark blue-gray
            'panel': (15, 76, 117),     # Medium blue
            'text': (238, 238, 238),    # Light gray
            'highlight': (187, 225, 250),# Light blue
            'raise': (255, 200, 87),    # Gold
            'call': (111, 223, 163),    # Green
            'fold': (255, 107, 107),    # Red
            'header': (50, 130, 184),   # Blue
            'border': (187, 225, 250, 50)# Semi-transparent light blue
        }
        
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
        
        # Get card rank and suit based on selected recognition method
        if self.recognition_method == "neural" and rank_model_loaded and suit_model_loaded:
            # Use neural network models
            rank = predict_card_with_models(preprocessed_rank)[0]  # Get only rank
            suit = predict_card_with_models(preprocessed_suit)[1]  # Get only suit
        elif self.recognition_method == "template":
            # Use template matching
            rank = recognize_card_template_matching(screenshot_rank, self.rank_templates)
            suit = recognize_card_template_matching(screenshot_suit, self.suit_templates)
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
            self.table_cards = table_cards            # Calculate hand analysis if we have player cards
            if any(not card.get('empty', False) for card in player_cards):
                self.hand_analysis = get_hand_analysis(player_cards, table_cards)
                
                # Detect position and get strategy advice
                position = self.position_detector.detect_position()
                self.strategy_advice = self.advisor.analyze_situation(
                    player_cards,
                    table_cards,
                    position=position
                )
            else:
                self.hand_analysis = None
                self.strategy_advice = None
              # Sleep to avoid high CPU usage
            time.sleep(2)

    def draw_panel(self, x, y, width, height, title=None):
        """Draw a panel with a border and optional title"""
        # Draw panel background
        panel_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, self.colors['panel'], panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, self.colors['border'], panel_rect, 1, border_radius=10)
        
        # Draw title if provided
        if title:
            title_surf = self.title_font.render(title, True, self.colors['highlight'])
            title_rect = title_surf.get_rect(topleft=(x + 15, y + 10))
            self.screen.blit(title_surf, title_rect)
            return y + 45  # Return y position after title
        return y + 15     # Return y position if no title

    def run_interface(self):
        """Run the main interface loop"""
        self.start_detection()
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Clear the screen
            self.screen.fill(self.colors['bg'])
            
            # Create three main panels
            left_panel_width = 250
            right_panel_width = 520
            panel_margin = 15
              # Left panel - Game Info
            y_pos = 20
            y_pos = self.draw_panel(panel_margin, y_pos, left_panel_width, 560, "Game Statistics")
            
            # Game state with visual indicator
            state_y = y_pos + 10
            state_text = self.title_font.render(self.game_state, True, self.colors['highlight'])
            state_rect = state_text.get_rect(centerx=panel_margin + left_panel_width/2, top=state_y)
            
            # State indicator circle
            circle_color = {
                'Pre-flop': self.colors['fold'],
                'Flop': self.colors['raise'],
                'Turn': self.colors['raise'],
                'River': self.colors['call']
            }.get(self.game_state, self.colors['text'])
            
            pygame.draw.circle(self.screen, circle_color, 
                             (panel_margin + 20, state_rect.centery), 8)
            self.screen.blit(state_text, state_rect)
            
            # Add session statistics
            stats_y = state_y + 60
            stats_data = [
                ("Recognition", self.recognition_method.capitalize()),
                ("Position", self.strategy_advice.get('position', 'Unknown') if self.strategy_advice else 'Unknown'),
                ("Mode", "6-max Table"),
                ("Status", "Active" if self.running else "Stopped")
            ]
            
            for label, value in stats_data:
                # Label in normal color
                label_surf = self.font_bold.render(f"{label}:", True, self.colors['text'])
                self.screen.blit(label_surf, (panel_margin + 20, stats_y))
                
                # Value in highlight color
                value_surf = self.font.render(value, True, self.colors['highlight'])
                self.screen.blit(value_surf, (panel_margin + 120, stats_y))
                stats_y += 30
            
            # Right panel - Cards and Analysis
            y_pos = 20
            cards_panel_height = 260
            y_pos = self.draw_panel(left_panel_width + 2*panel_margin, y_pos, 
                                  right_panel_width, cards_panel_height, "Cards")
              # Display cards section
            cards_x = left_panel_width + 3*panel_margin
            cards_y = y_pos + 10
            
            # Player cards
            self.screen.blit(self.title_font.render("Your Hand", True, self.colors['text']), 
                           (cards_x, cards_y))
            
            if not self.player_cards:
                self.screen.blit(self.font.render("Waiting for cards...", True, self.colors['text']), 
                               (cards_x, cards_y + 35))
            else:
                for i, card in enumerate(self.player_cards):
                    card_rect = pygame.Rect(cards_x + i*120, cards_y + 35, 100, 140)
                    pygame.draw.rect(self.screen, self.colors['highlight'], card_rect, border_radius=5)
                    
                    if card.get('empty', False):
                        pygame.draw.rect(self.screen, self.colors['panel'], card_rect, 2, border_radius=5)
                        empty_text = self.card_font.render("?", True, self.colors['text'])
                        text_rect = empty_text.get_rect(center=card_rect.center)
                        self.screen.blit(empty_text, text_rect)
                    else:
                        # Draw card
                        pygame.draw.rect(self.screen, (255, 255, 255), card_rect, border_radius=5)
                        
                        # Card value
                        value_text = self.card_font.render(card['rank'], True, (0, 0, 0))
                        self.screen.blit(value_text, (card_rect.x + 10, card_rect.y + 10))
                        
                        # Suit (with color)
                        suit_color = (255, 0, 0) if card['suit'] in ['Hearts', 'Diamonds'] else (0, 0, 0)
                        suit_text = self.card_font.render(card['suit'][0], True, suit_color)
                        self.screen.blit(suit_text, (card_rect.x + 10, card_rect.y + 35))                # Table cards
            table_y = cards_y + 190
            self.screen.blit(self.title_font.render("Community Cards", True, self.colors['text']), 
                           (cards_x, table_y))
            
            if not self.table_cards:
                self.screen.blit(self.font.render("Waiting for community cards...", True, self.colors['text']), 
                               (cards_x, table_y + 35))
            else:
                for i, card in enumerate(self.table_cards):
                    card_rect = pygame.Rect(cards_x + i*100, table_y + 35, 80, 112)
                    pygame.draw.rect(self.screen, self.colors['highlight'], card_rect, border_radius=5)
                    
                    # Strategy Analysis Panel
                    strategy_y = cards_y + cards_panel_height + panel_margin
                    strategy_height = 280
                    y_pos = self.draw_panel(left_panel_width + 2*panel_margin, strategy_y, 
                                          right_panel_width, strategy_height, "Strategy Analysis")
                    
                    content_x = left_panel_width + 3*panel_margin
                    content_y = strategy_y + 50
                    
                    if self.hand_analysis:
                        # Hand information with progress bar for win probability
                        hand_text = f"{self.hand_analysis['hand_name']}"
                        self.screen.blit(self.title_font.render(hand_text, True, self.colors['highlight']), 
                                       (content_x, content_y))
                        
                        # Win probability bar
                        prob = self.hand_analysis['win_probability']
                        bar_width = 400
                        bar_height = 25
                        bar_x = content_x
                        bar_y = content_y + 40
                        
                        # Background bar
                        pygame.draw.rect(self.screen, self.colors['panel'], 
                                       (bar_x, bar_y, bar_width, bar_height), border_radius=5)
                        
                        # Progress bar
                        prob_width = int((prob / 100) * bar_width)
                        prob_color = (111, 223, 163) if prob > 60 else \
                                   (255, 200, 87) if prob > 30 else \
                                   (255, 107, 107)
                        pygame.draw.rect(self.screen, prob_color,
                                       (bar_x, bar_y, prob_width, bar_height), border_radius=5)
                        
                        # Win probability text
                        prob_text = f"{prob:.1f}% Win Rate"
                        prob_surf = self.font_bold.render(prob_text, True, self.colors['text'])
                        prob_rect = prob_surf.get_rect(center=(bar_x + bar_width/2, bar_y + bar_height/2))
                        self.screen.blit(prob_surf, prob_rect)
                        
                        if self.strategy_advice:
                            advice_y = content_y + 90
                            
                            # Position indicator
                            position = self.strategy_advice.get('position', 'Unknown')
                            pos_color = self.colors['call'] if self.strategy_advice['position_strength'] > 0.6 else self.colors['text']
                            position_text = f"Position: {position}"
                            self.screen.blit(self.font_bold.render(position_text, True, pos_color), 
                                           (content_x, advice_y))
                            
                            # Action recommendation
                            action = self.strategy_advice['action']
                            action_color = self.colors['raise'] if 'Raise' in action else \
                                         self.colors['call'] if 'Call' in action else \
                                         self.colors['fold'] if 'Fold' in action else \
                                         self.colors['text']
                            
                            action_y = advice_y + 30
                            action_rect = pygame.Rect(content_x, action_y, 480, 40)
                            pygame.draw.rect(self.screen, action_color, action_rect, border_radius=5)
                            
                            action_text = self.title_font.render(action, True, self.colors['bg'])
                            text_rect = action_text.get_rect(center=action_rect.center)
                            self.screen.blit(action_text, text_rect)
                            
                            # Reasoning text with word wrap
                            reasoning_y = action_y + 60
                            reasoning = self.strategy_advice['reasoning']
                            self.screen.blit(self.font_bold.render("Reasoning:", True, self.colors['text']), 
                                           (content_x, reasoning_y))
                            
                            # Word wrap the reasoning text
                            y_offset = 25
                            words = reasoning.split()
                            line = ""
                            for word in words:
                                test_line = line + word + " "
                                if self.font.size(test_line)[0] < 460:  # Leave margin
                                    line = test_line
                                else:
                                    self.screen.blit(self.font.render(line, True, self.colors['text']), 
                                                   (content_x + 10, reasoning_y + y_offset))
                                    y_offset += 20
                                    line = word + " "
                            if line:
                                self.screen.blit(self.font.render(line, True, self.colors['text']), 
                                               (content_x + 10, reasoning_y + y_offset))
                    else:
                        self.screen.blit(self.font.render("Waiting for hand analysis...", True, self.colors['text']), 
                                       (content_x, content_y))
                    
                    if card.get('empty', False):
                        pygame.draw.rect(self.screen, self.colors['panel'], card_rect, 2, border_radius=5)
                        empty_text = self.card_font.render("?", True, self.colors['text'])
                        text_rect = empty_text.get_rect(center=card_rect.center)
                        self.screen.blit(empty_text, text_rect)
                    else:
                        # Draw card
                        pygame.draw.rect(self.screen, (255, 255, 255), card_rect, border_radius=5)
                        
                        # Card value                        value_text = self.card_font.render(card['rank'], True, (0, 0, 0))
                        self.screen.blit(value_text, (card_rect.x + 8, card_rect.y + 8))
                        
                        # Suit (with color)
                        suit_color = (255, 0, 0) if card['suit'] in ['Hearts', 'Diamonds'] else (0, 0, 0)
                        suit_text = self.card_font.render(card['suit'][0], True, suit_color)
                        self.screen.blit(suit_text, (card_rect.x + 8, card_rect.y + 28))
              
            # Start hand analysis section below the cards panel
            y_position = 300  # Fixed position for hand analysis
            
            # Display hand analysis
            self.screen.blit(self.font.render("Hand Analysis:", True, (255, 255, 255)), (20, y_position))
            y_position += 30
            
            if self.hand_analysis:
                self.screen.blit(self.font.render(f"Hand: {self.hand_analysis['hand_name']}", True, (255, 255, 255)), (20, y_position))
                y_position += 25
                self.screen.blit(self.font.render(f"Win Probability: {self.hand_analysis['win_probability']:.2f}%", True, (255, 255, 255)), (20, y_position))
                  # Display strategy advice
                if self.strategy_advice:
                    y_position += 40
                    self.screen.blit(self.font_bold.render("Strategy Advice:", True, (255, 255, 255)), (20, y_position))
                    y_position += 30
                    
                    # Show position and strength indicators
                    position_color = (100, 255, 100) if self.strategy_advice['position_strength'] > 0.6 else (255, 255, 255)
                    self.screen.blit(self.font.render(f"Position: {self.strategy_advice.get('position', 'Unknown')}", True, position_color), (20, y_position))
                    y_position += 25
                    
                    # Color-code the action based on aggressiveness
                    action = self.strategy_advice['action']
                    action_color = (255, 255, 100) if 'Raise' in action else \
                                 (100, 255, 100) if 'Call' in action else \
                                 (255, 100, 100) if 'Fold' in action else \
                                 (255, 255, 255)
                    
                    self.screen.blit(self.font_bold.render(f"Recommended Action:", True, (255, 255, 255)), (20, y_position))
                    y_position += 25
                    self.screen.blit(self.font.render(action, True, action_color), (40, y_position))
                    y_position += 25
                    
                    # Show reasoning with word wrap
                    reasoning = self.strategy_advice['reasoning']
                    self.screen.blit(self.font_bold.render("Reasoning:", True, (255, 255, 255)), (20, y_position))
                    y_position += 25
                    
                    # Word wrap the reasoning text
                    words = reasoning.split()
                    line = ""
                    for word in words:
                        test_line = line + word + " "
                        if self.font.size(test_line)[0] < self.screen.get_width() - 40:
                            line = test_line
                        else:
                            self.screen.blit(self.font.render(line, True, (255, 255, 255)), (40, y_position))
                            y_position += 20
                            line = word + " "
                    if line:
                        self.screen.blit(self.font.render(line, True, (255, 255, 255)), (40, y_position))
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
    templates_dir = os.path.join(data_dir, "templates")
    
    os.makedirs(suits_dir, exist_ok=True)
    os.makedirs(numbers_dir, exist_ok=True)
    os.makedirs(empty_dir, exist_ok=True)
    os.makedirs(templates_dir, exist_ok=True)
    os.makedirs(os.path.join(templates_dir, "ranks"), exist_ok=True)
    os.makedirs(os.path.join(templates_dir, "suits"), exist_ok=True)
    
    # Parse command line arguments with argparse
    parser = argparse.ArgumentParser(description='PokerVision - Card Recognition System')
    parser.add_argument('method', nargs='?', default='neural',
                        choices=RECOGNITION_METHODS,
                        help='Card recognition method: neural (neural networks) or template (template matching)')
    parser.add_argument('--list-methods', action='store_true',
                        help='List available recognition methods')
    
    args = parser.parse_args()
    
    if args.list_methods:
        print("Available recognition methods:")
        print("  neural   - Use trained neural networks for card recognition")
        print("  template - Use OpenCV template matching for card recognition")
        print("\nExample usage: python main.py template")
        sys.exit(0)
    
    recognition_method = args.method
    print(f"Using {recognition_method} method for card recognition")
    
    # Check for template availability if using template matching
    if recognition_method == "template":
        rank_templates, suit_templates = load_card_templates()
        if not rank_templates or not suit_templates:
            print("\nWARNING: Missing template images for template matching")
            print("Please add template images to:")
            for rank in ['2', '3', '4', '5', '6', '7', '8', '9', 'J', 'Q', 'K', 'A']:
                print(f"  {os.path.join(templates_dir, 'ranks', f'{rank}.png')}")
            for suit in ['Clubs', 'Diamonds', 'Hearts', 'Spades']:
                print(f"  {os.path.join(templates_dir, 'suits', f'{suit}.png')}")
            print("\nFalling back to neural network method")
            recognition_method = "neural"
    
    app = PokerCV(recognition_method)
    app.run_interface()