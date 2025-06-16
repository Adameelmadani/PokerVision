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
from nlp_module import SimpleNLP

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
            print("Using neural networks for card recognition")
            
        # Initialize pygame for the interface - even smaller window
        pygame.init()
        self.screen = pygame.display.set_mode((380, 550))  # Further reduced window size
        pygame.display.set_caption("PokerVision")
        
        # Even smaller fonts
        pygame.font.init()
        self.title_font = pygame.font.SysFont("Arial", 20, bold=True)
        self.header_font = pygame.font.SysFont("Arial", 16, bold=True)
        self.font = pygame.font.SysFont("Arial", 12)
        self.font_bold = pygame.font.SysFont("Arial", 12, bold=True)
        self.small_font = pygame.font.SysFont("Arial", 10)
        self.text_font = self.font  # Adding text_font as an alias for the regular font
        
        # Color scheme
        self.colors = {
            'background': (30, 60, 30),  # Dark green
            'panel': (20, 40, 20),       # Darker green
            'highlight': (40, 80, 40),   # Lighter green
            'text': (220, 220, 220),     # Off-white
            'header': (255, 255, 255),   # White
            'fold': (255, 100, 100),     # Red
            'call': (100, 255, 100),     # Green
            'raise': (255, 255, 100),    # Yellow
            'spades': (50, 50, 50),      # Dark gray
            'hearts': (200, 50, 50),     # Red
            'diamonds': (200, 50, 50),   # Red
            'clubs': (50, 50, 50),       # Dark gray
        }
        
        # Card suit symbols
        self.suit_symbols = {
            'Hearts': '♥',
            'Diamonds': '♦',
            'Clubs': '♣',
            'Spades': '♠'
        }
        
        self.clock = pygame.time.Clock()
        
        # NLP components
        self.nlp = SimpleNLP()
        self.current_nlp_prompt = None
        self.nlp_response = ""
        self.available_prompts = self.nlp.get_available_prompts()
        self.selected_prompt_index = 0

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
                # Update NLP module with current game state
                self.nlp.update_game_state(
                    self.hand_analysis,
                    self.strategy_advice,
                    self.game_state
                )
            else:
                self.hand_analysis = None
                self.strategy_advice = None
            
            # Sleep to avoid high CPU usage
            time.sleep(2)
        
    def run_interface(self):
        """Run the main interface loop"""
        clock = pygame.time.Clock()
        running = True
        
        # Start the detection thread
        self.start_detection()
        
        # Initialize tab system
        tabs = ["Cards", "Analysis", "NLP"]
        active_tab = 0
        
        while running:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # Tab switching
                if event.type == pygame.MOUSEBUTTONDOWN:
                    for i, tab in enumerate(tabs):
                        tab_rect = pygame.Rect(i * 120 + 10, 10, 100, 30)
                        if tab_rect.collidepoint(event.pos):
                            active_tab = i
                
                # Handle NLP tab events
                if active_tab == 2:  # NLP tab
                    self.handle_nlp_events(event, 50)
            
            # Clear the screen
            self.screen.fill((30, 30, 30))
            
            # Draw tabs
            for i, tab in enumerate(tabs):
                tab_rect = pygame.Rect(i * 120 + 10, 10, 100, 30)
                if i == active_tab:
                    pygame.draw.rect(self.screen, (80, 80, 120), tab_rect)
                else:
                    pygame.draw.rect(self.screen, (60, 60, 60), tab_rect)
                self.screen.blit(self.header_font.render(tab, True, (255, 255, 255)), 
                                (tab_rect.x + 25, tab_rect.y + 5))
            
            # Draw active tab content
            if active_tab == 0:  # Cards tab
                self.draw_player_cards(50)
                self.draw_table_cards(200)
            elif active_tab == 1:  # Analysis tab
                self.draw_game_state(50)
                self.draw_hand_analysis(150)
                self.draw_win_probability(270)
            elif active_tab == 2:  # NLP tab
                self.draw_nlp_tab(50)
            
            # Update the display
            pygame.display.flip()
            
            # Cap the frame rate
            clock.tick(30)
        
        # Clean up when exiting
        self.running = False
        pygame.quit()
    
    def draw_header(self):
        """Draw the header section"""
        # Title bar - smaller height
        pygame.draw.rect(self.screen, (20, 40, 20), (0, 0, self.screen.get_width(), 55))
        pygame.draw.line(self.screen, self.colors['highlight'], (0, 55), (self.screen.get_width(), 55), 1)
        
        # App title
        title_surf = self.title_font.render("PokerVision", True, self.colors['header'])
        self.screen.blit(title_surf, (10, 8))
        
        # Game state display
        state_text = f"State: {self.game_state}"
        state_color = (255, 255, 100) if self.game_state != "Pre-flop" else self.colors['text']
        state_surf = self.font.render(state_text, True, state_color)
        self.screen.blit(state_surf, (10, 32))
        
        # Detection method badge
        method_text = f"{self.recognition_method.upper()}"
        method_surf = self.small_font.render(method_text, True, self.colors['text'])
        method_rect = method_surf.get_rect()
        method_rect.topright = (self.screen.get_width() - 15, 12)
        pygame.draw.rect(self.screen, (60, 100, 60), 
                        (method_rect.left - 6, method_rect.top - 3, 
                        method_rect.width + 12, method_rect.height + 6), 
                        border_radius=6)
        self.screen.blit(method_surf, method_rect)
    
    def draw_panel(self, title, subtitle=None, y_position=0, height=None):
        """Draw a panel with title and optional subtitle"""
        padding = 10
        width = self.screen.get_width() - 20
        
        if height is None:
            height = 65 if subtitle else 40
        
        # Panel background
        pygame.draw.rect(self.screen, self.colors['panel'], 
                        (10, y_position, width, height),
                        border_radius=6)
        
        # Panel highlight
        pygame.draw.rect(self.screen, self.colors['highlight'], 
                        (10, y_position, width, height),
                        width=1, border_radius=6)
        
        # Panel title with increased padding
        title_surf = self.header_font.render(title, True, self.colors['header'])
        self.screen.blit(title_surf, (padding + 8, y_position + 8))
        
        # Optional subtitle
        if subtitle:
            subtitle_surf = self.font.render(subtitle, True, self.colors['text'])
            self.screen.blit(subtitle_surf, (padding + 18, y_position + 35))
            
        return y_position + height
    
    def draw_card(self, rank, suit, x, y, width=40, height=60):
        """Draw an even smaller card"""
        # Card background
        pygame.draw.rect(self.screen, (240, 240, 240), 
                        (x, y, width, height),
                        border_radius=3)
        
        # Card border
        pygame.draw.rect(self.screen, (180, 180, 180), 
                        (x, y, width, height),
                        width=1, border_radius=3)
        
        if rank == "Empty":
            # Draw an empty card placeholder
            pygame.draw.line(self.screen, (200, 200, 200), 
                            (x+6, y+6), (x+width-6, y+height-6), 1)
            pygame.draw.line(self.screen, (200, 200, 200), 
                            (x+width-6, y+6), (x+6, y+height-6), 1)
            return
            
        # Determine card color based on suit
        if suit in ['Hearts', 'Diamonds']:
            card_color = (200, 0, 0)
        else:
            card_color = (0, 0, 0)
            
        # Draw rank in corner
        rank_surf = self.small_font.render(rank, True, card_color)
        self.screen.blit(rank_surf, (x + 3, y + 3))
        
        # Draw suit symbols
        suit_symbol = self.suit_symbols.get(suit, '?')
        
        # Small suit in corner
        small_suit = self.small_font.render(suit_symbol, True, card_color)
        self.screen.blit(small_suit, (x + width - 14, y + 3))
        
        # Center suit
        large_suit = pygame.font.SysFont("Arial", 24).render(suit_symbol, True, card_color)
        large_suit_rect = large_suit.get_rect(center=(x + width//2, y + height//2))
        self.screen.blit(large_suit, large_suit_rect)
    
    def draw_player_cards(self, y_position):
        """Draw player cards section"""
        panel_height = 100
        y = self.draw_panel("Player Cards", None, y_position, panel_height)
        
        if not self.player_cards:
            text = self.font.render("No player cards", True, self.colors['text'])
            self.screen.blit(text, (20, y_position + 40))
        else:
            start_x = (self.screen.get_width() - (len(self.player_cards) * 50)) // 2
            for i, card in enumerate(self.player_cards):
                rank = card.get('rank', 'Empty')
                suit = card.get('suit', 'Empty')
                self.draw_card(rank, suit, start_x + i * 50, y_position + 30)
        
        return y
    
    def draw_table_cards(self, y_position):
        """Draw table cards section"""
        panel_height = 100
        y = self.draw_panel("Table Cards", None, y_position, panel_height)
        
        if not self.table_cards:
            text = self.font.render("No table cards", True, self.colors['text'])
            self.screen.blit(text, (20, y_position + 40))
        else:
            start_x = (self.screen.get_width() - (len(self.table_cards) * 45)) // 2
            for i, card in enumerate(self.table_cards):
                rank = card.get('rank', 'Empty')
                suit = card.get('suit', 'Empty')
                self.draw_card(rank, suit, start_x + i * 45, y_position + 30)
        
        return y
    
    def draw_hand_analysis(self, y_position):
        """Draw hand analysis section"""
        panel_height = 85
        
        hand_name = self.hand_analysis['hand_name']
        win_prob = self.hand_analysis['win_probability']
        text = f"{hand_name} | Win: {win_prob:.1f}%"
        
        y = self.draw_panel("Hand Analysis", text, y_position, panel_height)
        
        # Draw win probability bar
        bar_width = self.screen.get_width() - 70
        bar_height = 10
        bar_x = 35
        bar_y = y_position + 60
        
        # Background bar
        pygame.draw.rect(self.screen, (60, 60, 60), 
                        (bar_x, bar_y, bar_width, bar_height))
        
        # Progress bar
        win_prob = self.hand_analysis['win_probability'] / 100
        prob_width = int(win_prob * bar_width)
        
        # Color based on win probability
        if win_prob < 0.3:
            bar_color = (200, 50, 50)  # Red
        elif win_prob < 0.6:
            bar_color = (200, 200, 50)  # Yellow
        else:
            bar_color = (50, 200, 50)  # Green
            
        pygame.draw.rect(self.screen, bar_color, 
                        (bar_x, bar_y, prob_width, bar_height))
        
        # Border
        pygame.draw.rect(self.screen, (200, 200, 200), 
                        (bar_x, bar_y, bar_width, bar_height), 1)
                
        return y
    
    def draw_strategy_advice(self, y_position):
        """Draw strategy advice section"""
        # Calculate content
        advice_text = self.strategy_advice['action']
        reasoning = self.strategy_advice['reasoning']
        position = self.strategy_advice.get('position', 'Unknown')
        
        # Make reasoning more concise to fit smaller window
        if len(reasoning) > 150:
            reasoning = reasoning[:147] + "..."
            
        # Calculate how many lines the reasoning will take
        line_height = 15
        max_width = self.screen.get_width() - 50
        words = reasoning.split()
        lines = 1
        line = ""
        
        for word in words:
            test_line = line + word + " "
            if self.font.size(test_line)[0] < max_width:
                line = test_line
            else:
                lines += 1
                line = word + " "
                
        panel_height = 85 + (lines * line_height)
        y = self.draw_panel("Strategy Advice", None, y_position, panel_height)
        
        # Position indicator with action on same line to save space
        position_strength = self.strategy_advice['position_strength']
        position_color = (100, 255, 100) if position_strength > 0.6 else self.colors['text']
        pos_text = self.font_bold.render(f"Pos: {position}", True, position_color)
        self.screen.blit(pos_text, (20, y_position + 35))
        
        # Action recommendation
        action = self.strategy_advice['action']
        if 'Raise' in action:
            action_color = self.colors['raise']
        elif 'Call' in action:
            action_color = self.colors['call']
        elif 'Fold' in action:
            action_color = self.colors['fold']
        else:
            action_color = self.colors['text']
            
        action_text = self.font_bold.render(action, True, action_color)
        action_rect = action_text.get_rect()
        action_rect.right = self.screen.get_width() - 20
        action_rect.top = y_position + 35
        self.screen.blit(action_text, action_rect)
        
        # Reasoning with word wrap
        y_text = y_position + 60
        x_text = 20
        
        self.screen.blit(self.font_bold.render("Reasoning:", True, self.colors['text']), 
                        (x_text, y_text))
        y_text += line_height
        
        # Word wrap
        line = ""
        for word in words:
            test_line = line + word + " "
            if self.font.size(test_line)[0] < max_width:
                line = test_line
            else:
                self.screen.blit(self.font.render(line, True, self.colors['text']), 
                                (x_text + 5, y_text))
                y_text += line_height
                line = word + " "
                
        if line:
            self.screen.blit(self.font.render(line, True, self.colors['text']), 
                            (x_text + 5, y_text))
        
        return y

    def draw_game_state(self, y_position):
        """Draw game state section (new for tab system)"""
        panel_height = 65
        y = self.draw_panel("Game State", f"Current state: {self.game_state}", y_position, panel_height)
        return y
    
    def draw_win_probability(self, y_position):
        """Draw win probability section (new for tab system)"""
        if not self.hand_analysis:
            y = self.draw_panel("Win Probability", "No data available", y_position, 65)
            return y
            
        panel_height = 85
        win_prob = self.hand_analysis['win_probability']
        y = self.draw_panel("Win Probability", f"{win_prob:.1f}%", y_position, panel_height)
        
        # Draw win probability bar
        bar_width = self.screen.get_width() - 70
        bar_height = 10
        bar_x = 35
        bar_y = y_position + 60
        
        # Background bar
        pygame.draw.rect(self.screen, (60, 60, 60), 
                        (bar_x, bar_y, bar_width, bar_height))
        
        # Progress bar
        win_prob = self.hand_analysis['win_probability'] / 100
        prob_width = int(win_prob * bar_width)
        
        # Color based on win probability
        if win_prob < 0.3:
            bar_color = (200, 50, 50)  # Red
        elif win_prob < 0.6:
            bar_color = (200, 200, 50)  # Yellow
        else:
            bar_color = (50, 200, 50)  # Green
            
        pygame.draw.rect(self.screen, bar_color, 
                        (bar_x, bar_y, prob_width, bar_height))
        
        # Border
        pygame.draw.rect(self.screen, (200, 200, 200), 
                        (bar_x, bar_y, bar_width, bar_height), 1)
                
        return y
        
    def draw_nlp_tab(self, y_position):
        """Draw the NLP assistant tab"""
        # Draw NLP panel
        panel_height = 200
        self.draw_panel("NLP Assistant", "Ask about poker or the system", y_position, panel_height)
        
        # Display available prompts as a selectable list
        prompt_y = y_position + 50
        prompt_x = 20
        
        # Draw the currently selected prompt with up/down navigation
        self.screen.blit(self.header_font.render("Select a question:", True, (255, 255, 255)), (prompt_x, prompt_y))
        
        # Draw up arrow button 
        up_arrow_rect = pygame.Rect(340, prompt_y - 5, 20, 20)
        pygame.draw.rect(self.screen, (80, 80, 80), up_arrow_rect)
        pygame.draw.polygon(self.screen, (200, 200, 200), 
            [(340 + 10, prompt_y), (340 + 5, prompt_y + 10), (340 + 15, prompt_y + 10)])
        
        # Draw the currently selected prompt with highlighting
        current_prompt = self.available_prompts[self.selected_prompt_index]
        # Truncate if too long
        if len(current_prompt) > 40:
            display_prompt = current_prompt[:37] + "..."
        else:
            display_prompt = current_prompt
        
        # Draw selection background
        prompt_rect = pygame.Rect(prompt_x, prompt_y + 20, 310, 20)
        pygame.draw.rect(self.screen, (50, 70, 50), prompt_rect)
        pygame.draw.rect(self.screen, (100, 120, 100), prompt_rect, 1)  # Border
        
        self.screen.blit(self.text_font.render(display_prompt, True, (220, 220, 0)), 
                        (prompt_x + 5, prompt_y + 22))
        
        # Draw down arrow button
        down_arrow_rect = pygame.Rect(340, prompt_y + 20, 20, 20)
        pygame.draw.rect(self.screen, (80, 80, 80), down_arrow_rect)
        pygame.draw.polygon(self.screen, (200, 200, 200), 
            [(340 + 10, prompt_y + 35), (340 + 5, prompt_y + 25), (340 + 15, prompt_y + 25)])
        
        # Ask button
        ask_button_rect = pygame.Rect(240, prompt_y + 45, 80, 30)
        pygame.draw.rect(self.screen, (100, 150, 100), ask_button_rect)
        self.screen.blit(self.text_font.render("Ask", True, (255, 255, 255)), 
                         (ask_button_rect.x + 25, ask_button_rect.y + 8))
        
        # Display response
        response_y = prompt_y + 85
        response_rect = pygame.Rect(prompt_x, response_y, 340, 80)
        pygame.draw.rect(self.screen, (50, 50, 50), response_rect)
        
        # Add response header
        if self.nlp_response:
            self.screen.blit(self.font_bold.render("Answer:", True, (180, 220, 180)), 
                            (prompt_x + 5, response_y + 5))
            response_start_y = response_y + 25
        else:
            self.screen.blit(self.font.render("Select a question and click 'Ask'", True, (180, 180, 180)), 
                            (prompt_x + 10, response_y + 30))
            response_start_y = response_y + 10
        
        # Wrap text to fit in the response box
        if self.nlp_response:
            words = self.nlp_response.split(' ')
            lines = []
            current_line = ""
            for word in words:
                test_line = current_line + word + " "
                if self.text_font.size(test_line)[0] < response_rect.width - 20:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word + " "
            if current_line:
                lines.append(current_line)
            
            # Draw response text
            for i, line in enumerate(lines[:3]):  # Limit to 3 lines to fit better
                self.screen.blit(self.text_font.render(line, True, (220, 220, 220)), 
                                (response_rect.x + 10, response_start_y + i * 20))

    def handle_nlp_events(self, event, y_position):
        """Handle NLP tab interaction events"""
        prompt_y = y_position + 50
        
        # Check for prompt navigation (up/down)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and self.selected_prompt_index > 0:
                self.selected_prompt_index -= 1
            elif event.key == pygame.K_DOWN and self.selected_prompt_index < len(self.available_prompts) - 1:
                self.selected_prompt_index += 1
        
        # Check for mouse clicks
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check up arrow click
            up_arrow_rect = pygame.Rect(340, prompt_y - 5, 20, 20)
            if up_arrow_rect.collidepoint(event.pos) and self.selected_prompt_index > 0:
                self.selected_prompt_index -= 1
                
            # Check down arrow click
            down_arrow_rect = pygame.Rect(340, prompt_y + 20, 20, 20)
            if down_arrow_rect.collidepoint(event.pos) and self.selected_prompt_index < len(self.available_prompts) - 1:
                self.selected_prompt_index += 1
                
            # Check prompt selection area click
            prompt_rect = pygame.Rect(20, prompt_y + 20, 310, 20)
            if prompt_rect.collidepoint(event.pos):
                # Calculate which prompt to select based on mouse y position
                if event.button == 4:  # Mouse wheel up
                    if self.selected_prompt_index > 0:
                        self.selected_prompt_index -= 1
                elif event.button == 5:  # Mouse wheel down
                    if self.selected_prompt_index < len(self.available_prompts) - 1:
                        self.selected_prompt_index += 1
            
            # Check for Ask button click
            ask_button_rect = pygame.Rect(240, prompt_y + 45, 80, 30)
            if ask_button_rect.collidepoint(event.pos):
                prompt = self.available_prompts[self.selected_prompt_index]
                self.nlp_response = self.nlp.get_response(prompt)

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