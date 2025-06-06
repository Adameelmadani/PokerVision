import pyautogui
import numpy as np
import cv2
import os

def take_screenshot(region=None, save_path=None):
    """
    Take a screenshot of the screen and convert it to a numpy array
    
    Args:
        region (tuple, optional): Region to capture (left, top, width, height)
        save_path (str, optional): Path to save the screenshot
    
    Returns:
        numpy.ndarray: Screenshot image in BGR format
    """
    # Create screenshots directory if it doesn't exist
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if region:
        # Capture specific region
        screenshot = pyautogui.screenshot(region=region)
    else:
        # Capture full screen
        screenshot = pyautogui.screenshot()
    
    # Convert to numpy array
    image = np.array(screenshot)
    
    # Convert RGB to BGR (OpenCV format)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Save the screenshot if a path is provided
    if save_path:
        cv2.imwrite(save_path, image)
    
    return image