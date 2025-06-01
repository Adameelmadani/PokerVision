
import pyautogui
import numpy as np
import cv2

def take_screenshot():
    """
    Take a screenshot of the screen and convert it to a numpy array
    
    Returns:
        numpy.ndarray: Screenshot image in BGR format
    """
    # Capture screenshot
    screenshot = pyautogui.screenshot()
    
    # Convert to numpy array
    image = np.array(screenshot)
    
    # Convert RGB to BGR (OpenCV format)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image

def take_region_screenshot(region):
    """
    Take a screenshot of a specific region
    
    Args:
        region (tuple): Region to capture (left, top, width, height)
        
    Returns:
        numpy.ndarray: Region screenshot in BGR format
    """
    screenshot = pyautogui.screenshot(region=region)
    image = np.array(screenshot)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image