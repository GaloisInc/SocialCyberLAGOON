import os
import torch


DATA_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/data'
RESULTS_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/results'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOXICITY_CATEGORIES = [f'computed_badwords_{category}' for category in ['googleInstantB_any', 'swearing_any', 'mrezvan94Harassment_Appearance', 'mrezvan94Harassment_Generic', 'mrezvan94Harassment_Intelligence', 'mrezvan94Harassment_Politics', 'mrezvan94Harassment_Racial', 'mrezvan94Harassment_Sexual']]