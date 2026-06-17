import os
import json
from dotenv import load_dotenv

def load_main_config():
    """
    Carga las variables de entorno principales.
    """
    load_dotenv('.env')
    epochs = int(os.getenv('EPOCHS', 5000))
    learning_rate = float(os.getenv('LEARNING_RATE', 0.1))
    initial_bias = float(os.getenv('INITIAL_BIAS', 0.0))
    weights_str = os.getenv('INITIAL_WEIGHTS', '0.0,0.0')
    initial_weights = [float(w) for w in weights_str.split(',')]
    
    return epochs, learning_rate, initial_bias, initial_weights

def load_training_data():
    """
    Carga los datos de entrenamiento desde un archivo .env separado.
    """
    load_dotenv('training_data.env')
    x_data = json.loads(os.getenv('X_DATA', '[]'))
    y_data = json.loads(os.getenv('Y_DATA', '[]'))
    return x_data, y_data