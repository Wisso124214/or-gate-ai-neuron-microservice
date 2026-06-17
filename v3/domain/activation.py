import math

def sigmoid_activation(z: float) -> float:
    """
    Función de activación Sigmoide.
    Convierte cualquier valor 'z' en un número entre 0 y 1.
    """
    return 1.0 / (1.0 + math.exp(-z))