class Neuron:
    """
    Entidad del dominio que representa una neurona básica.
    """
    def __init__(self, weights: list, bias: float, learning_rate: float):
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate