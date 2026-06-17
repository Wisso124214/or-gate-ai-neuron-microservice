from domain.activation import sigmoid_activation

def predict(neuron, inputs: list) -> float:
    """
    Caso de uso: Calcular la predicción basándose en los pesos, el sesgo actual y unas entradas.
    """
    z = neuron.bias
    for i in range(len(inputs)):
        z += inputs[i] * neuron.weights[i]
    
    return sigmoid_activation(z)