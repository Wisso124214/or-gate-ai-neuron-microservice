from application.predict import predict
from colorama import Fore

def train_neuron(neuron, repository, training_data, expected_responses, epochs):
    """
    Caso de uso: Bucle de entrenamiento de la red. Ajusta los parámetros
    y delega al repositorio la escritura optimizada en disco cada cierto tiempo.
    """
    for epoch in range(epochs):
        total_error = 0
        
        for inputs, expected_output in zip(training_data, expected_responses):
            prediction = predict(neuron, inputs)
            error = expected_output - prediction
            total_error += abs(error)
            
            # Derivada de sigmoide y Descenso de gradiente
            derivative = prediction * (1.0 - prediction)
            adjustment = error * derivative * neuron.learning_rate
            
            for i in range(len(neuron.weights)):
                neuron.weights[i] += adjustment * inputs[i]
            
            neuron.bias += adjustment
            
        if epoch % 1000 == 0:
            print(f"{Fore.WHITE}Época {epoch}, Error total: {Fore.RED}{total_error:.4f}")
            # Escritura en BD optimizada cada 1000 épocas
            repository.save_params(neuron.weights, neuron.bias, neuron.learning_rate)
            
    # Guardado final tras terminar el ciclo
    repository.save_params(neuron.weights, neuron.bias, neuron.learning_rate)
    print(f"{Fore.GREEN}Entrenamiento completado y guardado final en base de datos realizado.")