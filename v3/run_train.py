from domain.neuron import Neuron
from application.train import train_neuron
from infrastructure.adapter_config import load_main_config, load_training_data
from infrastructure.adapter_db import NeuronRepository
from colorama import Fore, init

init(autoreset=True)

def main():
    print(f"{Fore.CYAN}--- MÓDULO DE ENTRENAMIENTO DE NEURONA ---")
    
    try:
        # Cargar Configuración e Hiperparámetros
        epochs, lr, initial_bias, initial_weights = load_main_config()
        x_data, y_data = load_training_data()
        
        # Adaptador de DB para instanciar el estado actual
        repo = NeuronRepository()
        weights, bias, learning_rate = repo.get_params(initial_weights, initial_bias, lr)
        
        # Construir Entidad (la neurona)
        neuron = Neuron(weights, bias, learning_rate)

        print(f"{Fore.WHITE}Iniciando un entrenamiento de {epochs} épocas...")
        print(f"{Fore.WHITE}Valores pre-entrenamiento: Pesos {neuron.weights} | Sesgo {neuron.bias:.4f}")
        
        # Caso de Uso
        train_neuron(neuron, repo, x_data, y_data, epochs)
    except Exception as e:
        print(f"{Fore.RED}Error durante el entrenamiento: {e}")

if __name__ == "__main__":
    main()