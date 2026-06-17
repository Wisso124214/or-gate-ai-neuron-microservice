from domain.neuron import Neuron
from application.predict import predict
from infrastructure.adapter_config import load_main_config
from infrastructure.adapter_db import NeuronRepository
from colorama import Fore, Style, init

init(autoreset=True)

def main():
    print(f"{Fore.CYAN}--- MÓDULO DE PREDICCIÓN Y PRUEBAS INTERACTIVAS ---")
    
    # Cargas Iniciales y Conexión a BD
    _, lr, initial_bias, initial_weights = load_main_config()
    repo = NeuronRepository()
    
    print(f"{Fore.WHITE}Conectándose a Base de Datos en caliente...\n")

    while True:
        # Envolviéndolo en input para pruebas continuas validando la entrada
        while True:
            x1 = input(f"{Fore.MAGENTA}Ingrese el valor de la primera entrada (0 o 1): {Style.RESET_ALL}{Fore.WHITE}")
            if x1 in ['0', '1']:
                break
            print(f"{Fore.RED}Entrada inválida. Por favor, ingrese 0 o 1.")
            
        while True:
            x2 = input(f"{Fore.MAGENTA}Ingrese el valor de la segunda entrada (0 o 1): {Style.RESET_ALL}{Fore.WHITE}")
            if x2 in ['0', '1']:
                break
            print(f"{Fore.RED}Entrada inválida. Por favor, ingrese 0 o 1.")
            
        # Re-construimos la neurona con los datos más recientes de la BD!
        # De esta forma si otra terminal acaba de entrenar, la predicción será super exacta.
        weights, bias, learning_rate = repo.get_params(initial_weights, initial_bias, lr)
        neuron = Neuron(weights, bias, learning_rate)
        
        # Efectuar Operación (Caso de Uso)
        result = predict(neuron, [int(x1), int(x2)])
        
        print(f"{Fore.WHITE}Entrada actual : {Fore.CYAN}[{x1}, {x2}]")
        print(f"{Fore.WHITE}Predicción cruda: {Fore.YELLOW}{result:.4f}")
        print(f"{Fore.WHITE}Salida final   : {Fore.GREEN}{round(result)}\n")

        continue_prompt = input(f"{Fore.MAGENTA}¿Desea probar otra entrada? (s/n): {Style.RESET_ALL}{Fore.WHITE}")
        if continue_prompt.lower() == 'n':
            print(f"{Fore.CYAN}Saliendo de la terminal de pruebas.")
            break

if __name__ == "__main__":
    main()