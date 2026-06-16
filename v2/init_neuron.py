import time
from domain.neuron import Neuron
from infrastructure.adapter_config import load_main_config
from infrastructure.adapter_db import NeuronRepository
from colorama import Fore, init

init(autoreset=True)

def main():
    print(f"{Fore.CYAN}--- INICIALIZANDO INSTANCIA DE NEURONA ---")
    
    # 1. Cargar config básica desde .env
    epochs, lr, initial_bias, initial_weights = load_main_config()

    # 2. Conectar a BD
    repo = NeuronRepository()

    # 3. Traer datos desde DB si existen, o inyectar los 'initial' recién sacados del .env
    weights, bias, learning_rate = repo.get_params(initial_weights, initial_bias, lr)

    # 4. Asegurarnos que la BD está poblada desde el inicio
    repo.save_params(weights, bias, learning_rate)

    print(f"{Fore.GREEN}¡Neurona en línea y parámetros validados en BD local!")
    print(f"{Fore.WHITE}Pesos Actuales: {weights}")
    print(f"{Fore.WHITE}Sesgo Actual: {bias}")
    print(f"{Fore.YELLOW}\nEl proceso quedará en ejecución. Presiona Ctrl+C para salir.")
    print(f"{Fore.YELLOW}Mientras corre, abre otra terminal para usar 'run_train.py' o 'run_predict.py'.\n")

    try:
        while True:
            # Simulamos el servicio activo.
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Se ha detenido el servicio de la neurona.")

if __name__ == "__main__":
    main()