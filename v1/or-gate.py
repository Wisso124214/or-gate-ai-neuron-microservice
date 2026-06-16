import math
import chromadb
from colorama import Fore, Style, init

# Inicializar colorama para que los colores se reseteen automáticamente
init(autoreset=True)

class Neurona:
    def __init__(self, num_entradas):
        # Inicializamos los pesos a 0 y el sesgo a 0
        self.pesos = [0.0 for _ in range(num_entradas)]
        self.sesgo = 0.0
        self.tasa_aprendizaje = 0.1 # Qué tan rápido aprende

        # Configuración de ChromaDB
        self.db_client = chromadb.PersistentClient(path="./neuron_db")
        self.collection = self.db_client.get_or_create_collection(name="neuron_params")
        
        # Consultar la BD para ver si ya existen los pesos
        result = self.collection.get(ids=["params"])
        if result and result['ids']:
            # Extraer metadatos de la BD
            metadata = result['metadatas'][0]
            self.pesos = [float(metadata[f"peso_{i}"]) for i in range(num_entradas)]
            self.sesgo = float(metadata["sesgo"])
            self.tasa_aprendizaje = float(metadata["tasa_aprendizaje"])
            print(f"{Fore.GREEN}Parámetros cargados desde ChromaDB.")
        else:
            # Si no hay datos, inicializamos la BD
            self.guardar_en_bd()
            print(f"{Fore.YELLOW}Base de datos inicializada con parámetros por defecto.")

    def guardar_en_bd(self):
        # Preparar metadatos para guardar
        metadata = {
            "sesgo": float(self.sesgo),
            "tasa_aprendizaje": float(self.tasa_aprendizaje)
        }
        for i, w in enumerate(self.pesos):
            metadata[f"peso_{i}"] = float(w)
        
        # Upsert actualiza el registro si existe o lo crea si no existe
        self.collection.upsert(
            ids=["params"],
            documents=["Parámetros de la neurona"],
            metadatas=[metadata]
        )

    def _activacion_sigmoide(self, z):
        # Función que convierte el valor 'z' en un número entre 0 y 1
        return 1.0 / (1.0 + math.exp(-z))

    def predecir(self, entradas):
        # 1. Suma Ponderada: z = (x1*w1) + (x2*w2) + b
        #    Donde:
        #    - x: son las entradas (inputs)
        #    - w: son los pesos (weights)
        #    - b: es el sesgo (bias)
        
        z = self.sesgo
        for i in range(len(entradas)):
            z += entradas[i] * self.pesos[i]
        
        # 2. Pasar por la función de activación
        return self._activacion_sigmoide(z)

    def entrenar(self, datos_entrenamiento, respuestas_reales, epocas):
        # Una "época" es una pasada completa por todos los datos
        for epoca in range(epocas):
            error_total = 0
            
            for entradas, respuesta_real in zip(datos_entrenamiento, respuestas_reales):
                # Paso 1: Hacer una predicción con los pesos actuales
                prediccion = self.predecir(entradas)
                
                # Paso 2: Calcular el error (Diferencia entre lo real y la predicción)
                error = respuesta_real - prediccion
                error_total += abs(error)
                
                # Paso 3: Ajustar los pesos y el sesgo (Descenso del gradiente simplificado)
                # Para la sigmoide, la derivada es prediccion * (1 - prediccion)
                derivada = prediccion * (1.0 - prediccion)
                ajuste = error * derivada * self.tasa_aprendizaje
                
                for i in range(len(self.pesos)):
                    self.pesos[i] += ajuste * entradas[i]
                
                self.sesgo += ajuste
                
            # Imprimir el progreso cada 1000 épocas
            if epoca % 1000 == 0:
                print(f"{Fore.WHITE}Época {epoca}, Error total: {Fore.RED}{error_total:.4f}")
                # Actualizar la base de datos cada vez que mostremos un progreso
                self.guardar_en_bd()
        
        # Actualizar base de datos con los pesos finales al terminar el entrenamiento
        self.guardar_en_bd()
        print(f"{Fore.GREEN}Entrenamiento completado y parámetros finales guardados en ChromaDB.")

# --- PONIENDO A PRUEBA LA NEURONA ---

# Datos de entrada para una compuerta OR
datos_X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
# Respuestas esperadas (Y)
respuestas_Y = [0, 1, 1, 1]

# Instanciamos una neurona que recibe 2 entradas
mi_neurona = Neurona(num_entradas=2)

print(f"{Fore.WHITE}Entrenando la neurona...")
mi_neurona.entrenar(datos_X, respuestas_Y, epocas=5000)

print(f"\n{Fore.WHITE}--- PRUEBA FINAL ---")
for x in datos_X:
    resultado = mi_neurona.predecir(x)
    # Redondeamos para obtener un 0 o un 1 absoluto
    print(f"{Fore.WHITE}Entrada: {Fore.CYAN}{x} {Style.RESET_ALL}{Fore.WHITE}-> Predicción cruda: {Fore.YELLOW}{resultado:.4f} {Style.RESET_ALL}{Fore.WHITE} -> Salida final: {Fore.GREEN}{round(resultado)}")


# Permitir al usuario ingresar nuevas entradas para probar la neurona
while True:
  print("\n")
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
  resultado = mi_neurona.predecir([int(x1), int(x2)])
  print(f"{Fore.WHITE}Entrada: {Fore.CYAN}[{x1}, {x2}] {Style.RESET_ALL}{Fore.WHITE}-> Predicción cruda: {Fore.YELLOW}{resultado:.4f} -> Salida final: {Fore.GREEN}{round(resultado)}")

  continuar = input(f"{Fore.MAGENTA}¿Desea probar otra entrada? (s/n): {Style.RESET_ALL}{Fore.WHITE}")
  if continuar.lower() == 'n':
    break
