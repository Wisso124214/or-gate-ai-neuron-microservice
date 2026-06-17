# Neurona Compuerta OR (Arquitectura Hexagonal)

Este proyecto implementa una sola Neurona (Perceptrón) configurada para aprender el comportamiento de una compuerta OR usando Python, Base de datos en memoria (ChromaDB) y una clara separación de la arquitectura utilizando la **Arquitectura Hexagonal (Puertos y Adaptadores)**.

## Arquitectura

- **Archivos `.env`** _(Configuraciones externas)_: Contienen hiperparámetros como tasas de aprendizaje, número de épocas y los datos de set de pruebas en Listas JSON preparadas.
- **`domain/`** _(El Núcleo / Entidades)_:
  - `neuron.py`: Clase pura que solo modela los datos de los pesos, el sesgo y la tasa.
  - `activation.py`: Extrae las funciones matemáticas como la métrica y evaluación (función sigmoide).
- **`application/`** _(Casos de Uso)_:
  - `predict.py`: Manipula las multiplicaciones de entradas usando las variables de la clase y la función matemática.
  - `train.py`: Maneja bucles usando las funciones de predicciones para estimar y reducir el margen de error, y dictamina en qué punto exacto es mejor persistir la data en nuestra capa de Base de Datos.
- **`infrastructure/`** _(Los Adaptadores o Enchufes para interactuar con lo de afuera)_:
  - `adapter_config.py`: Traduce los archivos `.env` a valores con los que la red puede entender y trabajar en python.
  - `adapter_db.py`: Centraliza TODO ChromaDB conectando, leyendo y suscribiendo los datos en disco (`upsert`).
- **Puntos de Entrada / CLI** _(Drivers / Primary Adapters)_:
  - `init_neuron.py`: Programa principal para crear el estado vital de la neurona y crear la BD.
  - `run_train.py`: CLI de entrenamiento (puedes ejecutarlo un sinfín de veces).
  - `run_predict.py`: CLI para testear los diferentes inputs de forma interactiva (se trae la configuración siempre actualizadita desde la base de datos).

---

## 🛠 Instalación Inicial

Asegúrate de que estás en la raíz principal. Si tienes un pip virtual (venv) teniéndolo activado.

1. Instala los conectores requeridos de Chroma e implementadores de entornos `.env`:

```bash
pip install chromadb python-dotenv colorama
```

> La aplicación por defecto espera estar alojada dentro de `or-gate/v2`

## 🚀 Cómo Usarlo como Microservicio

Para probarlo y usar la capacidad asincrónica en "otras terminales":

### 1️⃣ Inicializa el Servidor / Creador

Abre tu 1.ª terminal, entra en la carpeta correspondiente donde hiciste clone del proyecto y corre:

```bash
python or-gate/v2/init_neuron.py
```

> **Nota**: Esto validará tus archivos `.env`, guardará todo en los `pesos a 0` (o últimos usados si lo cierras) y creará la carpeta local SQLite `neuron_db` dejándolo activado y escuchando.

### 2️⃣ (Terminal 2) Realiza tu Entrenamiento

Mantén la primera terminal intacta, abre otra terminal e inicia una orden masiva de "aprendizaje" corriendo las 5000 épocas. A medida que avance, le irá guardando los parámetros a la base de datos en directo.

```bash
python or-gate/v2/run_train.py
```

### 3️⃣ (Terminal 3) Interactúa y predice "en vivo"

Con el entrenamiento finalizado (o incluso en pleno proceso!), abres una 3.ª terminal y ejecuta tu probador interactivo para la Compuerta OR:

```bash
python or-gate/v2/run_predict.py
```

Te preguntará por consola las variables de entradas! Al estar desconectada e independiente, la misma se encargará de rearmar la instancia consultando la base de datos!
