import logging
import psycopg
from pgvector.psycopg import register_vector
from typing import List, Tuple
import os
from dotenv import load_dotenv

def load_db_config():
    """
    Carga las variables de entorno principales.
    """
    load_dotenv('.env')
    user = os.getenv('USER_DB', 'postgres')
    password = os.getenv('PASSWORD_DB', 'postgres')
    host = os.getenv('HOST_DB', 'localhost')
    port = os.getenv('PORT_DB', '5432')
    name = os.getenv('NAME_DB', 'db')

    return {'USER_DB': user, 'PASSWORD_DB': password, 'HOST_DB': host, 'PORT_DB': port, 'NAME_DB': name}

db_config = load_db_config()
DEF_DB_URL = f"postgresql://{db_config['USER_DB']}:{db_config['PASSWORD_DB']}@{db_config['HOST_DB']}:{db_config['PORT_DB']}/{db_config['NAME_DB']}"

class NeuronRepository:
    """
    Adaptador de Base de Datos para comunicarse con PostgreSQL (pgvector)
    y persistir los parámetros de la neurona.
    """
    def __init__(self, db_url: str = DEF_DB_URL):
        """
        Inicializa la conexión y asegura que la extensión pgvector y la tabla existan.
        Se reemplaza db_path por un db_url (DSN) válido para PostgreSQL.
        """
        self.db_url = db_url
        
        # Inicializamos la base de datos: habilitar la extensión pgvector y crear la tabla
        try:
            with psycopg.connect(self.db_url, autocommit=True) as conn:
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Usamos el tipo 'vector' sin dimensión fija para permitir flexibilidad.
                # Si el número de inputs fuera inmutable, se podría especificar: vector(N)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS neuron_params (
                        id VARCHAR(50) PRIMARY KEY,
                        weights vector,
                        bias FLOAT,
                        learning_rate FLOAT
                    );
                """)
        except Exception as e:
            logging.warning(f"No se pudo inicializar la tabla o la extensión pgvector: {e}")

    def get_params(self, initial_weights: List[float], initial_bias: float, initial_lr: float) -> Tuple[List[float], float, float]:
        """
        Obtiene los parámetros actuales si existen en la BD. De lo contrario, devuelve los iniciales.
        """
        try:
            with psycopg.connect(self.db_url) as conn:
                # 1. Registrar el tipo vector en la conexión para que lo deserialice automáticamente
                register_vector(conn)
                
                with conn.cursor() as cur:
                    cur.execute("SELECT weights, bias, learning_rate FROM neuron_params WHERE id = 'params';")
                    result = cur.fetchone()
                    
                    # Verificamos que el resultado exista
                    if not result:
                        return initial_weights, initial_bias, initial_lr
                        
                    weights_db, bias_db, learning_rate_db = result
                    
                    # 2. Prevenir errores de tipos o valores nulos extraídos de la BD
                    if weights_db is None or bias_db is None or learning_rate_db is None:
                        raise ValueError("Existen campos nulos en la base de datos.")
                    
                    # pgvector retorna los vectores como arrays/listas. Los forzamos a float.
                    weights: List[float] = [float(w) for w in weights_db]
                    
                    bias = float(bias_db)
                    learning_rate = float(learning_rate_db)

                    # Opcional: validar que la cantidad de pesos guardados coincida con la arquitectura actual
                    if len(weights) != len(initial_weights):
                        raise ValueError("La dimensión del vector recuperado no coincide con los pesos iniciales.")

                    return weights, bias, learning_rate

        except Exception as e:
            # Si hay un error de conexión, fallo al convertir datos, o tabla inexistente,
            # capturamos el error y retornamos los parámetros iniciales como red de seguridad.
            logging.warning(f"Error al recuperar parámetros, usando los valores por defecto: {e}")
            return initial_weights, initial_bias, initial_lr

    def save_params(self, weights: List[float], bias: float, learning_rate: float) -> None:
        """
        Actualiza el documento en la BD con los parámetros actuales.
        """
        try:
            with psycopg.connect(self.db_url) as conn:
                # Registrar tipo vector para que serialice la lista de Python correctamente a PostgreSQL
                register_vector(conn)
                
                with conn.cursor() as cur:
                    # Usamos UPSERT de PostgreSQL (INSERT ... ON CONFLICT DO UPDATE)
                    # para crear el registro si no existe, o actualizarlo si ya existe.
                    cur.execute("""
                        INSERT INTO neuron_params (id, weights, bias, learning_rate)
                        VALUES ('params', %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE
                        SET weights = EXCLUDED.weights,
                            bias = EXCLUDED.bias,
                            learning_rate = EXCLUDED.learning_rate;
                    """, (weights, float(bias), float(learning_rate)))
                
                # psycopg 3 no utiliza autocommit por defecto en bloques transaccionales
                conn.commit()
                
        except Exception as e:
            logging.error(f"Error al guardar los parámetros en la base de datos: {e}")