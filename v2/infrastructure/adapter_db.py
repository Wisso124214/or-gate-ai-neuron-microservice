import chromadb
from typing import List, Tuple

class NeuronRepository:
    """
    Adaptador de Base de Datos para comunicarse con ChromaDB y persistir la neurona.
    """
    def __init__(self, db_path="./neuron_db"):
        self.db_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.db_client.get_or_create_collection(name="neuron_params")

    def get_params(self, initial_weights: List[float], initial_bias: float, initial_lr: float) -> Tuple[List[float], float, float]:
        """
        Obtiene los parámetros actuales si existen en la BD. De lo contrario, devuelve los iniciales.
        """
        result = self.collection.get(ids=["params"])

        # 1. Prevenir el error de "None": Verificamos que result exista y tenga identificadores
        if not result or not result.get('ids'):
            return initial_weights, initial_bias, initial_lr
            
        # Verificamos que metadatas exista, no sea None, y tenga al menos un elemento válido
        metadatas = result.get('metadatas')
        if not metadatas or metadatas[0] is None:
            return initial_weights, initial_bias, initial_lr

        metadata = metadatas[0]
        num_inputs = len(initial_weights)
        weights: List[float] = []

        try:
            # 2. Prevenir el error de tipo al convertir a float:
            # Usamos isinstance para confirmar al analizador de tipos que el valor es seguro para float()
            for i in range(num_inputs):
                w_val = metadata.get(f"weight_{i}")
                if w_val is None or not isinstance(w_val, (int, float, str)):
                    raise ValueError(f"Valor nulo o de tipo inválido para weight_{i}")
                weights.append(float(w_val))

            bias_val = metadata.get("bias")
            if bias_val is None or not isinstance(bias_val, (int, float, str)):
                raise ValueError("Valor nulo o de tipo inválido para bias")
            bias = float(bias_val)

            lr_val = metadata.get("learning_rate")
            if lr_val is None or not isinstance(lr_val, (int, float, str)):
                raise ValueError("Valor nulo o de tipo inválido para learning_rate")
            learning_rate = float(lr_val)

            return weights, bias, learning_rate

        except (ValueError, TypeError):
            # Si un dato falta, es de un tipo complejo (ej. SparseVector) o falla la conversión,
            # capturamos el error y retornamos los parámetros iniciales como red de seguridad.
            return initial_weights, initial_bias, initial_lr

    def save_params(self, weights, bias, learning_rate):
        """
        Actualiza el documento en ChromaDB con los parámetros actuales.
        """
        metadata = {
            "bias": float(bias),
            "learning_rate": float(learning_rate)
        }
        for i, w in enumerate(weights):
            metadata[f"weight_{i}"] = float(w)
        
        self.collection.upsert(
            ids=["params"],
            documents=["Parámetros de la neurona"],
            metadatas=[metadata]
        )