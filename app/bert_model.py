import tensorflow as tf
import keras_nlp

SEQUENCE_LENGTH = 128

def load_model(model_path: str):
    print(f"Chargement du modèle depuis : {model_path}")

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "DistilBertBackbone": keras_nlp.models.DistilBertBackbone,
            "DistilBertTokenizer": keras_nlp.models.DistilBertTokenizer,
            "DistilBertPreprocessor": keras_nlp.models.DistilBertPreprocessor
        }
    )
    print("Modèle chargé avec succès.")

    preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(
        "distil_bert_base_en",
        sequence_length=SEQUENCE_LENGTH
    )

    return model, preprocessor


def predict(model, preprocessor, text: str):
    encoded = preprocessor([text])
    preds = model.predict(encoded)
    return preds.tolist()
