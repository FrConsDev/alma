# ======================================================================
#  1. Chargement COMPLET du dataset GoEmotions
# ======================================================================

from datasets import load_dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import keras_nlp
from tensorflow.keras import layers, Model
import tensorflow as tf
import keras_nlp
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("Téléchargement du dataset GoEmotions...\n")
dataset = load_dataset("go_emotions")

# ➤ Fusionner train + validation + test en un seul DataFrame
full_df = pd.concat([
    pd.DataFrame(dataset["train"]),
    pd.DataFrame(dataset["validation"]),
    pd.DataFrame(dataset["test"])
]).reset_index(drop=True)

print(f"Total GoEmotions fusionné : {len(full_df):,} lignes")


# ======================================================================
#  2. Reclassification en POSITIVE / NEGATIVE + exclusion NEUTRAL
# ======================================================================

emotion_groups = {
    "joy": "positive", "amusement": "positive", "approval": "positive",
    "gratitude": "positive", "love": "positive", "pride": "positive",
    "relief": "positive", "admiration": "positive", "desire": "positive",

    "anger": "negative", "annoyance": "negative", "disgust": "negative",
    "disappointment": "negative", "disapproval": "negative",
}

neutral_labels = ["neutral", "realization", "surprise"]

id2label = dataset["train"].features["labels"].feature.names

texts, labels = [], []

print("\n Re-classification en cours...\n")

for row in tqdm(full_df.itertuples(), total=len(full_df)):
    emotion_list = [id2label[e] for e in row.labels]

    if any(em in neutral_labels for em in emotion_list):
        continue

    is_positive = any(em in emotion_groups and emotion_groups[em] == "positive"
                      for em in emotion_list)

    label = "positive" if is_positive else "negative"

    texts.append(row.text)
    labels.append(label)

# DataFrame final
df = pd.DataFrame({"text": texts, "label": labels})

print("\nAprès filtre neutral :")
print(df["label"].value_counts())


# ======================================================================
# 3. Équilibrage POSITIVE / NEGATIVE
# ======================================================================

print("\nÉquilibrage des classes (undersampling)…")
min_size = df["label"].value_counts().min()

df_balanced = (
    df.groupby("label", group_keys=False)
      .apply(lambda x: x.sample(n=min_size, random_state=42))
      .reset_index(drop=True)
)

print("\nRépartition équilibrée :")
print(df_balanced["label"].value_counts())


# ======================================================================
#  4. Nettoyage du texte (adapté Whisper)
# ======================================================================

import re
import string

def clean_text_whisper(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)            # URLs
    text = re.sub(f"[{string.punctuation}]", " ", text)  # ponctuation
    text = re.sub(r"[^\w\s]", " ", text)           # emojis et caractères spéciaux
    text = re.sub(r"\s+", " ", text).strip()       # espaces multiples
    return text

df_balanced["text"] = df_balanced["text"].apply(clean_text_whisper)

df_balanced["label_bin"] = df_balanced["label"].map({"negative": 0, "positive": 1})


# ======================================================================
#  5. Split Train / Validation / Test
# ======================================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df_balanced["text"],
    df_balanced["label_bin"],
    test_size=0.20,
    random_state=42,
    stratify=df_balanced["label_bin"]
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.10,
    random_state=42,
    stratify=y_train
)

print("\nSIZES:")
print(f"Train      : {len(X_train):,}")
print(f"Validation : {len(X_val):,}")
print(f"Test       : {len(X_test):,}")


# ======================================================================
#  6. Écriture CSV (format TEXT + LABEL dans un seul fichier)
# ======================================================================

# --- Train ---
train_df = pd.DataFrame({
    "text": X_train,
    "label": y_train
})
train_df.to_csv("train.csv", index=False, header=True)

# --- Validation ---
val_df = pd.DataFrame({
    "text": X_val,
    "label": y_val
})
val_df.to_csv("val.csv", index=False, header=True)

# --- Test ---
test_df = pd.DataFrame({
    "text": X_test,
    "label": y_test
})
test_df.to_csv("test.csv", index=False, header=True)



# ======================================================================
# 7. Exemples de données
# ======================================================================

import random

def show_examples(X, y, name, n=3):
    print(f"\n=== {name.upper()} ({n} exemples) ===")
    idx = random.sample(range(len(X)), n)
    for i in idx:
        print(f"\n📝 Texte : {X.iloc[i]}")
        print(f"🏷️ Label : {y.iloc[i]}")

show_examples(X_train, y_train, "train", 3)
show_examples(X_val, y_val, "validation", 3)
show_examples(X_test, y_test, "test", 3)


# ======================================================================
# 8. Visualisations
# ======================================================================

train_lengths = X_train.apply(lambda x: len(x.split()))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# -----------------------------
# 1) Distribution longueurs
# -----------------------------
ax1.hist(train_lengths, bins=50, edgecolor='black', color='skyblue', alpha=0.7)
mean_len = np.mean(train_lengths)
ax1.axvline(mean_len, color='red', linestyle='--', linewidth=2,
            label=f'Moyenne: {mean_len:.1f} mots')
ax1.set_title("Distribution Longueurs (Train)")
ax1.set_xlabel("Mots")
ax1.set_ylabel("Fréquence")
ax1.legend()
ax1.grid(alpha=0.3)

# -----------------------------
# 2) Répartition classes
# -----------------------------
labels_names = ["Negative", "Positive"]
counts = [np.sum(y_train == 0), np.sum(y_train == 1)]
colors = ['#ff6b6b', '#51cf66']

ax2.bar(labels_names, counts, color=colors, edgecolor="black")
ax2.set_title("Répartition Classes (Train)")
ax2.set_ylabel("Nombre d'exemples")

for i, count in enumerate(counts):
    mean_words = np.mean([len(text.split()) for text, label in zip(X_train, y_train) if label == i])
    ax2.text(i, count + 20, f'{count:,}\nMoy: {mean_words:.1f} mots', 
             ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

SEQUENCE_LENGTH = 128

print("Chargement du préprocesseur DistilBERT...\n")
preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(
    "distil_bert_base_en",
    sequence_length=SEQUENCE_LENGTH
)
print("Preprocesseur chargé !")
print(f"Vocabulaire: {preprocessor.tokenizer.vocabulary_size():,} tokens")
print(f"Longueur de séquence: {SEQUENCE_LENGTH}")


def creer_modele_distilbert(num_classes=2, train_backbone=False):
    distilbert_backbone = keras_nlp.models.DistilBertBackbone.from_preset(
        "distil_bert_base_en",
        trainable=train_backbone
    )

    inputs = {
        "token_ids": layers.Input(shape=(SEQUENCE_LENGTH,), dtype=tf.int32, name="token_ids"),
        "padding_mask": layers.Input(shape=(SEQUENCE_LENGTH,), dtype=tf.int32, name="padding_mask"),
    }

    backbone_outputs = distilbert_backbone(inputs)

    cls_token = backbone_outputs[:, 0, :]

    x = layers.Dropout(0.2)(cls_token)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="distilbert_head_only")
    return model


model_non_finetuned = creer_modele_distilbert(num_classes=2, train_backbone=False)
model_non_finetuned.summary()

test_reviews = [ "I feel so happy today!", "I am terrified of what might happen.", "I'm disappointed and sad about the result.", "I forgot my keys." ]

test_preprocessed = preprocessor(test_reviews)

# --- Test AVANT le fine-tuning ---
print("\n=================== TEST AVANT FINE-TUNING ===================")
predictions_prob = model_non_finetuned.predict(test_preprocessed, verbose=0)
y_pred_classes = np.argmax(predictions_prob, axis=1)
confidences = np.max(predictions_prob, axis=1)

for i, (review, pred_class, conf) in enumerate(zip(test_reviews, y_pred_classes, confidences), 1):
    sentiment = "POSITIF" if pred_class == 1 else "NÉGATIF"
    emoji = "✅" if sentiment == "POSITIF" else "❌"
    print(f"\n{i}. \"{review}\"")
    print(f"   {emoji} {sentiment} (Confiance: {conf*100:.1f}%)")

train_encodings = preprocessor(X_train)
val_encodings   = preprocessor(X_val)

X_train_distilbert = {
    "token_ids": train_encodings["token_ids"],
    "padding_mask": train_encodings["padding_mask"]
}
X_val_distilbert = {
    "token_ids": val_encodings["token_ids"],
    "padding_mask": val_encodings["padding_mask"]
}

y_train_cat = to_categorical(y_train, num_classes=2)
y_val_cat   = to_categorical(y_val, num_classes=2)

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=2,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=1,
    min_lr=1e-7
)

BATCH_SIZE = 16
EPOCHS = 5

model_non_finetuned.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model_non_finetuned.fit(
    X_train_distilbert,
    y_train_cat,
    validation_data=(X_val_distilbert, y_val_cat),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr]
)

loss_train = history.history['loss']
loss_val   = history.history['val_loss']
acc_train  = history.history['accuracy']
acc_val    = history.history['val_accuracy']

# --- Visualisation ---
fig, axes = plt.subplots(1, 2, figsize=(14,5))
fig.suptitle('DistilBERT Fine-Tuned', fontsize=16, fontweight='bold')

# Loss
axes[0].plot(loss_train, label='Train', color='blue', linewidth=2)
axes[0].plot(loss_val, label='Validation', color='blue', linestyle='--', linewidth=2)
axes[0].set_title('Loss', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Accuracy
axes[1].plot(acc_train, label='Train', color='green', linewidth=2)
axes[1].plot(acc_val, label='Validation', color='green', linestyle='--', linewidth=2)
axes[1].set_title('Accuracy', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Prédictions sur l'ensemble de validation
y_pred_probs = model_non_finetuned.predict(X_val_distilbert)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_val_cat, axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Négatif', 'Positif'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Matrice de confusion - DistilBERT')
plt.show()

# Prédictions sur le jeu de validation / test
y_pred_distilbert = model_non_finetuned.predict(X_val_distilbert)
y_pred_classes = np.argmax(y_pred_distilbert, axis=1) 

# Rapport détaillé
print("\n📈 CLASSIFICATION REPORT - DistilBERT Fine-Tuned\n")
print("-" * 80)
print(classification_report(y_val, y_pred_classes, target_names=['Négatif', 'Positif']))

# --- Test APRÈS le fine-tuning ---
print("\n=================== TEST APRÈS FINE-TUNING ===================")
predictions_prob = model_non_finetuned.predict(test_preprocessed, verbose=0)
y_pred_classes = np.argmax(predictions_prob, axis=1)
confidences = np.max(predictions_prob, axis=1)

for i, (review, pred_class, conf) in enumerate(zip(test_reviews, y_pred_classes, confidences), 1):
    sentiment = "POSITIF" if pred_class == 1 else "NÉGATIF"
    emoji = "✅" if sentiment == "POSITIF" else "❌"
    print(f"\n{i}. \"{review}\"")
    print(f"   {emoji} {sentiment} (Confiance: {conf*100:.1f}%)")


