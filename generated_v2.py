# Character-level text generator: a small LSTM trained on ten short sentences.

import numpy as np
import tensorflow as tf


# ---------- Step 1: reproducibility ----------
SEED = 1710
if SEED is not None:
    tf.keras.utils.set_random_seed(SEED)
rng = np.random.default_rng(SEED)


# ---------- Step 2: training text and vocabulary ----------
sentences = [
    "I like cats.",
    "I like dogs.",
    "I like noodles.",
    "I like tacos.",
    "I like books.",
    "I like robots.",
    "I like music.",
    "I like puzzles.",
    "I like pizza.",
    "I like coding.",
]
text = "\n".join(sentences)

chars = sorted(set(text))
stoi = {char: i for i, char in enumerate(chars)}
itos = {i: char for i, char in enumerate(chars)}
vocab_size = len(chars)

print("Text length:", len(text))
print("Vocabulary:", chars)


# ---------- Step 3: (40 characters -> next character) training pairs ----------
seq_len = 40
step = 1

X_ids, y_ids = [], []
for i in range(0, len(text) - seq_len, step):
    X_ids.append([stoi[char] for char in text[i : i + seq_len]])
    y_ids.append(stoi[text[i + seq_len]])

X = np.array(X_ids, dtype=np.int32)
y = np.array(y_ids, dtype=np.int32)
print("Number of training samples:", len(X))


# ---------- Step 4: model ----------
model = tf.keras.Sequential(
    [
        # Each character id becomes a vector of 32 numbers.
        tf.keras.layers.Embedding(vocab_size, 32),
        # Reads the 40 vectors in order; returns only its final 128-number memory.
        tf.keras.layers.LSTM(128),
        # One raw score (logit) per possible next character.
        tf.keras.layers.Dense(vocab_size),
    ]
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    # The Dense layer outputs logits, not probabilities.
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)


# ---------- Step 5: training ----------
history = model.fit(X, y, batch_size=64, epochs=20, verbose=0)
print("Final loss:", history.history["loss"][-1])


# ---------- Step 6: helper functions ----------
def sample_logits(logits, temperature=1.0):
    """Pick the id of the next character."""
    if temperature <= 0:
        return int(np.argmax(logits))
    probabilities = tf.nn.softmax(logits / temperature).numpy()
    return int(rng.choice(len(probabilities), p=probabilities))


def next_char_probabilities(seed, temperature=1.0, top_n=5):
    """Top (character, probability) pairs for the next character, highest first."""
    seed = seed.rjust(seq_len)
    context = [stoi.get(char, 0) for char in seed[-seq_len:]]
    logits = model.predict(np.array([context], dtype=np.int32), verbose=0)[0]

    if temperature <= 0:
        probabilities = np.zeros(len(logits))
        probabilities[np.argmax(logits)] = 1.0
    else:
        probabilities = tf.nn.softmax(logits / temperature).numpy()

    top_ids = np.argsort(probabilities)[::-1][:top_n]
    return [(itos[int(i)], float(probabilities[i])) for i in top_ids]


def generate(seed, n, temperature=1.0):
    """Extend the seed by n characters; the result keeps the padding spaces."""
    seed = seed.rjust(seq_len)
    context = [stoi.get(char, 0) for char in seed[-seq_len:]]
    output = list(seed)
    for _ in range(n):
        logits = model.predict(np.array([context], dtype=np.int32), verbose=0)[0]
        next_id = sample_logits(logits, temperature)
        output.append(itos[next_id])
        # Slide the window: drop the oldest id, add the new one.
        context = context[1:] + [next_id]
    return "".join(output)


# ---------- Step 7: compare temperatures ----------
prompt = "I like "
max_text_chars = 188  # seed plus new characters, not counting padding spaces
n_new_chars = min(180, max_text_chars - len(prompt))

for temperature in [0.1, 0.5, 0.7, 1.0]:
    print(f"\n=== Temperature {temperature} ===")
    print("Top 5 probabilities for the first generated character:")
    for char, probability in next_char_probabilities(prompt, temperature):
        print(f"  {char!r}: {probability:.1%}")
    print(generate(prompt, n_new_chars, temperature))
