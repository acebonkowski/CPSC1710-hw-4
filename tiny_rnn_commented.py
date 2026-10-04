# We import our two necessary tools
import numpy as np
import tensorflow as tf

# We define the starting value by a) setting it at a whole number for comparability or b) leaving it as None for random initialization.
SEED = 1710
# If seed is defined, pass it to our ML libraries (tensorflow, keras) to randomize their starting value too.
if SEED is not None:
    tf.keras.utils.set_random_seed(SEED)
# Define attitional seed/starting value 
rng = np.random.default_rng(SEED)

# 1) Define, sort & prepare the text to train on
## Set of sentences to train the model on.
tiny_lines = [
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
# Glue list of 10 sentences together with new starting line for each.
text = "\n".join(tiny_lines)

# Alternative text to glue together & train on: 
# DNA: text = "TATAAA\nCGCGCG\nATG...TAA\nACGTACGTACGT\n"
# Emoji: text = "☀️🌤️⛅🌧️⛈️🌈\n🍞🧈🍯\n🥚🍳🍞\n🙂➡️😊\n"
# Nursery: text = "Twinkle twinkle little star,\nHow I wonder what you are.\n"

print("Corpus length:", len(text)) # Output character length of glued together text
chars = sorted(list(set(text))) # Define list of different characters in the text & sort (new line > spaces > punctuation > capitalized letters > non-capitalized letters) w/o duplicates via set()
stoi = {c: i for i, c in enumerate(chars)} # Covert characters to numbers
itos = {i: c for c, i in stoi.items()} # Puts number-character-pairs into dictionary (= list of lists)
vocab_size = len(chars) # Output number of non-duplicate characters in text
print("Vocab:", chars) # Print list of sorted non-duplicate characters


#2 Transform Sequence Characters & Next Letters into Array (number grid)
seq_len = 40 # Read 40 characters before predicting the next one...
step = 1 # ...by sliding one character after every step.
X_idx, y_idx = [], [] # define to containers for x & y to collect question & answer
for i in range(0, len(text) - seq_len, step): 
    seq = text[i : i + seq_len]  # define sequence as consisting of character #i to character #i + 40 (excl.) 
    nxt = text[i + seq_len]  # define the next character following the sequence
    X_idx.append([stoi[c] for c in seq])  # convert each sequence letter into an integer & store it in container x
    y_idx.append(stoi[nxt])  # convert next character following the sequence into integer and score in container y 

X = np.array(X_idx, dtype=np.int32) # convert list of sequence numbers into a grid of numbers of type whole number integer
y = np.array(y_idx, dtype=np.int32) # convert list of next numbers into a grid of numbers of type whole number integer
print("Num training samples:", len(X)) # print # of training samples, calculated as total # of characters in text (143) - necessary sequence length (40) = 103


#3 Define the model according to sequential steps with 3 layers
model = tf.keras.Sequential(
    [
        # define that each of the 22 characters is represented by random 32 numbers (32 dimension vector) that can be later used to adapt depending on "similarity" of one a to another a
        tf.keras.layers.Embedding(vocab_size, 32), # result is 22 * 32 e= 704 embeddings
        tf.keras.layers.LSTM(128), # create 128-dimension memory rewritten after each step
        # turn 128 numbers into 22 scores (one per distinct character)
        tf.keras.layers.Dense(vocab_size),
    ]
)

#chose optimizer & loss functions for training
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-2), #Adam is learning rate of 1e-2 = 0.01
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # quantify model‘s "suprise" about answer & convert into %
)

#4 Train the Model
# We are training with a batch size of 64 (of 103 questions) across 20 rounds & print the final loss after epoch 20
history = model.fit(X, y, batch_size=64, epochs=20, verbose=0)
print("Final loss:", history.history["loss"][-1])


#5 Define necessary functions: 1) to return the top 5 most likeliest chars, 2) to sample the next char & output probability and 3) to generate text based on a seed, temperature & output length
# Temperature determined how likely the model picks the most likeliest choice as number approaches 0)
def sample_logits(logits, temperature=1.0):
    # function that returns the most likely character is temp is ≤ 0 (greedy)
    if temperature <= 0:  # greedy
        return int(np.argmax(logits))
    logits = logits / temperature # score is derived from dividing it by the temperature
    probabilities = tf.nn.softmax(logits).numpy() # softmax turns scores into probabilities that add up to 100
    return int(rng.choice(len(probabilities), p=probabilities))


def next_char_probabilities(seed="I like ", temperature=1.0, top_n=5): #return the top 5 characters
    """Return the most likely next characters for a prompt."""
    seed = seed if len(seed) >= seq_len else (" " * (seq_len - len(seed)) + seed) # ensure seed always consists of 40 characters by adding missing ones to front
    # look at last 40 characters (= sequence length) & convert the context characters into corresponding numbers (use 0 if not in dict)
    context = [stoi.get(c, 0) for c in seed[-seq_len:]]
    # convert into 2D array for with one row of 40 values
    x = np.array([context], dtype=np.int32)
    # runs 40 ints through LSMT & Dense, returning the output shape [1, 22] in [0]
    logits = model.predict(x, verbose=0)[0]

    if temperature <= 0: #greedy selection, pick the most likely character
        probabilities = np.zeros_like(logits, dtype=np.float64)
        probabilities[np.argmax(logits)] = 1.0
    else:  # probabilistic selection based on temperature
        probabilities = tf.nn.softmax(logits / temperature).numpy()

    # return the top 5 characters based on probabilities
    top_indices = np.argsort(probabilities)[-top_n:][::-1]
    return [(itos[int(i)], float(probabilities[i])) for i in top_indices]


def generate(seed="I like ", n_chars=200, temperature=0.7):
    # Ensure seed length is at least seq_len by left-padding with spaces.
    seed = seed if len(seed) >= seq_len else (" " * (seq_len - len(seed)) + seed) # ensure sequence length is 40 by padding if not
    context = [stoi.get(c, 0) for c in seed[-seq_len:]] # Define next 40 last characters as converted integers to look at
    output = list(seed) # Define written characters to be stored in list, starting with seed
    for _ in range(n_chars): #For every of the 200 to be written characters, ...
        x = np.array([context], dtype=np.int32) #... convert it into 2D array with one row for character & its 40 values 
        logits = model.predict(x, verbose=0)[0] #... run it through LSMT & Dense, returning its output shape to [1, 22] to [0]
        idx = sample_logits(logits, temperature) #... return the most likely character based on the score of the last layer & temperature
        char = itos[idx] # ... convert score into character
        output.append(char) # ... add the character to the output list
        context = context[1:] + [idx] # ... update the context by removing the first character and adding the new one
    # Join the list of characters into a single string and return it.
    return "".join(output)


# ---------- 6) Try a few temperatures ----------
for temperature in [0.1, 0.5, 0.7, 1.0]:
    print("\n=== Temperature", temperature, "===")
    print("Top probabilities for the first generated character:")
    # Get the top character probabilities for the next character based on the current seed and temperature.
    for char, probability in next_char_probabilities(
        seed="I like ", temperature=temperature
    ):
        print(f"  {char!r}: {probability:.1%}")
    # Print a 180 character sequence based on the current temperature & the seed.
    print(generate(seed="I like ", n_chars=180, temperature=temperature))
