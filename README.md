# Controllable Persian Text Generation: A From-Scratch GPT-2 Implementation

This repository contains a from-scratch implementation of a scaled-down GPT-2 model, built entirely in PyTorch. The primary goal of this project is to demonstrate the inner workings of a modern transformer-based language model by building one from the ground up. The model is trained on the Persian Snappfood Comments dataset to perform **sentiment-controlled text generation**. By conditioning the model on special tokens (`<POSITIVE>` and `<NEGATIVE>`), it can generate new, synthetic Persian comments with a specified sentiment. This project was developed as a practical assignment for the Deep Learning course (Spring 2025).

## Features

* **From-Scratch Implementation:** The core GPT-2 architecture (Causal Self-Attention, MLP, Transformer Blocks) is built from scratch using pure PyTorch.
* **Sentiment Control:** The model is conditioned on special tokens (`<POSITIVE>`, `<NEGATIVE>`) to control the sentiment of the generated text.
* **Persian Text Generation:** Trained on a large corpus of Persian comments to generate coherent, in-domain text.
* **Advanced Decoding:** Implements temperature, top-k, and nucleus (top-p) sampling for flexible and high-quality text generation.

## Core Concepts & Techniques

* **Transformer Architecture (Decoder-Only):** Implements the decoder-only architecture popularized by GPT.
* **Causal Self-Attention:** Uses masked multi-head attention to ensure tokens can only attend to preceding tokens, which is essential for auto-regressive generation.
* **Learnable Positional Embeddings:** Uses a learnable embedding layer (`wpe`) to inject positional information, as opposed to fixed sinusoidal encodings.
* **Sentiment Conditioning:** The model learns to associate the special prefix tokens with the sentiment of the following text, allowing for controlled generation by providing the desired token as a prompt.
* **Text Generation & Decoding:** The `generate` method demonstrates auto-regressive sampling strategies to decode text from the model's probability distributions.

---

## How It Works

This project is a from-scratch implementation of the GPT-2 architecture, demonstrating the core mechanics of a decoder-only transformer. The entire system is built to be modular, explainable, and controllable.

### 1. Core Model Architecture

The model's architecture, defined in `src/model.py`, follows the standard GPT-2 design at a smaller scale for feasible training.

1.  **Input Embeddings:** An input sequence of token IDs (e.g., `[128000, 503, 201, 89]`) is passed through two embedding layers simultaneously:
    * **Token Embeddings (`wte`):** Converts each token ID into a dense vector (size `n_embd=192`). This vector represents "what" the token is.
    * **Positional Embeddings (`wpe`):** A separate, *learnable* embedding layer creates a vector for each *position* (0, 1, 2, 3...) in the sequence. This represents "where" the token is.

2.  **Initial Representation:** The token and positional embeddings are summed element-wise. This combined tensor, which now contains both "what" and "where", is passed through a dropout layer and then fed into the main transformer stack.

3.  **Transformer Blocks (`Block`):** The model is a stack of $N=3$ identical transformer blocks. Each block refines the text representation by gathering and processing information from other tokens. We use a **Pre-Norm** architecture for stability:
    * **Step 1. Causal Self-Attention:** The input `x` is first normalized (`self.ln_1`). This normalized output is then fed into the `CausalSelfAttention` module. The module's output is added back to the original `x` (a residual connection).
    * **Step 2. Feed-Forward Network (MLP):** The output from the attention step is normalized again (`self.ln_2`). This is passed through a two-layer `MLP`. The MLP's output is added back to its input (a second residual connection).

4.  **Final Output:** After passing through all $N$ blocks, the final representation is normalized (`ln_f`) and passed through the final linear layer (`lm_head`). This projects the vector from the embedding dimension $C$ up to the full vocabulary size $V$, producing the **logits** (shape `[B, T, V]`)—the raw, un-normalized scores for every possible next token.

### 2. Key Mechanisms and Algorithms

#### How Causal Self-Attention Works
This is the most critical component for auto-regressive generation. Its purpose is to **prevent a token from "seeing" tokens that come after it in the sequence.**

1.  **Score Calculation:** For each token, the model calculates a *Query* vector ($Q$). It also calculates a *Key* vector ($K$) and a *Value* vector ($V$) for all tokens in the context.
2.  **Relevance Scores:** To decide "how much attention" a token at position $i$ should pay to a token at position $j$, the model computes a score: $Score_{i,j} = Q_i \cdot K_j$. This is done in parallel for all tokens using matrix multiplication: $Scores = QK^T$.
3.  **Masking:** This is the "causal" part. We apply a mask $M$ to the scores *before* the softmax step. This mask is a matrix that sets all values *above* the main diagonal to $-\infty$.
    * $Score_{i,j}$ (where $j > i$) is set to $-\infty$.
    * This means a token at position 2 (e.g., "food") *cannot* see the token at position 3 (e.g., "was"). It can only see itself (pos 2) and the tokens before it (pos 0, 1).
4.  **Softmax:** When we apply $softmax(Scores)$, any score of $-\infty$ becomes $0$ (since $e^{-\infty} = 0$). This effectively zeroes out all "future" positions.
5.  **Final Output:** The resulting attention weights are multiplied by the *Value* vectors ($V$) to create a new representation for each token, now enriched with context from its past.

The full formula is:

$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}} + M)V
$$

Where $M$ is the causal mask (with $-\infty$ in the upper triangle and $0$ elsewhere), and $d_k$ is the dimension of the key vectors for scaling.

#### How Sentiment Conditioning Works
The model isn't just generating text; it's generating text *given a condition*. This is not a hard-coded rule but an emergent property of the training.

1.  **Training Data:** The model is trained on tens of thousands of examples like:
    * `"<POSITIVE> عالی بود ممنون"`
    * `"<NEGATIVE> خیلی سرد بود"`
2.  **Learning Associations:** Through backpropagation, the model learns that the token `<POSITIVE>` is a *strong statistical predictor* for sequences of words like "عالی", "خوب", and "ممنون". Conversely, `<NEGATIVE>` predicts "بد", "سرد", and "افتضاح".
3.  **Steering Generation:** The token embedding for `<POSITIVE>` (a learnable vector) essentially acts as a "control signal." When the model sees this token, it "steers" the subsequent computations, pushing the probability distribution at each step towards the part of its learned vocabulary associated with positive reviews.
4.  **Generation:** When we want to generate a positive comment, we simply feed the model the token `<POSITIVE>` as the starting prompt. The model's training has taught it that the most likely next tokens after this prompt are the beginnings of a positive comment.

#### Training: Next-Token Prediction
The model learns by being trained on a **next-token prediction** task using Cross-Entropy Loss. For any given sequence, we use the logits at position $i$ to predict the actual token at position $i+1$.
* **Input:** `["<POSITIVE>", "غذا", "خوب"]`
* **Target:** `["غذا", "خوب", "بود"]`
The `forward` pass in our `GPT2` model handles this "shifting" automatically by comparing `logits[..., :-1, :]` (all logits except the last) with `labels[..., 1:]` (all labels except the first). This teaches the model to answer the question: "Given the text so far, what is the most likely *next* word?"

#### Generation: Auto-Regressive Sampling
This is where the project comes to life. Generation is an **auto-regressive loop**, meaning we generate one token at a time, feed it back into the model, and then generate the next.

1.  **Prompt:** We start by providing a prompt, which is just our special control token (e.g., `input_ids = ["<POSITIVE>"]`).
2.  **Prediction:** The model takes this prompt and produces logits for the *next* token.
3.  **Sampling:** We now have a probability distribution over the entire vocabulary. Instead of just picking the *most likely* token (greedy decoding), which is repetitive, we *sample* from this distribution using techniques like `temperature`, `top_k`, and `top_p` to control the creativity and coherence of the output.
4.  **Append:** A new token (e.g., `"خیلی"`) is sampled.
5.  **Loop:** The new token is appended to our input sequence. The new sequence (`input_ids = ["<POSITIVE>", "خیلی"]`) is fed back into the model.
6.  This loop repeats $N$ times to generate a complete comment.

---

## Project Structure

```
pytorch-gpt2-persian-sentiment-generation/
├── .gitignore         # Ignores data, models, logs, and Python cache
├── LICENSE            # MIT License file
├── README.md          # This file
├── requirements.txt   # Project dependencies
├── notebooks/
│   └── demo.ipynb     # A guided notebook to run training and generation
├── scripts/
│   ├── train.py       # Main script to train the model
│   └── generate.py    # Script to generate text with a trained model
└── src/
    ├── init.py
    ├── config.py      # GPT2Config class
    ├── dataset.py     # CommentDataset and dataloader functions
    ├── model.py       # CausalSelfAttention, MLP, Block, and GPT2 classes
    └── utils.py       # Logging setup, plotting, and data download helpers
```

## How to Use

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/msmrexe/pytorch-gpt2-persian-sentiment-generation.git
    cd pytorch-gpt2-persian-sentiment-generation
    ```

2.  **Setup Environment and Install Dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Authenticate (One-Time Setup):**
    You must authenticate with Hugging Face to download the tokenizer and Kaggle to download the dataset.
    ```bash
    # 1. Log in to Hugging Face
    huggingface-cli login
    
    # 2. Set up your Kaggle API token
    # Download 'kaggle.json' from your Kaggle account
    # And place it in ~/.kaggle/kaggle.json
    mkdir -p ~/.kaggle
    cp /path/to/your/kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    ```

4.  **Train the Model:**
    Run the `train.py` script. The script will automatically download the dataset to the `data/` directory. You can customize all hyperparameters via arguments.
    ```bash
    python scripts/train.py \
        --epochs 5 \
        --batch_size 32 \
        --lr 1e-4 \
        --n_embd 192 \
        --n_layer 3 \
        --n_head 3
    ```
    The best model will be saved to `models/best_gpt2_model.pt`.

5.  **Generate Text:**
    Use the `generate.py` script to generate text with your trained model.
    ```bash
    # Generate positive comments
    python scripts/generate.py --sentiment positive --num_samples 5
    
    # Generate negative comments with different parameters
    python scripts/generate.py \
        --sentiment negative \
        --num_samples 3 \
        --temperature 1.2 \
        --top_k 50
    ```

6.  **Run the Demo Notebook:**
    For a guided, step-by-step walkthrough, open and run the `notebooks/demo.ipynb` notebook.
    ```bash
    jupyter notebook notebooks/demo.ipynb
    ```

---

## Author

Feel free to connect or reach out if you have any questions!

* **Maryam Rezaee**
* **GitHub:** [@msmrexe](https://github.com/msmrexe)
* **Email:** [ms.maryamrezaee@gmail.com](mailto:ms.maryamrezaee@gmail.com)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.
