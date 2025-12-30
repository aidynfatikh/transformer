# Transformer Models from Scratch

PyTorch implementations of transformer-based models built from scratch for learning purposes.

## Models

### GPT (Generative Pre-trained Transformer)
Character-level language model using decoder-only transformer architecture. Trained on Shakespeare text to generate similar text.

**Architecture:**
- 3 decoder blocks with masked self-attention
- 4 attention heads
- 128-dimensional embeddings
- 256 sequence length
- Trained on `input.txt`

### English-Russian Translator
Sequence-to-sequence transformer for character-level translation between English and Russian.

**Architecture:**
- 8 encoder blocks (bidirectional attention)
- 8 decoder blocks (masked self-attention)
- Cross-attention between encoder and decoder
- 8 attention heads
- 256-dimensional embeddings
- 128 max sequence length
- Trained on Tatoeba parallel corpus

## Setup

```bash
# Install dependencies
pip install torch numpy

# For notebooks
pip install jupyter

# Download training data for translator
mkdir -p data
cd data
wget https://object.pouta.csc.fi/OPUS-Tatoeba/v2/moses/en-ru.txt.zip
unzip en-ru.txt.zip
rm en-ru.txt.zip
cd ..
```

Note: The translator model requires the Tatoeba English-Russian parallel corpus. The GPT model uses the included `input.txt` file.

## Usage

### GPT Model
```python
# Train
jupyter notebook gpt.ipynb
# Run cells sequentially to train and generate text

# Or use the model module directly
from model import InputEmbedding, DecoderBlock
```

### Translator
```python
# Train
jupyter notebook translator.ipynb

# Load trained model
from model import InputEmbedding, EncoderBlock, DecoderBlock, CrossMultiHead
checkpoint_path = 'checkpoints/checkpoint_latest.pt'
load_checkpoint(checkpoint_path, model, optimizer)

# Translate
translated = translate(model, "Hello", temperature=0.8)
```

## Files

- `model.py` - Core transformer components (attention, encoder, decoder blocks)
- `gpt.ipynb` - GPT training and generation
- `translator.ipynb` - Translation model training and inference
- `input.txt` - Shakespeare text for GPT training
- `checkpoints/` - Saved model weights (checkpoint_latest.pt tracked in git)
- `en-ru.txt/` - Parallel corpus data (download separately, see Setup)

## Training

Both models use:
- AdamW optimizer (lr=3e-4)
- Gradient clipping (max_norm=1.0)
- Dropout (p=0.1)
- Teacher forcing for translation

Checkpoints saved every 1000 steps during training.

## Model Components

**Attention Mechanisms:**
- Self-attention with causal masking (decoder)
- Bidirectional self-attention (encoder)
- Cross-attention (encoder-decoder)

**Architecture:**
- Multi-head attention (8 heads)
- Feed-forward networks (4x expansion)
- Layer normalization (pre-norm)
- Residual connections
- Positional embeddings
