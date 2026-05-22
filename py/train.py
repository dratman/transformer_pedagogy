#!/usr/bin/env python3
"""
Training script supporting sentence-per-block, word-per-block, and continuous training.

Modes:
- sentence: One sentence per block with padding (for prose corpora)
- word: One word per block with padding (for word lists)
- continuous: Sliding window over full corpus (traditional nanoGPT style, no padding)

Supports three attention mechanisms:
- softmax (default): Standard transformer attention
- linear: O(N) linear attention
- autocorrelation: FFT-based autocorrelation attention (Autoformer-inspired)

Usage:
  # Sentence mode (default) - for prose corpora
  python train.py --input corpus.txt --mode sentence --block_size 128 --batch_size 64

  # Word mode - for word lists (one word per line)
  python train.py --input wordlist.txt --mode word --block_size 18 --batch_size 64

  # Continuous mode - traditional sliding window (best for long-range dependencies)
  python train.py --input corpus.txt --mode continuous --block_size 256 --batch_size 32

  # With autocorrelation attention
  python train.py --input corpus.txt --autocorrelation_attention
"""

import argparse
import os
import sys
import time
import math
import pickle
import re
import random
import numpy as np
import torch
from datetime import datetime
from contextlib import nullcontext
from model import GPT, GPTConfig
from tokenizer import CharTokenizer, BPETokenizer, load_tokenizer


# Padding token ID - always at vocab_size
PADDING_TOKEN = None  # Will be set after tokenizer is initialized


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train GPT model with one-sentence-per-block or one-word-per-block',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--input', type=str, required=True,
                       help='Input text file (prose corpus for sentence mode, word list for word mode)')

    # Mode selection
    parser.add_argument('--mode', type=str, default='sentence',
                       choices=['sentence', 'word', 'continuous'],
                       help='Training mode: sentence (prose), word (word list), or continuous (sliding window)')

    # Optional arguments with defaults
    parser.add_argument('--output', type=str, default=None,
                       help='Output model file name (default: based on input name)')
    parser.add_argument('--checkpoints_to', type=str, default='pt',
                       help='Directory for checkpoint files')
    parser.add_argument('--n_layer', type=int, default=1,
                       help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=256,
                       help='Embedding dimension')
    parser.add_argument('--block_size', type=int, default=128,
                       help='Maximum sequence length in tokens')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--max_iters', type=int, default=30000,
                       help='Maximum training iterations')
    parser.add_argument('--eval_interval', type=int, default=500,
                       help='How often to evaluate')
    parser.add_argument('--eval_iters', type=int, default=20,
                       help='Number of iterations for evaluation')
    parser.add_argument('--learning_rate', type=float, default=6e-4,
                       help='Learning rate')
    parser.add_argument('--warmup_iters', type=int, default=2000,
                       help='Warmup iterations')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--save_interval', type=int, default=10000,
                       help='Checkpoint save interval')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Loss logging interval')
    parser.add_argument('--sample_interval', type=int, default=500,
                       help='How often to sample from the model')
    parser.add_argument('--sample_max_tokens', type=int, default=128,
                       help='Maximum tokens to generate when sampling')

    # Attention mechanism options (mutually exclusive)
    parser.add_argument('--linear_attention', action='store_true',
                       help='Use linear attention instead of softmax attention')
    parser.add_argument('--autocorrelation_attention', action='store_true',
                       help='Use autocorrelation attention (FFT-based, Autoformer-inspired)')

    # Weight tying option
    parser.add_argument('--untie_weights', action='store_true',
                       help='Do not tie input embeddings to output projection')

    # GELU option
    parser.add_argument('--no_gelu', action='store_true',
                       help='Disable GELU nonlinearity in MLP (makes it purely linear)')

    # Precision option
    parser.add_argument('--precision', type=str, default='float32',
                       choices=['float16', 'float32', 'float64', 'bfloat16'],
                       help='Floating-point precision')

    # Tokenizer options
    parser.add_argument('--tokenizer', type=str, default='bpe',
                       choices=['char', 'bpe'],
                       help='Tokenization scheme: bpe (default) or char')
    parser.add_argument('--vocab_size', type=int, default=8192,
                       help='Vocabulary size for BPE tokenizer (ignored for char)')

    # Gradient accumulation
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                       help='Number of gradient accumulation steps (effective batch = batch_size * grad_accum_steps)')

    # Resume option
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')

    # Sentence mode specific options
    parser.add_argument('--min_sentence_tokens', type=int, default=5,
                       help='Minimum sentence length in tokens (skip shorter) [sentence mode only]')
    parser.add_argument('--max_sentences', type=int, default=None,
                       help='Maximum number of sentences to use (None = all) [sentence mode only]')

    # Hot-reload overrides
    parser.add_argument('--accept_overrides', type=str, default=None,
                       help='Path to a JSON file checked every 1000 iters for '
                            'on-the-fly changes to learning_rate, log_interval, '
                            'eval_interval, sample_interval, save_interval, max_iters')

    # Debug output
    parser.add_argument('--debug', action='store_true',
                       help='Enable verbose debug output')

    args = parser.parse_args()

    # Validate mutually exclusive attention options
    if args.linear_attention and args.autocorrelation_attention:
        parser.error("Cannot use both --linear_attention and --autocorrelation_attention")

    return args


def get_timestamp():
    """Get current timestamp in readable format"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_elapsed(start_time):
    """Format elapsed time since start_time as 'Xd Xh Xm'"""
    elapsed = time.time() - start_time
    days = int(elapsed // 86400)
    hours = int((elapsed % 86400) // 3600)
    minutes = int((elapsed % 3600) // 60)
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


def calculate_epoch(iter_num, batch_size, block_size, train_tokens):
    """Calculate approximate epoch from iteration number"""
    tokens_per_iter = batch_size * block_size
    tokens_processed = iter_num * tokens_per_iter
    epoch = tokens_processed / train_tokens
    return epoch


# =============================================================================
# Sentence Mode Functions
# =============================================================================

def split_into_sentences(text):
    """
    Split text into sentences on sentence-ending punctuation.
    Handles common abbreviations and edge cases.

    Returns list of sentences (strings), each ending with its punctuation.
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Split on sentence boundaries: .!? followed by space and capital letter or end
    # This regex captures the punctuation with the sentence
    # Pattern: split after .!? when followed by space+capital or end of string

    # Simple approach: split on .!? followed by whitespace
    # Then filter out empty strings and very short fragments

    # Use a regex that keeps the delimiter with the sentence
    pattern = r'(?<=[.!?])\s+'
    raw_sentences = re.split(pattern, text)

    sentences = []
    for s in raw_sentences:
        s = s.strip()
        if len(s) > 0:
            sentences.append(s)

    return sentences


def prepare_sentence_data(text, tokenizer, block_size, padding_token,
                          min_tokens=5, max_sentences=None, debug=False):
    """
    Prepare data for one-sentence-per-block training.

    Each sentence is tokenized with a leading newline (BOS) and trailing newline (EOS).
    Input: [BOS, tok1, tok2, ..., tokN]
    Target: [tok1, tok2, ..., tokN, EOS]

    Returns:
        sentence_blocks: list of (input_block, target_block) tuples
        Each block is padded to block_size with padding_token
    """
    sentences = split_into_sentences(text)

    if max_sentences is not None:
        sentences = sentences[:max_sentences]

    print(f"[{get_timestamp()}] Found {len(sentences)} sentences in corpus")

    sentence_blocks = []
    skipped_short = 0
    skipped_long = 0

    # Get newline token for BOS/EOS
    newline_tokens = tokenizer.encode('\n')
    if len(newline_tokens) != 1:
        print(f"Warning: newline encodes to {len(newline_tokens)} tokens, expected 1")
    newline_token = newline_tokens[0]

    print("Tokenizing sentences...")
    for i, sentence in enumerate(sentences):
        # Tokenize sentence
        sentence_tokens = tokenizer.encode(sentence)

        # Skip very short sentences
        if len(sentence_tokens) < min_tokens:
            skipped_short += 1
            continue

        # Add BOS (newline) at start and EOS (newline) at end
        # Full sequence: [newline, sentence_tokens..., newline]
        full_tokens = [newline_token] + sentence_tokens + [newline_token]

        # Skip sentences that are too long (need room for input/target split)
        if len(full_tokens) > block_size:
            skipped_long += 1
            continue

        # Create input and target sequences
        # Input: all but last token [BOS, tok1, tok2, ..., tokN]
        # Target: all but first token [tok1, tok2, ..., tokN, EOS]
        input_tokens = full_tokens[:-1]
        target_tokens = full_tokens[1:]

        # Pad to block_size
        input_padded = input_tokens + [padding_token] * (block_size - len(input_tokens))
        target_padded = target_tokens + [padding_token] * (block_size - len(target_tokens))

        sentence_blocks.append((input_padded, target_padded))

        if debug and i < 3:
            print(f"  Example {i}: '{sentence[:50]}...'")
            print(f"    Tokens: {len(sentence_tokens)}, Full: {len(full_tokens)}")

    print(f"[{get_timestamp()}] Created {len(sentence_blocks)} sentence blocks")
    print(f"[{get_timestamp()}] Skipped {skipped_short} sentences (too short, <{min_tokens} tokens)")
    print(f"[{get_timestamp()}] Skipped {skipped_long} sentences (too long, >{block_size} tokens)")

    # Report token length statistics
    if sentence_blocks:
        lengths = [len([t for t in block[0] if t != padding_token]) for block in sentence_blocks]
        print(f"[{get_timestamp()}] Token lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")

    return sentence_blocks


# =============================================================================
# Word Mode Functions
# =============================================================================

def prepare_word_data(text, tokenizer, block_size, padding_token, debug=False):
    """
    Prepare data for one-word-per-block training.

    Returns:
        word_blocks: list of (input_block, target_block) tuples
        Each block is padded to block_size with padding_token
        Words longer than block_size-1 are skipped
    """
    words = text.strip().split('\n')
    word_blocks = []
    skipped_count = 0

    print(f"[{get_timestamp()}] Preparing word blocks from {len(words)} words...")

    for i, word in enumerate(words):
        # Encode word with leading and trailing newline
        # This teaches: given \n, predict first letter; given last letter, predict \n
        word_with_newlines = '\n' + word + '\n'
        tokens = tokenizer.encode(word_with_newlines)

        # Skip words that are too long (need room for at least one target token)
        if len(tokens) > block_size - 1:
            skipped_count += 1
            continue

        # Create input and target sequences
        # Input: [\n, w, o, r, d] (all but final \n)
        # Target: [w, o, r, d, \n] (all but leading \n)
        input_tokens = tokens[:-1]  # All but last
        target_tokens = tokens[1:]   # All but first

        # Pad to block_size
        input_padded = input_tokens + [padding_token] * (block_size - len(input_tokens))
        target_padded = target_tokens + [padding_token] * (block_size - len(target_tokens))

        word_blocks.append((input_padded, target_padded))

        if debug and i < 3:
            print(f"  Example {i}: '{word}'")
            print(f"    Tokens: {len(tokens)}")

    print(f"[{get_timestamp()}] Created {len(word_blocks)} word blocks")
    print(f"[{get_timestamp()}] Skipped {skipped_count} words (too long for block_size={block_size})")

    return word_blocks


# =============================================================================
# Continuous Mode Functions
# =============================================================================

def prepare_continuous_data(text, tokenizer, block_size, debug=False):
    """
    Prepare data for continuous (sliding window) training.

    This is the traditional nanoGPT approach: tokenize the entire corpus,
    then create training examples using a sliding window.

    No padding is used - every position predicts the next token.

    Returns:
        train_data: torch tensor of all training tokens
        val_data: torch tensor of all validation tokens
    """
    print(f"[{get_timestamp()}] Tokenizing entire corpus ({len(text):,} characters)...")
    tokens = tokenizer.encode(text)
    print(f"[{get_timestamp()}] Tokenization complete: {len(tokens):,} tokens")
    print(f"[{get_timestamp()}] Converting to tensor...")
    tokens = torch.tensor(tokens, dtype=torch.long)
    print(f"[{get_timestamp()}] Tensor ready: {len(tokens):,} tokens")

    return tokens


def get_batch_continuous(data, batch_size, block_size, device='cpu'):
    """
    Generate a batch for continuous mode training.
    Randomly samples starting positions and extracts block_size chunks.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss_continuous(model, train_data, val_data, batch_size, block_size, eval_iters, device):
    """Estimate loss on train and validation sets for continuous mode"""
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_continuous(data, batch_size, block_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# =============================================================================
# Common Functions
# =============================================================================

def get_batch(blocks, batch_size, device='cpu'):
    """
    Generate a batch of blocks for training.
    Each batch contains batch_size randomly selected blocks.
    Works for both sentence and word blocks.
    """
    indices = torch.randint(len(blocks), (batch_size,))

    x_list = []
    y_list = []

    for idx in indices:
        input_block, target_block = blocks[idx]
        x_list.append(input_block)
        y_list.append(target_block)

    x = torch.tensor(x_list, dtype=torch.long, device=device)
    y = torch.tensor(y_list, dtype=torch.long, device=device)

    return x, y


@torch.no_grad()
def estimate_loss(model, train_blocks, val_blocks, batch_size, eval_iters, device):
    """Estimate loss on train and validation sets"""
    out = {}
    model.eval()
    for split, blocks in [('train', train_blocks), ('val', val_blocks)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(blocks, batch_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(iter_num, learning_rate, warmup_iters, max_iters):
    """Learning rate schedule with linear warmup and cosine decay"""
    min_lr = learning_rate / 10
    # Linear warmup
    if iter_num < warmup_iters:
        return learning_rate * iter_num / warmup_iters
    # Cosine decay
    if iter_num > max_iters:
        return min_lr
    decay_ratio = (iter_num - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def check_overrides(args, override_path, lr_override):
    """
    Check for a JSON override file and apply any changes.
    Called every 100 iterations during training.

    Changeable parameters:
        learning_rate  — bypasses cosine schedule, uses fixed value
        log_interval, eval_interval, sample_interval, save_interval
        max_iters      — extend or shorten the run

    Returns:
        lr_override — the fixed learning rate if set, or None to use schedule

    The override file is left in place after reading. To revert learning_rate
    to the cosine schedule, remove the "learning_rate" key from the file
    (or delete the file entirely).
    """
    import json
    if not os.path.exists(override_path):
        return lr_override
    try:
        with open(override_path, 'r') as f:
            overrides = json.load(f)
    except (json.JSONDecodeError, IOError):
        return lr_override

    changed = []
    safe_intervals = ['log_interval', 'eval_interval', 'sample_interval',
                      'save_interval', 'max_iters']
    for key in safe_intervals:
        if key in overrides and getattr(args, key) != overrides[key]:
            old_val = getattr(args, key)
            setattr(args, key, overrides[key])
            changed.append(f"{key}: {old_val} -> {overrides[key]}")

    if 'learning_rate' in overrides:
        new_lr = overrides['learning_rate']
        if lr_override != new_lr:
            changed.append(f"learning_rate: override to fixed {new_lr:.2e}")
            lr_override = new_lr
    else:
        if lr_override is not None:
            changed.append("learning_rate: reverting to cosine schedule")
            lr_override = None

    if changed:
        print(f"[{get_timestamp()}] Overrides applied from {override_path}:")
        for c in changed:
            print(f"  {c}")

    return lr_override


# =============================================================================
# Sampling Functions
# =============================================================================

@torch.no_grad()
def sample_sentences(model, tokenizer, device, num_sentences=5, max_tokens=100,
                     temperature=0.8, top_k=40, padding_token=None, debug=False):
    """
    Generate sample sentences from the model.
    Start with newline (BOS), generate until newline (EOS) or max_tokens.
    """
    model.eval()

    newline_token = tokenizer.encode('\n')[0]
    sentences = []

    for i in range(num_sentences):
        # Start with BOS (newline)
        context = torch.tensor([[newline_token]], dtype=torch.long, device=device)

        generated = model.generate(context, max_new_tokens=max_tokens,
                                   temperature=temperature, top_k=top_k)[0].tolist()

        if debug:
            print(f"DEBUG sentence {i}: Raw generated tokens: {generated[:15]}...")

        # Skip BOS and find EOS (newline)
        generated = generated[1:]  # Skip initial newline
        if newline_token in generated:
            eos_idx = generated.index(newline_token)
            generated = generated[:eos_idx]
            if debug:
                print(f"DEBUG sentence {i}: Found EOS at index {eos_idx}")

        # Filter out padding tokens if any
        if padding_token is not None:
            generated = [t for t in generated if t != padding_token]

        sentence = tokenizer.decode(generated)
        sentences.append(sentence)

        if debug:
            print(f"DEBUG sentence {i}: Decoded: '{sentence[:50]}...'")

    model.train()
    return sentences


@torch.no_grad()
def sample_words(model, tokenizer, device, num_words=20, max_tokens=20,
                 temperature=0.8, top_k=40, padding_token=None, debug=False):
    """
    Generate sample words from the model.
    Start with newline (BOS), generate until newline (EOS) or max_tokens.
    """
    model.eval()

    newline_token = tokenizer.encode('\n')[0]
    words = []

    if debug:
        print(f"DEBUG: Newline token ID: {newline_token}")

    for i in range(num_words):
        # Start with BOS (newline)
        context = torch.tensor([[newline_token]], dtype=torch.long, device=device)

        generated = model.generate(context, max_new_tokens=max_tokens,
                                   temperature=temperature, top_k=top_k)[0].tolist()

        if debug:
            print(f"DEBUG word {i}: Raw generated tokens: {generated[:10]}")

        # Skip BOS and find EOS (newline)
        generated = generated[1:]  # Skip initial newline
        if newline_token in generated:
            newline_idx = generated.index(newline_token)
            generated = generated[:newline_idx + 1]  # Include the newline for word mode
            if debug:
                print(f"DEBUG word {i}: Found newline at index {newline_idx}")

        # Filter out padding tokens
        if padding_token is not None:
            generated = [t for t in generated if t != padding_token]

        if debug:
            print(f"DEBUG word {i}: Filtered tokens: {generated}")

        word = tokenizer.decode(generated)
        words.append(word)

        if debug:
            print(f"DEBUG word {i}: Decoded word: '{word}'")

    model.train()
    return words


@torch.no_grad()
def sample_continuous(model, tokenizer, device, num_samples=3, max_tokens=200,
                      temperature=0.8, top_k=40, debug=False):
    """
    Generate samples from the model in continuous mode.
    Starts with a random character and generates max_tokens.
    """
    model.eval()

    samples = []

    for i in range(num_samples):
        # Start with a newline or space
        start_tokens = tokenizer.encode('\n')
        context = torch.tensor([start_tokens], dtype=torch.long, device=device)

        generated = model.generate(context, max_new_tokens=max_tokens,
                                   temperature=temperature, top_k=top_k)[0].tolist()

        if debug:
            print(f"DEBUG continuous {i}: Raw generated tokens: {generated[:20]}...")

        # Decode the full sequence (skip the initial newline)
        text = tokenizer.decode(generated[1:])
        samples.append(text)

        if debug:
            print(f"DEBUG continuous {i}: Decoded: '{text[:100]}...'")

    model.train()
    return samples


def main():
    global PADDING_TOKEN

    args = parse_args()

    # Validate tokenizer arguments
    if not args.resume and args.tokenizer == 'char' and args.vocab_size != 8192:
        print("Warning: --vocab_size is ignored with --tokenizer char")

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"[{get_timestamp()}] Using device: {device}")

    # Set up precision
    precision_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
        'bfloat16': torch.bfloat16,
    }
    ptdtype = precision_map[args.precision]

    # Check device compatibility
    original_precision = args.precision
    if device == 'mps':
        if args.precision == 'float64':
            print(f"[{get_timestamp()}] WARNING: MPS does not support float64. Falling back to CPU.")
            device = 'cpu'
        elif args.precision == 'bfloat16':
            # bfloat16 is supported on MPS with recent PyTorch versions
            pass
    elif device == 'cuda':
        if args.precision == 'bfloat16' and not torch.cuda.is_bf16_supported():
            print(f"[{get_timestamp()}] WARNING: CUDA device does not support bfloat16. Falling back to float16.")
            args.precision = 'float16'
            ptdtype = torch.float16

    print(f"[{get_timestamp()}] Using precision: {args.precision} (requested: {original_precision})")
    print(f"[{get_timestamp()}] Training mode: {args.mode}")

    # Determine attention type for naming
    if args.autocorrelation_attention:
        attn_suffix = '_autocorr'
        attn_type = 'autocorrelation'
    elif args.linear_attention:
        attn_suffix = '_linear'
        attn_type = 'linear'
    else:
        attn_suffix = ''
        attn_type = 'softmax'

    # Determine output filename
    if args.output is None:
        input_base = os.path.splitext(os.path.basename(args.input))[0]
        suffix = f'_{args.mode}'  # Mark training mode
        if args.tokenizer == 'bpe':
            suffix += '_bpe'
        suffix += attn_suffix
        if args.untie_weights:
            suffix += '_untied'
        args.output = f'{args.checkpoints_to}/{input_base}{suffix}.pt'

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    # Read input text
    print(f"[{get_timestamp()}] Reading input from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"[{get_timestamp()}] Read {len(text):,} characters")

    # Check if resuming
    resume_iter = 0
    best_val_loss = float('inf')

    if args.resume:
        print(f"[{get_timestamp()}] Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        resume_iter = checkpoint.get('iter_num', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        # Load tokenizer from checkpoint metadata
        # Handle _iter{N} and _final checkpoint names
        ckpt_base = args.resume.replace('.pt', '')
        ckpt_base = re.sub(r'_iter\d+$', '', ckpt_base)
        ckpt_base = re.sub(r'_final$', '', ckpt_base)
        meta_path = ckpt_base + '_meta.pkl'
        if not os.path.exists(meta_path):
            print(f"Error: Metadata file '{meta_path}' not found")
            sys.exit(1)
        print(f"[{get_timestamp()}] Loading tokenizer from {meta_path}")
        tokenizer = load_tokenizer(meta_path)
        vocab_size = tokenizer.vocab_size
    else:
        # Create tokenizer
        print(f"[{get_timestamp()}] Training {args.tokenizer} tokenizer...")
        if args.tokenizer == 'char':
            tokenizer = CharTokenizer()
            tokenizer.train(text)
            vocab_size = tokenizer.vocab_size
        else:  # bpe
            tokenizer = BPETokenizer(vocab_size=args.vocab_size)
            tokenizer.train(text)
            vocab_size = tokenizer.vocab_size

    # Set padding token (not used in continuous mode)
    PADDING_TOKEN = vocab_size
    if args.mode != 'continuous':
        print(f"[{get_timestamp()}] Padding token ID: {PADDING_TOKEN}")

    # Token-cache status (used for banner). Only set in continuous mode.
    used_token_cache = False
    token_cache_path = None

    # Prepare data based on mode
    if args.mode == 'continuous':
        # Continuous mode: tokenize entire corpus, split into train/val.
        # Cache the tokenized stream next to the checkpoint so --resume
        # skips the BPE encode cost (minutes on a multi-GB corpus). Cache
        # is invalidated when a fresh training starts (we overwrite it on
        # the non-resume path) and also if the corpus file is newer than
        # the cache file. On resume with a valid cache, we never call
        # prepare_continuous_data, so text is unused but still in memory
        # briefly; it gets freed by the `del text` later in main().
        token_cache_path = os.path.splitext(args.output)[0] + '_tokens.pt'
        if args.resume and os.path.exists(token_cache_path) \
                and os.path.getmtime(token_cache_path) >= os.path.getmtime(args.input):
            print(f"[{get_timestamp()}] Loading tokenized corpus from cache: {token_cache_path}")
            all_tokens = torch.load(token_cache_path, map_location='cpu')
            print(f"[{get_timestamp()}] Loaded {len(all_tokens):,} tokens from cache")
            used_token_cache = True
        else:
            all_tokens = prepare_continuous_data(text, tokenizer, args.block_size, debug=args.debug)
            print(f"[{get_timestamp()}] Saving tokenized corpus cache: {token_cache_path}")
            torch.save(all_tokens, token_cache_path)

        # Split into train and validation
        print(f"[{get_timestamp()}] Splitting into train and validation sets...")
        n = len(all_tokens)
        n_val = int(n * args.val_split)
        n_train = n - n_val

        train_data = all_tokens[:n_train]
        val_data = all_tokens[n_train:]
        print(f"[{get_timestamp()}] Split complete.")

        train_tokens = len(train_data)

        print(f"[{get_timestamp()}] Training tokens: {len(train_data):,}")
        print(f"[{get_timestamp()}] Validation tokens: {len(val_data):,}")

        # Placeholders for compatibility
        train_blocks = None
        val_blocks = None

    else:
        # Block-based modes (sentence, word)
        if args.mode == 'sentence':
            data_blocks = prepare_sentence_data(
                text, tokenizer, args.block_size, PADDING_TOKEN,
                min_tokens=args.min_sentence_tokens,
                max_sentences=args.max_sentences,
                debug=args.debug
            )
        else:  # word mode
            data_blocks = prepare_word_data(
                text, tokenizer, args.block_size, PADDING_TOKEN,
                debug=args.debug
            )

        if len(data_blocks) == 0:
            print("Error: No valid data blocks found. Check block_size and input data.")
            sys.exit(1)

        # Split into train and validation
        n = len(data_blocks)
        n_val = int(n * args.val_split)
        n_train = n - n_val

        # Shuffle data blocks
        random.shuffle(data_blocks)

        train_blocks = data_blocks[:n_train]
        val_blocks = data_blocks[n_train:]

        # Calculate total training tokens for epoch calculation
        train_tokens = len(train_blocks) * args.block_size

        print(f"[{get_timestamp()}] Training blocks: {len(train_blocks):,} ({train_tokens:,} tokens)")
        print(f"[{get_timestamp()}] Validation blocks: {len(val_blocks)}")

        # Placeholders for compatibility
        train_data = None
        val_data = None

    # Free the corpus string — it is no longer referenced for the rest
    # of main(), and on a multi-day run it would otherwise sit in
    # main()'s scope holding ~1 GB+ of Python heap that the compressor
    # and swap then have to manage.
    del text

    # Save metadata
    if not args.resume:
        meta_path = os.path.splitext(args.output)[0] + '_meta.pkl'
        print(f"[{get_timestamp()}] Saving tokenizer to {meta_path}")
        tokenizer.save(meta_path)

    # Initialize or load model
    if args.resume:
        print(f"[{get_timestamp()}] Loading model from checkpoint...")
        model_args = checkpoint['model_args']
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
    else:
        # Model vocab size: +1 for padding token in block modes, no padding for continuous
        if args.mode == 'continuous':
            model_vocab_size = vocab_size
            print(f"[{get_timestamp()}] Initializing new model...")
            print(f"  Vocabulary size: {vocab_size}")
        else:
            model_vocab_size = vocab_size + 1
            print(f"[{get_timestamp()}] Initializing new model...")
            print(f"  Vocabulary size: {vocab_size} (+ 1 padding token = {model_vocab_size})")

        model_args = dict(
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            block_size=args.block_size,
            bias=True,
            vocab_size=model_vocab_size,
            dropout=args.dropout,
            use_linear_attention=args.linear_attention,
            use_autocorrelation_attention=args.autocorrelation_attention,
            tie_weights=not args.untie_weights,
            no_gelu=args.no_gelu,
        )

        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        model.to(device)

    # Modify model to use padding token in loss calculation (only for block modes)
    if args.mode != 'continuous':
        def forward_with_padding(self, idx, targets=None):
            """Modified forward that ignores padding tokens in loss"""
            device = idx.device
            b, t = idx.size()
            assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            pos = torch.arange(0, t, dtype=torch.long, device=device)

            # Forward the GPT model itself
            tok_emb = self.transformer.wte(idx)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
            for block in self.transformer.h:
                x = block(x)
            x = self.transformer.ln_f(x)

            if targets is not None:
                logits = self.lm_head(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=PADDING_TOKEN
                )
            else:
                logits = self.lm_head(x[:, [-1], :])
                loss = None

            return logits, loss

        # Replace model's forward method
        import types
        model.forward = types.MethodType(forward_with_padding, model)

    # Set up optimizer
    if args.resume:
        optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=args.learning_rate,
                                              betas=(0.9, 0.95), device_type=device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"[{get_timestamp()}] Loaded optimizer state from checkpoint")
    else:
        optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=args.learning_rate,
                                              betas=(0.9, 0.95), device_type=device)

    # Set up mixed precision training
    ctx = nullcontext()
    scaler = None
    if ptdtype != torch.float32:
        if device == 'cuda':
            ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
            if ptdtype == torch.float16:
                scaler = torch.amp.GradScaler('cuda')
            print(f"[{get_timestamp()}] Using automatic mixed precision (AMP) with {args.precision}")
        elif device == 'mps':
            ctx = torch.amp.autocast(device_type='mps', dtype=ptdtype)
            print(f"[{get_timestamp()}] Using {args.precision} precision on MPS")

    # Training loop
    print(f"\n[{get_timestamp()}] Starting training...")
    print(f"Command line: {' '.join(sys.argv)}")
    print()
    print(f"--- Mode and tokenizer ---")
    print(f"  mode:               {args.mode}")
    print(f"  tokenizer:          {tokenizer.tokenizer_type}")
    print(f"  vocab_size:         {vocab_size}" +
          (f" (+ 1 padding = {vocab_size + 1})" if args.mode != 'continuous' else ''))
    if args.mode != 'continuous':
        print(f"  padding token ID:   {PADDING_TOKEN}")
    print(f"  attention type:     {attn_type}")
    print(f"  precision:          {args.precision}")
    print(f"--- Architecture ---")
    print(f"  n_layer:            {args.n_layer}")
    print(f"  n_head:             {args.n_head}")
    print(f"  n_embd:             {args.n_embd}")
    print(f"  block_size:         {args.block_size}")
    print(f"  dropout:            {args.dropout}")
    print(f"  bias:               True  (hardcoded; LayerNorm + Linear bias)")
    print(f"  tie_weights:        {not args.untie_weights}")
    print(f"  no_gelu:            {args.no_gelu}")
    print(f"--- Training ---")
    print(f"  batch_size:         {args.batch_size}")
    if args.grad_accum_steps > 1:
        print(f"  grad_accum_steps:   {args.grad_accum_steps} "
              f"(effective batch = {args.batch_size * args.grad_accum_steps})")
    else:
        print(f"  grad_accum_steps:   1")
    print(f"  max_iters:          {args.max_iters}")
    print(f"  learning_rate:      {args.learning_rate:.2e} (target after warmup)")
    print(f"  min_lr (cosine):    {args.learning_rate / 10:.2e}")
    print(f"  warmup_iters:       {args.warmup_iters}")
    print(f"  weight_decay:       0.1 (decay tensors), 0.0 (no-decay tensors)  [hardcoded]")
    print(f"  betas:              (0.9, 0.95)  [hardcoded]")
    print(f"  grad_clip:          1.0  [hardcoded]")
    print(f"--- Data ---")
    print(f"  input:              {args.input}")
    print(f"  val_split:          {args.val_split}")
    if args.mode == 'continuous':
        print(f"  token_cache:        {token_cache_path} "
              f"({'loaded' if used_token_cache else 'rebuilt'})")
    if args.mode == 'sentence':
        print(f"  min_sentence_tokens:{args.min_sentence_tokens}")
        print(f"  max_sentences:      {args.max_sentences}")
    print(f"--- Logging ---")
    print(f"  log_interval:       {args.log_interval}")
    print(f"  eval_interval:      {args.eval_interval}")
    print(f"  eval_iters:         {args.eval_iters}")
    print(f"  save_interval:      {args.save_interval}")
    print(f"  sample_interval:    {args.sample_interval}")
    print(f"  sample_max_tokens:  {args.sample_max_tokens}")
    # Rolling-save path: overwritten via atomic rename after every eval, so
    # it's always the most-recent durable state. Use this as --resume target
    # after an involuntary break.
    rolling_path = os.path.splitext(args.output)[0] + '_rolling.pt'

    print(f"--- Output ---")
    print(f"  output:             {args.output}")
    print(f"  checkpoints_to:     {args.checkpoints_to}")
    print(f"  rolling_save:       {rolling_path} (written every eval_interval)")
    if args.accept_overrides:
        print(f"  accept_overrides:   {args.accept_overrides}")
    if args.resume:
        print(f"--- Resume ---")
        print(f"  resume:             {args.resume}")
        print(f"  resume_iter:        {resume_iter}")
    print("=" * 60)

    iter_num = resume_iter
    running_mfu = -1.0

    # Get first batch
    if args.mode == 'continuous':
        X, Y = get_batch_continuous(train_data, args.batch_size, args.block_size, device)
    else:
        X, Y = get_batch(train_blocks, args.batch_size, device)

    # Initialize timing
    t0 = time.time()
    speed_t0 = time.time()
    training_start_time = time.time()  # For elapsed time display
    local_iter_num = 0
    raw_model = model

    # Hot-reload overrides
    lr_override = None  # None = use cosine schedule; float = use fixed LR
    if args.accept_overrides:
        print(f"[{get_timestamp()}] Will check for overrides at: {args.accept_overrides}")

    while iter_num < args.max_iters:
        # Check for override file every 1000 iterations
        if args.accept_overrides and iter_num % 1000 == 0:
            lr_override = check_overrides(args, args.accept_overrides, lr_override)

        # Set learning rate (override bypasses schedule)
        if lr_override is not None:
            lr = lr_override
        else:
            lr = get_lr(iter_num, args.learning_rate, args.warmup_iters, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluate
        if iter_num % args.eval_interval == 0:
            if args.mode == 'continuous':
                losses = estimate_loss_continuous(model, train_data, val_data,
                                                  args.batch_size, args.block_size, args.eval_iters, device)
            else:
                losses = estimate_loss(model, train_blocks, val_blocks,
                                       args.batch_size, args.eval_iters, device)
            epoch = calculate_epoch(iter_num, args.batch_size * args.grad_accum_steps, args.block_size, train_tokens)
            elapsed = format_elapsed(training_start_time)
            print(f"[{get_timestamp()}] [{elapsed}] Step {iter_num:5d} | Epoch {epoch:6.2f} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | lr {lr:.2e}")

            # Check for NaN
            if math.isnan(losses['train']) or math.isnan(losses['val']):
                print("\n" + "!" * 60)
                print("WARNING: NaN detected in loss! Stopping training.")
                print("!" * 60)
                break

            # Check for loss explosion
            if losses['train'] > 100 or losses['val'] > 100:
                print("WARNING: Loss explosion detected! Consider reducing learning rate.")

            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': vars(args),
                        'precision': args.precision,
                        'tokenizer_type': tokenizer.tokenizer_type,
                        'padding_token': PADDING_TOKEN,
                        'mode': args.mode,
                    }
                    torch.save(checkpoint, args.output)
                    print(f"  Saved checkpoint to {args.output} (best val loss: {best_val_loss:.4f})")

            # Save rolling checkpoint via atomic rename. Overwrites the prior
            # rolling save every eval; this is the canonical --resume target
            # for crash recovery (loses at most eval_interval iters of work).
            if iter_num > 0:
                rolling_checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': vars(args),
                    'precision': args.precision,
                    'tokenizer_type': tokenizer.tokenizer_type,
                    'padding_token': PADDING_TOKEN,
                    'mode': args.mode,
                    'val_loss': losses['val'],
                }
                tmp_path = rolling_path + '.tmp'
                torch.save(rolling_checkpoint, tmp_path)
                os.rename(tmp_path, rolling_path)
                print(f"  Saved rolling checkpoint to {rolling_path} (iter {iter_num}, val {losses['val']:.4f})")

        # Save checkpoint at regular intervals
        if iter_num > 0 and iter_num % args.save_interval == 0:
            checkpoint_file = os.path.splitext(args.output)[0] + f'_iter{iter_num}.pt'
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'val_loss': losses.get('val', None) if 'losses' in locals() else None,
                'config': vars(args),
                'tokenizer_type': tokenizer.tokenizer_type,
                'padding_token': PADDING_TOKEN,
                'mode': args.mode,
            }
            torch.save(checkpoint, checkpoint_file)
            epoch = calculate_epoch(iter_num, args.batch_size * args.grad_accum_steps, args.block_size, train_tokens)
            elapsed = format_elapsed(training_start_time)
            print(f"[{get_timestamp()}] [{elapsed}] Saved periodic checkpoint to {checkpoint_file} (iter {iter_num}, epoch {epoch:.2f})")

        # Sample from the model periodically
        if iter_num % args.sample_interval == 0 and iter_num > 0:
            epoch = calculate_epoch(iter_num, args.batch_size * args.grad_accum_steps, args.block_size, train_tokens)
            elapsed = format_elapsed(training_start_time)
            print(f"\n[{get_timestamp()}] [{elapsed}] Sampling at iter {iter_num} (epoch {epoch:.2f}):")
            if args.mode == 'sentence':
                samples = sample_sentences(model, tokenizer, device,
                                          num_sentences=3, max_tokens=args.sample_max_tokens,
                                          temperature=0.8, top_k=40,
                                          padding_token=PADDING_TOKEN,
                                          debug=args.debug)
                for i, sent in enumerate(samples):
                    print(f"  {i+1}. {sent}")
            elif args.mode == 'continuous':
                samples = sample_continuous(model, tokenizer, device,
                                           num_samples=3, max_tokens=args.sample_max_tokens,
                                           temperature=0.8, top_k=40,
                                           debug=args.debug)
                for i, text in enumerate(samples):
                    # Show first 150 chars on one line
                    display = text[:150].replace('\n', ' ')
                    print(f"  {i+1}. {display}")
            else:  # word mode
                words = sample_words(model, tokenizer, device,
                                    num_words=20, max_tokens=20,
                                    temperature=0.8, top_k=40,
                                    padding_token=PADDING_TOKEN,
                                    debug=args.debug)
                sample = ''.join(words)
                print(f"\nSample (first 200 chars): {sample[:200]}")
            print()

        # Forward backward update with gradient accumulation
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(args.grad_accum_steps):
            with ctx:
                logits, micro_loss = model(X, Y)
                # Scale loss by accumulation steps so the average is correct
                micro_loss = micro_loss / args.grad_accum_steps

            # Check for NaN
            if torch.isnan(micro_loss):
                print(f"\n" + "!" * 60)
                print(f"WARNING: NaN detected at iteration {iter_num}! Stopping training.")
                print("!" * 60)
                break

            if scaler is not None:
                scaler.scale(micro_loss).backward()
            else:
                micro_loss.backward()

            # Get next batch for next micro-step (except on last step)
            if micro_step < args.grad_accum_steps - 1:
                if args.mode == 'continuous':
                    X, Y = get_batch_continuous(train_data, args.batch_size, args.block_size, device)
                else:
                    X, Y = get_batch(train_blocks, args.batch_size, device)

        # Check if we broke out due to NaN
        if torch.isnan(micro_loss * args.grad_accum_steps):
            break

        # Unscaled loss for logging (sum of micro_losses = original scale)
        loss = micro_loss * args.grad_accum_steps

        if scaler is not None:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Timing
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        # Speed monitoring
        if iter_num % 100 == 0 and iter_num > 0:
            speed_dt = time.time() - speed_t0
            epoch = calculate_epoch(iter_num, args.batch_size * args.grad_accum_steps, args.block_size, train_tokens)
            elapsed = format_elapsed(training_start_time)
            print(f"[{get_timestamp()}] [{elapsed}] Speed at iter {iter_num} (epoch {epoch:.2f}): {100/speed_dt:.2f} iters/sec")
            speed_t0 = time.time()

        # Loss logging
        if iter_num % args.log_interval == 0:
            lossf = loss.item()
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(args.batch_size * 1, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

        iter_num += 1
        local_iter_num += 1

        # Get next batch
        if args.mode == 'continuous':
            X, Y = get_batch_continuous(train_data, args.batch_size, args.block_size, device)
        else:
            X, Y = get_batch(train_blocks, args.batch_size, device)

    # Final save
    final_epoch = calculate_epoch(iter_num, args.batch_size * args.grad_accum_steps, args.block_size, train_tokens)
    elapsed = format_elapsed(training_start_time)
    print(f"\n[{get_timestamp()}] [{elapsed}] Training complete!")
    print(f"Mode: {args.mode}")
    print(f"Tokenizer: {tokenizer.tokenizer_type}")
    print(f"Attention type: {attn_type}")
    print(f"Final iteration: {iter_num}, Final epoch: {final_epoch:.2f}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Save final checkpoint
    if iter_num > 0:
        final_checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'config': vars(args),
            'tokenizer_type': tokenizer.tokenizer_type,
            'padding_token': PADDING_TOKEN,
            'mode': args.mode,
        }
        final_file = os.path.splitext(args.output)[0] + '_final.pt'
        print(f"[{get_timestamp()}] Saving final model to {final_file}")
        torch.save(final_checkpoint, final_file)

    print(f"Final model saved to: {args.output}")
    print("=" * 60)

    # Final sample
    print(f"\n[{get_timestamp()}] Final sample from trained model:")
    if args.mode == 'sentence':
        samples = sample_sentences(model, tokenizer, device,
                                  num_sentences=5, max_tokens=args.sample_max_tokens,
                                  temperature=0.8, top_k=40,
                                  padding_token=PADDING_TOKEN,
                                  debug=args.debug)
        for i, sent in enumerate(samples):
            print(f"  {i+1}. {sent}")
    elif args.mode == 'continuous':
        samples = sample_continuous(model, tokenizer, device,
                                   num_samples=5, max_tokens=args.sample_max_tokens,
                                   temperature=0.8, top_k=40,
                                   debug=args.debug)
        for i, text in enumerate(samples):
            display = text[:200].replace('\n', ' ')
            print(f"  {i+1}. {display}")
    else:  # word mode
        words = sample_words(model, tokenizer, device,
                            num_words=20, max_tokens=20,
                            temperature=0.8, top_k=40,
                            padding_token=PADDING_TOKEN,
                            debug=args.debug)
        sample = ''.join(words)
        print(sample)


if __name__ == "__main__":
    main()
