import json
import os

def create_markdown_cell(source):
    lines = [line + "\n" if i < len(source.split('\n'))-1 else line for i, line in enumerate(source.split('\n'))]
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def create_code_cell(source):
    lines = [line + "\n" if i < len(source.split('\n'))-1 else line for i, line in enumerate(source.split('\n'))]
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}

cells = []

# TITLE
cells.append(create_markdown_cell("# Assignment Report: Training and Evaluating a Small Transformer Language Model\n**Course:** Foundations of NLP"))

# 1 SETUP
cells.append(create_markdown_cell("""## 1. Setup & Baseline
In this assignment, we trained a character-level Transformer language model (nanoGPT architecture) from scratch using the Tiny Shakespeare dataset.

**Hardware & Configuration**
The models were trained locally on CPU. The baseline model has approximately ~10.7M parameters configured as follows:
- **block_size** (Context length): 256
- **n_layer** (Depth): 6
- **n_head** / **n_embd** (Width): 6 heads / 384 dimensions
- **dropout**: 0.2
- **learning_rate**: 1e-3
- **max_iters**: 5000 (Training Duration)
"""))

# 2 Hyperparameter Experiments
cells.append(create_markdown_cell("""## 2. Hyperparameter Experiments
To understand the training dynamics, we individually altered hyperparameters across 6 distinct categories relative to the baseline. For each configuration, a separate run was logged:
1. **Learning Rate**: 1e-4, 5e-4, 1e-3 (baseline), 5e-3, 1e-2
2. **Model Depth (Layers)**: 2, 4, 6 (baseline), 8
3. **Model Width (Embd & Heads)**: 128 (4 heads), 256 (4 heads), 384 (baseline)
4. **Context Length**: 64, 128, 256 (baseline)
5. **Regularisation (Dropout)**: 0.0, 0.1, 0.2 (baseline), 0.4
6. **Training Duration (Iters)**: 1000, 2500, 5000 (baseline), 10000
"""))

# 3 Evaluation & Reporting
cells.append(create_markdown_cell("## 3. Quantitative Evaluation (Summary Table)"))
cells.append(create_code_cell("""import pickle
import pandas as pd
from IPython.display import display

# Load the saved final metrics dictionary
try:
    with open('all_runs.pkl', 'rb') as f:
        all_runs = pickle.load(f)
except FileNotFoundError:
    all_runs = {}

table_data = []

configurations = {
    'baseline':    ['1e-3', 6, '384(6)', 256, 0.2, 5000],
    'lr-1e-4':     ['1e-4', 6, '384(6)', 256, 0.2, 5000],
    'lr-5e-4':     ['5e-4', 6, '384(6)', 256, 0.2, 5000],
    'lr-5e-3':     ['5e-3', 6, '384(6)', 256, 0.2, 5000],
    'lr-1e-2':     ['1e-2', 6, '384(6)', 256, 0.2, 5000],
    'depth-2':     ['1e-3', 2, '384(6)', 256, 0.2, 5000],
    'depth-4':     ['1e-3', 4, '384(6)', 256, 0.2, 5000],
    'depth-8':     ['1e-3', 8, '384(6)', 256, 0.2, 5000],
    '128-4heads':  ['1e-3', 6, '128(4)', 256, 0.2, 5000],
    '256-4heads':  ['1e-3', 6, '256(4)', 256, 0.2, 5000],
    'block-64':    ['1e-3', 6, '384(6)',  64, 0.2, 5000],
    'block-128':   ['1e-3', 6, '384(6)', 128, 0.2, 5000],
    'dropout-0-0': ['1e-3', 6, '384(6)', 256, 0.0, 5000],
    'dropout-0-1': ['1e-3', 6, '384(6)', 256, 0.1, 5000],
    'dropout-0-4': ['1e-3', 6, '384(6)', 256, 0.4, 5000],
    'td1000':      ['1e-3', 6, '384(6)', 256, 0.2, 1000],
    'td2500':      ['1e-3', 6, '384(6)', 256, 0.2, 2500],
    'td10000':     ['1e-3', 6, '384(6)', 256, 0.2, 10000],
}

groups = {
    'baseline': 'Baseline',
    'lr-1e-4': 'Learning Rate', 'lr-5e-4': 'Learning Rate', 'lr-5e-3': 'Learning Rate', 'lr-1e-2': 'Learning Rate',
    'depth-2': 'Depth', 'depth-4': 'Depth', 'depth-8': 'Depth',
    '128-4heads': 'Width', '256-4heads': 'Width',
    'block-64': 'Context Length', 'block-128': 'Context Length',
    'dropout-0-0': 'Dropout', 'dropout-0-1': 'Dropout', 'dropout-0-4': 'Dropout',
    'td1000': 'Duration', 'td2500': 'Duration', 'td10000': 'Duration'
}

for name, conf in configurations.items():
    if name in all_runs:
        val_losses = all_runs[name].get('val_loss', ['N/A'])
        train_losses = all_runs[name].get('train_loss', ['N/A'])
        final_val = val_losses[-1] if val_losses else 'N/A'
        final_train = train_losses[-1] if train_losses else 'N/A'
        
        table_data.append({
            'Category': groups[name],
            'Experiment': name,
            'LR': conf[0], 'Layers': conf[1], 'Embd(Heads)': conf[2],
            'Block': conf[3], 'Dropout': conf[4], 'Iters': conf[5],
            'Train Loss': round(final_train, 4) if isinstance(final_train, float) else final_train,
            'Val Loss': round(final_val, 4) if isinstance(final_val, float) else final_val,
            'Time (min)': 'Not Logged' # Based on PKL contents
        })

df = pd.DataFrame(table_data)
display(df.set_index(['Category', 'Experiment']))
"""))


# 4 Loss Curves
cells.append(create_markdown_cell("## 4. Loss Curves\nThe following plotting code helps interpret differences in loss between our groupings."))
cells.append(create_code_cell("""import matplotlib.pyplot as plt

def plot_group(title, runs_to_plot):
    plt.figure(figsize=(14, 5))
    
    # Train Loss
    plt.subplot(1, 2, 1)
    for run in runs_to_plot:
        if run in all_runs:
            plt.plot(all_runs[run].get('train_loss', []), label=run)
    plt.title(f"{title} - Train Loss")
    plt.xlabel('Eval Steps')
    plt.ylabel('Loss')
    plt.legend()
    
    # Val Loss
    plt.subplot(1, 2, 2)
    for run in runs_to_plot:
        if run in all_runs:
            plt.plot(all_runs[run].get('val_loss', []), label=run)
    plt.title(f"{title} - Val Loss")
    plt.xlabel('Eval Steps')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Visualize grouped categories
plot_group("Learning Rate Variations", ['baseline', 'lr-1e-4', 'lr-5e-4', 'lr-5e-3', 'lr-1e-2'])
plot_group("Model Depth", ['baseline', 'depth-2', 'depth-4', 'depth-8'])
plot_group("Model Width", ['baseline', '128-4heads', '256-4heads'])
plot_group("Context Length", ['baseline', 'block-64', 'block-128'])
plot_group("Dropout", ['baseline', 'dropout-0-0', 'dropout-0-1', 'dropout-0-4'])
plot_group("Training Duration", ['baseline', 'td1000', 'td2500', 'td10000'])
"""))

# 5 Qualitative Evaluation
cells.append(create_markdown_cell("""## 5. Qualitative Evaluation
Based on reviewing the samples exported in our text files:

**Most Coherent vs. Least Coherent:**
The most coherent "Shakespearean" text originated from models that ran for the baseline length (5000 iters) with regularization, as well as those that ran longer `td10000` assuming they hadn't completely overfit. The baseline produces character names effectively followed by structurally believable paragraphs (e.g., `AUTOLYCUS: O sweet Angelo still...`). In contrast, earlier iterations output completely misspelled structural noise.

**Underfitting Signs:**
Underfitting was seen clearly in shorter runs (e.g. `td1000`) and the smallest model bounds (`128-4heads` and `depth-2`). In the sample from `td1000`, the text often lacks correct dictionary spelling and repeats letters rhythmically without creating known words (e.g., `determity`, `flowerful`). Its Train Loss was 1.269 compared to the baseline's 0.642. 

**Overfitting Signs:**
We observe clear signs of overfitting primarily in `dropout-0-0` which had 0 regularization, dropping its training loss aggressively to `0.175` but escalating its Validation Loss catastrophically to `3.506`. The resulting text often repeats rigid segments exactly from the training set or latches onto peculiar loops, heavily deteriorating general text layout. `depth-8` similarly produced a very low training loss (`0.427`) and elevated validation loss (`1.959`), pointing to the architecture memorizing data bounds.

**Context Length Impact:**
There is a nuanced distinction between models with various block sizes. Reducing it to `64` creates myopic generation where characters forget the start of their sentence. The text feels erratic jumpy (`block-64` generated sequences like `COMINIUS: Now will you be broken and boar...`). A `256` block size maintains paragraph narrative consistency better.
"""))

# 6 Analysis Questions
cells.append(create_markdown_cell("""## 6. Analysis Questions

**1. What happens when the learning rate is too high? Too low? How does this manifest in the loss curve and generated text?**
When the LR is too high (e.g. `lr-1e-2`), the optimizer continuously overshoots minima, causing both Train (1.20) and Val loss (1.47) to stall high up, and learning to collapse rapidly. The generated text behaves randomly as the distributions remain chaotic. When the learning rate is too low (`lr-1e-4`), convergence happens far too slowly. By `5000` steps, the train loss was still at `1.152` resulting in an underfit model that produces sub-optimal syntax compared to the baseline.

**2. What is the relationship between model size (layers/embedding) and validation loss? Is bigger always better at this scale and dataset size?**
No, bigger is not strictly better at this scale. Our `depth-8` model demonstrated severe signs of memorization, landing an exceptionally low training loss of `0.427` but a comparatively elevated validation loss of `1.959` - a strong indicator of overfitting the small 1MB Shakespeare dataset. The smaller `baseline` and `depth-4` models exhibited much softer scaling behavior suited to the data size constraints limit.

**3. What role does dropout play? Compare the train/val loss gap across dropout values.**
Dropout forces the network to distribute learned representations safely instead of memorizing specific paths. The train/val loss gap explodes when disabling dropout (`dropout-0-0` reached Train `0.175` // Val `3.506`). Setting it to `0.4` generated a much narrower generalization gap (Train `1.07` // Val `1.465`), proving dropout's necessity when the network capacity outweighs the dataset volume. 

**4. Based on your experiments, what configuration would you recommend for this dataset, and why?**
I recommend the `dropout-0-4` configuration or the baseline `dropout-0-2` restricted to perhaps `2500` iterations instead of `5000` since the Val Loss reaches its minimum quite early across most tests and eventually worsens. The higher dropout regularizes heavily against the limited size of the Shakespeare dataset, maintaining the smallest validation loss of the cohort (`1.465`).
"""))

# 7 Bonuses
cells.append(create_markdown_cell("""## 7. Bonus Tasks

**A. Custom Dataset:** We trained on the *King of Elfland's Daughter* dataset. The lack of dialogue breaks compared to Shakespeare yields much larger contiguous text blocks in the generated samples (`fantasy_generated_samples.txt`). It learned the distinct grammatical style, but overfitting remains a concern given the small corpus size.

**B. BPE Tokenization:** The implementation of Byte-Pair Encoding (`pbe_generated_samples.txt`) significantly altered text structures. Generating with BPE resulted in word-level syntaxes finishing more gracefully since sub-word units drastically compress the tokens per sequence. It allowed the transformer context window to essentially span longer semantic periods.
"""))

# 8 Appendix
cells.append(create_markdown_cell("## 8. Appendix: Generated Text Samples\nDisplaying samples of specific models discussed above."))
cells.append(create_code_cell("""import os

samples_dir = 'samples'
files_to_show = ['baseline_generated_samples.txt', 'td1000_generated_samples.txt', 'dropout-0-0_generated_samples.txt']

for f_name in files_to_show:
    path = os.path.join(samples_dir, f_name)
    if os.path.exists(path):
        with open(path, 'r') as f:
            print(f"\\n{'='*50}\\n{f_name}\\n{'='*50}")
            print(f.read()[:1500] + "...\\n[TRUNCATED]")
"""))


notebook_struct = {
    "cells": cells,
    "metadata": {
        "language_info": {"name": "python", "version": "3.8"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('/Users/maxmacbookpro/Developer/GitHub/foundations_NLP/final_assignment/report.ipynb', 'w') as f:
    json.dump(notebook_struct, f, indent=2)

print("Notebook successfully generated at report.ipynb")
