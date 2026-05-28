Transformer from Scratch for English-to-French Translation using OPUS Books

This project implements a Transformer model from scratch in PyTorch for English-to-French machine translation. The project is based on the original idea of building a Transformer architecture manually, but extends it by replacing randomly generated toy training data with the real OPUS Books English-French translation dataset.

The goal of this project is to better understand how the Transformer architecture works internally, including multi-head attention, positional encoding, encoder-decoder layers, masking, tokenization, training, validation, and inference.

Project Overview

The model was built without using a pretrained translation model. A pretrained tokenizer from Helsinki-NLP/opus-mt-en-fr was used only to convert English and French text into token IDs. The Transformer architecture itself was implemented manually in PyTorch.

The project includes:

Custom Multi-Head Attention implementation
Positional Encoding
Encoder and Decoder layers
Full Transformer encoder-decoder architecture
OPUS Books English-French dataset loading
Tokenization and padding
Decoder start-token handling
Training and validation loops
K-Fold cross-validation
Early stopping based on validation loss
Final model training on the full dataset
Greedy decoding for English-to-French translation
Dataset

This project uses the OPUS Books English-French dataset from Hugging Face:

load_dataset("Helsinki-NLP/opus_books", "en-fr")

The dataset consists of paired English and French sentences from books. These sentence pairs are used to train the Transformer to translate English input sentences into French output sentences.

Validation Method

K-Fold cross-validation was added to evaluate the model more reliably. Instead of relying on a single train-validation split, the dataset subset is divided into multiple folds. For each fold, a new Transformer model is trained from scratch on K-1 folds and validated on the remaining fold.

This helps evaluate whether the model configuration is stable across different train-validation splits.

Final Model Training

After validating the setup with K-Fold cross-validation, the final model is intended to be trained on the full OPUS Books dataset using early stopping. The final model uses a 90/10 train-validation split and continues training while validation loss improves.

During the full-dataset training run, the Google Colab GPU usage limit was reached before the final training could complete. The full-dataset training will be resumed and completed after the next Colab GPU reset.

Current Status
Transformer architecture implemented from scratch
OPUS Books dataset integrated
K-Fold validation implemented
Early stopping implemented
Initial training and validation runs completed successfully
Training and validation loss decreased over epochs
Full-dataset training started but paused due to Google Colab GPU limit
Full training will continue after GPU access resets
Notes

This project is intended as a learning-focused implementation. Since the model is trained from scratch, translation quality depends heavily on dataset size, training time, model size, and available GPU resources. The main purpose is to demonstrate a working Transformer pipeline and understand the components behind sequence-to-sequence translation.
