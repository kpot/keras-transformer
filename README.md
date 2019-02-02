Keras-Transformer
=================

Keras-transformer is a Python library implementing nuts and bolts,
for building (Universal) Transformer models using [Keras](http://keras.io),
and equipped with [examples](#language-modelling-examples-with-bert-and-gpt)
of how it can be applied.

The library supports:

* positional encoding and embeddings,
* attention masking,
* memory-compressed attention,
* ACT (adaptive computation time),
* a general implementation of [BERT][3] (because the Transformer
  is mainly applied to NLP tasks).

It allows you to piece together a multi-step Transformer model
in a flexible way, for example:

```python
transformer_block = TransformerBlock(
    name='transformer',
    num_heads=8,
    residual_dropout=0.1,
    attention_dropout=0.1,
    use_masking=True)
add_coordinate_embedding = TransformerCoordinateEmbedding(
    transformer_depth,
    name='coordinate_embedding')
    
output = transformer_input # shape: (<batch size>, <sequence length>, <input size>)
for step in range(transformer_depth):
    output = transformer_block(
        add_coordinate_embedding(output, step=step))
```


All pieces of the model (like self-attention, activation function,
layer normalization) are available as Keras layers, so, if necessary,
you can build your version of Transformer, by re-arranging them
differently or replacing some of them.

The (Universal) Transformer is a deep learning architecture
described in arguably one of the most impressive DL papers of 2017 and 2018:
the "[Attention is all you need][1]" and the "[Universal Transformers][2]"
by Google Research and Google Brain teams.

The authors brought the idea of recurrent multi-head self-attention,
which has inspired a big wave of new research models that keep coming ever since.
These models demonstrate new state-of-the-art results in various NLP tasks,
including translation, parsing, question answering, and even some algorithmic tasks.

Installation
------------
To install the library you need to clone the repository

    git clone https://github.com/kpot/keras-transformer.git

then switch to the cloned directory and run pip

    cd keras-transformer
    pip install .

Please note that the project requires Python >= 3.6.

Language modelling examples with BERT and GPT
---------------------------------------------
This repository contains simple [examples](./example) showing how
Keras-transformer works.
It's not a rigorous evaluation of the model's capabilities,
but rather a demonstration on how to use the code.

The code trains [simple language-modeling networks](./example/models.py) on the
[WikiText-2](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset)
dataset and evaluates their perplexity. The model is either a [vanilla
Transformer][1], or an [Adaptive Universal Transformer][2] (by default)
with five layers, each can be trained using either:

* [Generative pre-training][4] (GPT), which involves using masked self-attention
  to prevent the model from "looking into the future".
* [BERT][3], which doesn't restrict self-attention, allowing the model
  to fill the gaps using both left and right context.


To launch the code, you will first need to install the requirements listed
in [example/requirements.txt](./example/requirements.txt). Assuming you work
from a Python virtual environment, you can do this by running

    pip install -r example/requirements.txt

You will also need to make sure you have a backend for Keras.
For instance, you can install Tensorflow (the sample was tested using
Tensorflow and PlaidML as backends):

    pip install tensorflow

Now you can launch the GPT example as

    python -m example.run_gpt --save lm_model.h5

to see all command line options and their default values, try

    python -m example.run_gpt --help

If all goes well, after launching the example you should see
the perplexity falling with each epoch.

    Building vocabulary: 100%|█████████████████████████████████| 36718/36718 [00:04<00:00, 7642.33it/s]
    Learning BPE...Done
    Building BPE vocabulary: 100%|███████████████████████████████| 36718/36718 [00:06<00:00, 5743.74it/s]
    Train on 9414 samples, validate on 957 samples
    Epoch 1/50
    9414/9414 [==============================] - 76s 8ms/step - loss: 7.0847 - perplexity: 1044.2455
        - val_loss: 6.3167 - val_perplexity: 406.5031
    ...

After 200 epochs (~5 hours) of training on GeForce 1080 Ti, I've got
validation perplexity about 51.61 and test perplexity 50.82. The score
can be further improved, but that is not the point of this demo.

BERT model example can be launched similarly

    python -m example.run_bert --save lm_model.h5 --model vanilla

but you will need to be patient. BERT easily achieves better performance
than GPT, but requires much more training time to converge.

[1]: https://arxiv.org/abs/1706.03762 "Attention Is All You Need"
[2]: https://arxiv.org/abs/1807.03819 "Universal Transformers"
[3]: https://arxiv.org/abs/1810.04805 "BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding"
[4]: https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
     "Improving Language Understanding by Generative Pre-Training"
