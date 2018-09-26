Keras-Transformer
=================

Keras-transformer it's a library implementing nuts and bolts for
building (Universal) Transformer models using Keras. It allows you
to assemble a multi-step Transformer model in a flexible way, for example:

    transformer_block = TransformerBlock(
        name='transformer',
        num_heads=8,
        residual_dropout=0.1,
        attention_dropout=0.1,
        use_masking=True)
    add_coordinate_embedding = TransformerCoordinateEmbedding(
        transformer_depth,
        name='coordinate_embedding')

    for step in range(transformer_depth):
        output = transformer_block(
            add_coordinate_embedding(input, step=step))

The library supports positional encoding and embeddings,
attention masking, memory-compressed attention, ACT (adaptive computation time).
All pieces of the model (like self-attention, activation function, layer normalization)
are available as Keras layers, so, if necessary, you can build your
version of Transformer, by re-arranging them differently or replacing some of them.

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

Language modelling example
--------------------------
This repository contains a simple [example](./example) showing how Keras-transformer works.
It's not a rigorous evaluation of the model's capabilities,
but rather a demonstration on how to use the code.

The code trains a simple language-modeling network on the
[WikiText-2](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset)
dataset and evaluates its perplexity.
The model itself is an Adaptive Universal Transformer with five layers.

To launch the code, you will first need to install the requirements listed
in [example/requirements.txt](./example/requirements.txt). Assuming you work
from a Python virtual environment, you can do this by running

    pip install -r example/requirements.txt

You will also need to make sure you have a backend for Keras.
For instance, you can install Tensorflow:

    pip install tensorflow

Now you can launch the example itself as

    pip -m sample.run

If all goes well, you should see the perplexity falling with each epoch.

    Building vocabulary: 100%|█████████████████████████████████| 36718/36718 [00:04<00:00, 7642.33it/s]
    Learning BPE...Done
    Building BPE vocabulary: 100%|███████████████████████████████| 36718/36718 [00:06<00:00, 5743.74it/s]
    Train on 9414 samples, validate on 957 samples
    Epoch 1/50
    9414/9414 [==============================] - 76s 8ms/step - loss: 7.0847 - perplexity: 1044.2455
        - val_loss: 6.3167 - val_perplexity: 406.5031
    ...

After 50 epochs (~2 hours) of training on GeForce 1080 Ti, I've got
validation perplexity about 87 and test perplexity 85.26. The score
can be further improved, but that is not the point of this demo.

[1]: https://arxiv.org/abs/1706.03762 "Attention Is All You Need"
[2]: https://arxiv.org/abs/1807.03819 "Universal Transformers"