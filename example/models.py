from keras import regularizers
from keras.models import Model
# noinspection PyPep8Naming
from keras import backend as K
from keras.layers import Input, Softmax, Embedding, Add, Lambda, Dense

from keras_transformer.extras import ReusableEmbedding, TiedOutputEmbedding
from keras_transformer.position import TransformerCoordinateEmbedding
from keras_transformer.transformer import TransformerACT, TransformerBlock


def universal_transformer_gpt_model(
        max_seq_length: int, vocabulary_size: int,
        word_embedding_size: int, transformer_depth: int,
        num_heads: int, transformer_dropout: float = 0.1,
        embedding_dropout: float = 0.6,
        l2_reg_penalty: float = 1e-6,
        confidence_penalty_weight: float = 0.1):
    """
    A model which is similar to the one described by OpenAI in paper
    "Improving Language Understanding by Generative Pre-Training", except
    that it relies L2 regularization of the word embedding matrix
    (instead of the dropout), and uses Universal Transformer architecture.
    """
    word_ids = Input(shape=(max_seq_length,), dtype='int32', name='word_ids')
    l2_regularizer = (regularizers.l2(l2_reg_penalty) if l2_reg_penalty
                      else None)
    embedding_layer = ReusableEmbedding(
        vocabulary_size, word_embedding_size,
        input_length=max_seq_length,
        name='bpe_embeddings',
        # Regularization is based on paper "A Comparative Study on
        # Regularization Strategies for Embedding-based Neural Networks"
        # https://arxiv.org/pdf/1508.03721.pdf
        embeddings_regularizer=l2_regularizer)
    output_layer = TiedOutputEmbedding(
        projection_regularizer=l2_regularizer,
        projection_dropout=embedding_dropout,
        name='word_prediction_logits')
    coordinate_embedding_layer = TransformerCoordinateEmbedding(
        transformer_depth,
        name='coordinate_embedding')
    transformer_act_layer = TransformerACT(name='adaptive_computation_time')
    transformer_block = TransformerBlock(
        name='transformer', num_heads=num_heads,
        residual_dropout=transformer_dropout,
        attention_dropout=transformer_dropout,
        use_masking=True, vanilla_wiring=False)
    output_softmax_layer = Softmax(name='word_predictions')

    next_step_input, embedding_matrix = embedding_layer(word_ids)
    act_output = next_step_input

    for i in range(transformer_depth):
        next_step_input = coordinate_embedding_layer(next_step_input, step=i)
        next_step_input = transformer_block(next_step_input)
        next_step_input, act_output = transformer_act_layer(next_step_input)

    transformer_act_layer.finalize()
    next_step_input = act_output
    word_predictions = output_softmax_layer(
        output_layer([next_step_input, embedding_matrix]))
    model = Model(inputs=[word_ids], outputs=[word_predictions])
    # Penalty for confidence of the output distribution, as described in
    # "Regularizing Neural Networks by Penalizing Confident
    # Output Distributions" (https://arxiv.org/abs/1701.06548)
    confidence_penalty = K.mean(
        confidence_penalty_weight *
        K.sum(word_predictions * K.log(word_predictions), axis=-1))
    model.add_loss(confidence_penalty)
    return model


def vanilla_transformer_gpt_model(
        max_seq_length: int, vocabulary_size: int,
        word_embedding_size: int, transformer_depth: int,
        num_heads: int, transformer_dropout: float = 0.1,
        embedding_dropout: float = 0.6,
        l2_reg_penalty: float = 1e-6,
        confidence_penalty_weight: float = 0.1):
    """
    A model which is almost identical to the one described by OpenAI in paper
    "Improving Language Understanding by Generative Pre-Training", except
    that it uses L2 regularization of the word embedding matrix,
    instead of the dropout.
    """
    word_ids = Input(shape=(max_seq_length,), dtype='int32', name='word_ids')
    l2_regularizer = (regularizers.l2(l2_reg_penalty) if l2_reg_penalty
                      else None)
    embedding_layer = ReusableEmbedding(
        vocabulary_size, word_embedding_size,
        input_length=max_seq_length,
        name='bpe_embeddings',
        # Regularization is based on paper "A Comparative Study on
        # Regularization Strategies for Embedding-based Neural Networks"
        # https://arxiv.org/pdf/1508.03721.pdf
        embeddings_regularizer=l2_regularizer)
    output_layer = TiedOutputEmbedding(
        projection_regularizer=l2_regularizer,
        projection_dropout=embedding_dropout,
        name='word_prediction_logits')
    coordinate_embedding_layer = TransformerCoordinateEmbedding(
        1,
        name='coordinate_embedding')
    output_softmax_layer = Softmax(name='word_predictions')

    next_step_input, embedding_matrix = embedding_layer(word_ids)

    next_step_input = coordinate_embedding_layer(next_step_input, step=0)
    for i in range(transformer_depth):
        next_step_input = (
            TransformerBlock(
                name='transformer' + str(i), num_heads=num_heads,
                residual_dropout=transformer_dropout,
                attention_dropout=transformer_dropout,
                use_masking=True,
                vanilla_wiring=True)
            (next_step_input))

    word_predictions = output_softmax_layer(
        output_layer([next_step_input, embedding_matrix]))
    model = Model(inputs=[word_ids], outputs=[word_predictions])
    # Penalty for confidence of the output distribution, as described in
    # "Regularizing Neural Networks by Penalizing Confident
    # Output Distributions" (https://arxiv.org/abs/1701.06548)
    confidence_penalty = K.mean(
        confidence_penalty_weight *
        K.sum(word_predictions * K.log(word_predictions), axis=-1))
    model.add_loss(confidence_penalty)
    return model


def transformer_bert_model(
        max_seq_length: int, vocabulary_size: int,
        word_embedding_size: int,
        use_universal_transformer: bool,
        transformer_depth: int,
        num_heads: int,
        transformer_dropout: float = 0.1,
        embedding_dropout: float = 0.6,
        l2_reg_penalty: float = 1e-4):
    """
    Builds a BERT-based model (Bidirectional Encoder Representations
    from Transformers) following paper "BERT: Pre-training of Deep
    Bidirectional Transformers for Language Understanding"
    (https://arxiv.org/abs/1810.04805)

    Depending on the value passed with `use_universal_transformer` argument,
    this function applies either an Adaptive Universal Transformer (2018)
    or a vanilla Transformer (2017) to do the job (the original paper uses
    vanilla Transformer).
    """
    word_ids = Input(shape=(max_seq_length,), dtype='int32', name='word_ids')
    segment_ids = Input(
        shape=(max_seq_length,), dtype='int32', name='segment_ids')
    l2_regularizer = (regularizers.l2(l2_reg_penalty) if l2_reg_penalty
                      else None)
    embedding_layer = ReusableEmbedding(
        vocabulary_size, word_embedding_size,
        input_length=max_seq_length,
        name='bpe_embeddings',
        # Regularization is based on paper "A Comparative Study on
        # Regularization Strategies for Embedding-based Neural Networks"
        # https://arxiv.org/pdf/1508.03721.pdf
        embeddings_regularizer=l2_regularizer)
    segment_embedding_layer = Embedding(
        2,  # "Segment A" and "Segment B" embeddings
        word_embedding_size, name='segment_embeddings')
    add_segment_layer = Add(name='add_segment')
    output_layer = TiedOutputEmbedding(
        projection_regularizer=l2_regularizer,
        projection_dropout=embedding_dropout,
        name='word_prediction_logits')
    output_softmax_layer = Softmax(name='word_predictions')
    coordinate_embedding_layer = TransformerCoordinateEmbedding(
        transformer_depth if use_universal_transformer else 1,
        name='coordinate_embedding')

    next_step_input, embedding_matrix = embedding_layer(word_ids)
    segment_embeddings = segment_embedding_layer(segment_ids)

    if use_universal_transformer:
        # Building a Universal Transformer (2018)
        act_layer = TransformerACT(
            name='adaptive_computation_time')
        transformer_block = TransformerBlock(
            name='transformer', num_heads=num_heads,
            residual_dropout=transformer_dropout,
            attention_dropout=transformer_dropout,
            # Allow bi-directional attention
            use_masking=False)

        act_output = next_step_input
        for i in range(transformer_depth):
            next_step_input = coordinate_embedding_layer(
                next_step_input, step=i)
            next_step_input = add_segment_layer(
                [next_step_input, segment_embeddings])
            next_step_input = transformer_block(next_step_input)
            next_step_input, act_output = act_layer(next_step_input)

        act_layer.finalize()
        next_step_input = act_output
    else:
        # Building a Vanilla Transformer (described in
        # "Attention is all you need", 2017)
        next_step_input = coordinate_embedding_layer(next_step_input, step=0)
        next_step_input = add_segment_layer(
            [next_step_input, segment_embeddings])
        for i in range(transformer_depth):
            next_step_input = (
                TransformerBlock(
                    name='transformer' + str(i), num_heads=num_heads,
                    residual_dropout=transformer_dropout,
                    attention_dropout=transformer_dropout,
                    use_masking=False,  # Allow bi-directional attention
                    vanilla_wiring=True)
                (next_step_input))

    word_predictions = output_softmax_layer(
        output_layer([next_step_input, embedding_matrix]))
    cls_node_slice = (
        # selecting the first output position in each sequence
        # (responsible for classification)
        Lambda(lambda x: x[:, 0], name='cls_node_slicer')
        (next_step_input))
    class_prediction = (
        Dense(1, name='class_prediction', activation='sigmoid')
        (cls_node_slice))
    model = Model(
        inputs=[word_ids, segment_ids],
        outputs=[word_predictions, class_prediction])
    return model
