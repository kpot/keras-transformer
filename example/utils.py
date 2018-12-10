import math
import warnings

import h5py
from keras import Model


def load_optimizer_weights(model: Model, model_save_path: str):
    """
    Loads optimizer's weights for the model from an HDF5 file.
    """
    with h5py.File(model_save_path, mode='r') as f:
        if 'optimizer_weights' in f:
            # Build train function (to get weight updates).
            # noinspection PyProtectedMember
            model._make_train_function()
            optimizer_weights_group = f['optimizer_weights']
            optimizer_weight_names = [
                n.decode('utf8') for n in
                optimizer_weights_group.attrs['weight_names']]
            optimizer_weight_values = [
                optimizer_weights_group[n]
                for n in optimizer_weight_names]
            try:
                model.optimizer.set_weights(optimizer_weight_values)
            except ValueError:
                warnings.warn('Error in loading the saved optimizer '
                              'state. As a result, your model is '
                              'starting with a freshly initialized '
                              'optimizer.')


def contain_tf_gpu_mem_usage():
    """
    By default TensorFlow may try to reserve all available GPU memory
    making it impossible to train multiple networks at once.
    This function will disable such behaviour in TensorFlow.
    """
    from keras import backend
    if backend.backend() != 'tensorflow':
        return
    try:
        # noinspection PyPackageRequirements
        import tensorflow as tf
    except ImportError:
        pass
    else:
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory
        sess = tf.Session(config=config)
        set_session(sess)


class CosineLRSchedule:
    """
    Cosine annealing with warm restarts, described in paper
    "SGDR: stochastic gradient descent with warm restarts"
    https://arxiv.org/abs/1608.03983

    Changes the learning rate, oscillating it between `lr_high` and `lr_low`.
    It takes `period` epochs for the learning rate to drop to its very minimum,
    after which it quickly returns back to `lr_high` (resets) and everything
    starts over again.

    With every reset:
        * the period grows, multiplied by factor `period_mult`
        * the maximum learning rate drops proportionally to `high_lr_mult`

    This class is supposed to be used with
    `keras.callbacks.LearningRateScheduler`.
    """
    def __init__(self, lr_high: float, lr_low: float, initial_period: int = 50,
                 period_mult: float = 2, high_lr_mult: float = 0.97):
        self._lr_high = lr_high
        self._lr_low = lr_low
        self._initial_period = initial_period
        self._period_mult = period_mult
        self._high_lr_mult = high_lr_mult

    def __call__(self, epoch, lr):
        return self.get_lr_for_epoch(epoch)

    def get_lr_for_epoch(self, epoch):
        assert epoch >= 0
        t_cur = 0
        lr_max = self._lr_high
        period = self._initial_period
        result = lr_max
        for i in range(epoch + 1):
            if i == epoch:  # last iteration
                result = (self._lr_low +
                          0.5 * (lr_max - self._lr_low) *
                          (1 + math.cos(math.pi * t_cur / period)))
            else:
                if t_cur == period:
                    period *= self._period_mult
                    lr_max *= self._high_lr_mult
                    t_cur = 0
                else:
                    t_cur += 1
        return result
