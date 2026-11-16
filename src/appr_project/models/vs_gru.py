"""VS-GRU with per-variable recurrences and optional nonlinear output fusion."""

import tensorflow as tf
from tensorflow import keras

from appr_project.models.tower_grud import VariableGRUDCell


@keras.utils.register_keras_serializable(package="appr_project")
class VSGRUCell(VariableGRUDCell):
    """Use one state per variable and add its missing rate to each GRU gate."""

    input_channels = 5
    gate_input_size = 3

    def __init__(self, variables, **kwargs):
        super().__init__(variables, units=1, **kwargs)

    def build(self, input_shape):
        self.missing_rate_weight = self.add_weight(
            name="missing_rate_weight", shape=(self.n_variables,), initializer="ones"
        )
        self.missing_rate_bias = self.add_weight(
            name="missing_rate_bias", shape=(self.n_variables,), initializer="zeros"
        )
        super().build(input_shape)

    def _gate_features(self, value, observed, inputs):
        missing_rate = inputs[..., 4]
        beta = tf.exp(
            -tf.nn.relu(
                missing_rate * self.missing_rate_weight + self.missing_rate_bias
            )
        )
        return tf.stack((value, observed, beta), axis=-1)

    def get_config(self):
        config = super().get_config()
        del config["units"]
        return config


def build_model(*, variables, static_features, window, use_hidden_dense):
    layers, ops = keras.layers, keras.ops
    temporal = keras.Input((window, variables, 4), name="temporal")
    valid_hours = keras.Input((window,), dtype="bool", name="valid_hours")
    static = keras.Input((static_features,), name="static")

    # Each window ends at the prediction hour. Exclude padding from the rate.
    valid = ops.cast(ops.expand_dims(valid_hours, axis=-1), temporal.dtype)
    missing = ops.sum(valid * (1 - temporal[..., 1]), axis=1, keepdims=True)
    missing_rate = ops.divide_no_nan(missing, ops.sum(valid, axis=1, keepdims=True))
    rates = ops.repeat(ops.expand_dims(missing_rate, -1), window, axis=1)
    flat = layers.Reshape((window, variables * VSGRUCell.input_channels))(
        ops.concatenate((temporal, rates), axis=-1)
    )
    summary = layers.RNN(VSGRUCell(variables), name="variable_states")(
        flat, mask=valid_hours
    )
    fused = layers.Concatenate()([summary, static])
    if use_hidden_dense:
        fused = layers.Dense(32, activation="relu", name="hidden_dense")(fused)
    probability = layers.Dense(1, activation="sigmoid", name="sepsis_probability")(
        fused
    )
    return keras.Model(
        inputs={"temporal": temporal, "valid_hours": valid_hours, "static": static},
        outputs=probability,
        name="vs_gru",
    )
