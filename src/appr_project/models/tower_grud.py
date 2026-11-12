"""Independent GRU-D towers with temporal attention and dense fusion."""

import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="appr_project")
class VariableGRUDCell(keras.layers.Layer):
    """Vectorize towers along the variable axis while keeping separate weights."""

    def __init__(self, variables, units, **kwargs):
        super().__init__(**kwargs)
        self.n_variables = variables
        self.units = units
        self.state_size = variables * units
        self.output_size = self.state_size

    def build(self, input_shape):
        variables, units = self.n_variables, self.units
        self.input_kernel = self.add_weight(
            name="input_kernel",
            shape=(variables, 2, 3 * units),
            initializer="glorot_uniform",
        )
        self.recurrent_kernel = self.add_weight(
            name="recurrent_kernel",
            shape=(variables, units, 3 * units),
            initializer="glorot_uniform",
        )
        self.bias = self.add_weight(
            name="bias", shape=(variables, 3 * units), initializer="zeros"
        )
        self.input_decay = self.add_weight(
            name="input_decay", shape=(variables,), initializer="ones"
        )
        self.input_decay_bias = self.add_weight(
            name="input_decay_bias", shape=(variables,), initializer="zeros"
        )
        self.state_decay = self.add_weight(
            name="state_decay", shape=(variables, units), initializer="ones"
        )
        self.state_decay_bias = self.add_weight(
            name="state_decay_bias", shape=(variables, units), initializer="zeros"
        )
        super().build(input_shape)

    def call(self, inputs, states):
        inputs = tf.reshape(inputs, (-1, self.n_variables, 4))
        value, observed, elapsed, previous = tf.unstack(inputs, axis=-1)
        hidden = tf.reshape(states[0], (-1, self.n_variables, self.units))

        # The training mean is zero in standardized units.
        gamma_x = tf.exp(
            -tf.nn.relu(elapsed * self.input_decay + self.input_decay_bias)
        )
        value = tf.where(observed > 0, value, gamma_x * previous)
        gamma_h = tf.exp(
            -tf.nn.relu(elapsed[..., None] * self.state_decay + self.state_decay_bias)
        )
        hidden = gamma_h * hidden

        features = tf.stack((value, observed), axis=-1)
        x_update, x_reset, x_candidate = tf.split(
            tf.einsum("bvi,viu->bvu", features, self.input_kernel) + self.bias,
            3,
            axis=-1,
        )
        gates, candidate_kernel = tf.split(
            self.recurrent_kernel, [2 * self.units, self.units], axis=-1
        )
        h_update, h_reset = tf.split(
            tf.einsum("bvi,viu->bvu", hidden, gates), 2, axis=-1
        )
        update = tf.sigmoid(x_update + h_update)
        reset = tf.sigmoid(x_reset + h_reset)
        candidate = tf.tanh(
            x_candidate + tf.einsum("bvi,viu->bvu", reset * hidden, candidate_kernel)
        )
        state = tf.reshape(
            (1 - update) * hidden + update * candidate, (-1, self.state_size)
        )
        return state, [state]

    def get_config(self):
        return {
            **super().get_config(),
            "variables": self.n_variables,
            "units": self.units,
        }


@keras.utils.register_keras_serializable(package="appr_project")
class VariableAttention(keras.layers.Layer):
    def build(self, input_shape):
        states_shape, _ = input_shape
        self.score = self.add_weight(
            name="score", shape=states_shape[-2:], initializer="glorot_uniform"
        )
        super().build(input_shape)

    def call(self, inputs):
        states, valid_hours = inputs
        scores = tf.einsum("btvu,vu->btv", tf.tanh(states), self.score)
        scores = tf.where(valid_hours[..., None], scores, tf.cast(-1e4, scores.dtype))
        weights = tf.nn.softmax(scores, axis=1)
        return tf.einsum("btv,btvu->bvu", weights, states)


def build_model(*, variables, static_features, window, units, dense_units, dropout):
    layers = keras.layers
    temporal = keras.Input((window, variables, 4), name="temporal")
    valid_hours = keras.Input((window,), dtype="bool", name="valid_hours")
    static = keras.Input((static_features,), name="static")
    flat = layers.Reshape((window, variables * 4))(temporal)
    states = layers.RNN(
        VariableGRUDCell(variables, units),
        return_sequences=True,
        zero_output_for_mask=True,
    )(flat, mask=valid_hours)
    states = layers.Reshape((window, variables, units))(states)
    summaries = VariableAttention()([states, valid_hours])
    fused = layers.Concatenate()([layers.Flatten()(summaries), static])
    hidden = layers.Dense(dense_units, activation="relu")(fused)
    hidden = layers.Dropout(dropout)(hidden)
    probability = layers.Dense(1, activation="sigmoid", name="sepsis_probability")(
        hidden
    )
    return keras.Model(
        inputs={"temporal": temporal, "valid_hours": valid_hours, "static": static},
        outputs=probability,
        name="tower_grud",
    )
