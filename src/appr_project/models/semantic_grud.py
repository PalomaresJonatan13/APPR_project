"""Multivariate GRU-D towers grouped by physiological domain."""

import tensorflow as tf
from tensorflow import keras

SEMANTIC_GROUPS = {
    "hemodynamics": ("HR", "SBP", "MAP", "DBP", "Lactate"),
    "respiration": ("Resp", "O2Sat", "SaO2", "FiO2", "PaCO2", "pH"),
    "renal_chemistry": (
        "BUN",
        "Creatinine",
        "Calcium",
        "Magnesium",
        "Phosphate",
        "Potassium",
        "Glucose",
    ),
    "hepatic": ("AST", "Alkalinephos", "Bilirubin_total"),
    "hematology": ("Hct", "Hgb", "WBC", "Platelets", "Temp"),
}

SEMANTIC_GROUPS_3 = {
    "hemodynamics_oxygenation": (
        *SEMANTIC_GROUPS["hemodynamics"],
        *SEMANTIC_GROUPS["respiration"],
        "Temp",
    ),
    "renal_hepatic": (
        *SEMANTIC_GROUPS["renal_chemistry"],
        *SEMANTIC_GROUPS["hepatic"],
    ),
    "hematology": ("Hct", "Hgb", "WBC", "Platelets"),
}

GROUPS_BY_ARCHITECTURE = {
    "semantic_grud": SEMANTIC_GROUPS,
    "semantic_grud_3": SEMANTIC_GROUPS_3,
}


def semantic_indices(temporal, groups=SEMANTIC_GROUPS):
    expected = {variable for group in groups.values() for variable in group}
    missing = expected.difference(temporal)
    extra = set(temporal).difference(expected)
    if missing or extra or len(temporal) != len(expected):
        raise ValueError(
            "Semantic towers require the 26 cleaned temporal variables exactly once. "
            f"Missing: {sorted(missing)}. Unexpected: {sorted(extra)}."
        )
    return {
        name: [temporal.index(variable) for variable in group]
        for name, group in groups.items()
    }


@keras.utils.register_keras_serializable(package="appr_project")
class SemanticGRUDCell(keras.layers.Layer):
    def __init__(self, variables, units, **kwargs):
        super().__init__(**kwargs)
        self.n_variables = variables
        self.units = units
        self.state_size = units
        self.output_size = units
        self.gru = keras.layers.GRUCell(units, reset_after=False)
        self.hidden_decay = keras.layers.Dense(units)

    def build(self, input_shape):
        self.input_decay = self.add_weight(
            name="input_decay", shape=(self.n_variables,), initializer="ones"
        )
        self.input_decay_bias = self.add_weight(
            name="input_decay_bias", shape=(self.n_variables,), initializer="zeros"
        )
        self.hidden_decay.build((None, self.n_variables))
        self.gru.build((None, 2 * self.n_variables))
        super().build(input_shape)

    def call(self, inputs, states, training=None):
        channels = tf.reshape(inputs, (-1, self.n_variables, 4))
        value, observed, elapsed, previous = tf.unstack(channels, axis=-1)
        gamma_x = tf.exp(
            -tf.nn.relu(elapsed * self.input_decay + self.input_decay_bias)
        )
        # The training mean is zero in standardized units.
        value = tf.where(observed > 0, value, gamma_x * previous)
        gamma_h = tf.exp(-tf.nn.relu(self.hidden_decay(elapsed)))
        return self.gru(
            tf.concat((value, observed), axis=-1),
            [gamma_h * states[0]],
            training=training,
        )

    def get_config(self):
        return {
            **super().get_config(),
            "variables": self.n_variables,
            "units": self.units,
        }


def build_model(
    *,
    temporal_variables,
    static_features,
    window,
    units,
    dense_units,
    dropout,
    architecture="semantic_grud",
):
    groups = semantic_indices(temporal_variables, GROUPS_BY_ARCHITECTURE[architecture])
    layers, ops = keras.layers, keras.ops
    temporal = keras.Input((window, len(temporal_variables), 4), name="temporal")
    valid_hours = keras.Input((window,), dtype="bool", name="valid_hours")
    static = keras.Input((static_features,), name="static")
    summaries = []
    for name, indices in groups.items():
        grouped = ops.take(temporal, indices, axis=2)
        flat = layers.Reshape((window, len(indices) * 4))(grouped)
        states = layers.RNN(
            SemanticGRUDCell(len(indices), units),
            return_sequences=True,
            zero_output_for_mask=True,
            name=name,
        )(flat, mask=valid_hours)
        scores = layers.Dense(1, use_bias=False, name=f"{name}_attention")(
            ops.tanh(states)
        )
        valid = ops.expand_dims(valid_hours, axis=-1)
        weights = ops.softmax(ops.where(valid, scores, -1e4), axis=1)
        summaries.append(ops.sum(weights * states, axis=1))

    fused = layers.Concatenate()([*summaries, static])
    hidden = layers.Dense(dense_units, activation="relu")(fused)
    hidden = layers.Dropout(dropout)(hidden)
    probability = layers.Dense(1, activation="sigmoid", name="sepsis_probability")(
        hidden
    )
    return keras.Model(
        inputs={"temporal": temporal, "valid_hours": valid_hours, "static": static},
        outputs=probability,
        name=architecture,
    )
