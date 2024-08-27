from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import Dense, TimeDistributed, GlobalAveragePooling2D, LSTM, Input, Concatenate
from tensorflow.keras.models import Model
import tensorflow as tf

class InceptionV3TimeDistributed(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.inception_v3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
        self.inception_v3.trainable = False
        super(InceptionV3TimeDistributed, self).__init__(**kwargs)

    def call(self, inputs):
        return self.inception_v3(inputs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.inception_v3.output_shape[1], self.inception_v3.output_shape[2], self.inception_v3.output_shape[3])

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.units = units
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features):
        # features shape: (batch_size, timesteps, feature_dim)
        score = self.V(tf.nn.tanh(self.W1(features) + self.W2(features)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

def build_model(units=512):
    video_input = Input(shape=(None, 299, 299, 3))
    x = TimeDistributed(InceptionV3TimeDistributed())(video_input)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = LSTM(units, return_sequences=True)(x)
    attention_output, attention_weights = AttentionLayer(units)(x)
    x = Dense(1024, activation='relu')(attention_output)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=video_input, outputs=[output, attention_weights])
    return model