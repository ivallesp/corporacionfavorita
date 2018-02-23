import tensorflow as tf


class NameSpacer:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Architecture:
    def __init__(self, name="architecture"):

        self.name = name
        self.define_computation_graph()

        # Aliases
        self.ph = self.placeholders
        self.op = self.optimizers
        self.summ = self.summaries

    def define_computation_graph(self):
        # Reset graph
        tf.reset_default_graph()
        self.placeholders = NameSpacer(**self.define_placeholders())
        self.core_model = NameSpacer(**self.define_core_model())
        self.losses = NameSpacer(**self.define_losses())
        self.optimizers = NameSpacer(**self.define_optimizers())
        self.summaries = NameSpacer(**self.define_summaries())

    def define_placeholders(self):
        with tf.variable_scope("Placeholders"):
            pass
            return {}

    def define_core_model(self):
        with tf.variable_scope("Core_Model"):
            pass
            return {}

    def define_losses(self):
        with tf.variable_scope("Losses"):
            pass
            return {}

    def define_optimizers(self):
        with tf.variable_scope("Optimization"):
            pass
            return {}

    def define_summaries(self):
        with tf.variable_scope("Summaries"):
            pass
            return {}
