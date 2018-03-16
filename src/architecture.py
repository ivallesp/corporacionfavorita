import tensorflow as tf
from src.tf_frankenstein.normalization import BatchNorm
from src.tf_frankenstein.decoder import decoder
from src.general_utilities import flatten

class NameSpacer:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Architecture:
    def __init__(self, n_timesteps_past, n_timesteps_future, cardinalities,
                 embedding_sizes, batch_size=None, name="architecture"):
        self.name = name
        self.n_timesteps_past = n_timesteps_past
        self.n_timesteps_future = n_timesteps_future
        self.cardinalities = cardinalities
        self.barch_size = batch_size
        self.embedding_sizes = embedding_sizes
        self.get_emb_shape = lambda x: (self.cardinalities[x], self.embedding_sizes[x])
        self.n_recurrent_cells = 128
        self.n_recurrent_layers = 3
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
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
            loss_dev = tf.placeholder(dtype=tf.float32, shape=(self.barch_size), name="loss_dev_manual")
            mape_dev = tf.placeholder(dtype=tf.float32, shape=(self.barch_size), name="mape_dev_manual")
            unit_sales = tf.placeholder(dtype=tf.float32, shape=(self.barch_size, self.n_timesteps_past, 1),
                                        name="unit_sales")
            target = tf.placeholder(dtype=tf.float32, shape=(self.barch_size, self.n_timesteps_future, 1),
                                    name="unit_sales")
            store_nbr = tf.placeholder(dtype=tf.int32, shape=(self.barch_size,), name="store_nbr")
            city = tf.placeholder(dtype=tf.int32, shape=(self.barch_size,), name="city")
            state = tf.placeholder(dtype=tf.int32, shape=(self.barch_size,), name="state")
            store_type = tf.placeholder(dtype=tf.int32, shape=(self.barch_size,), name="store_type")
            store_cluster = tf.placeholder(dtype=tf.int32, shape=(self.barch_size,), name="store_cluster")
            item_family = tf.placeholder(dtype=tf.int32, shape=(self.barch_size,), name="item_family")
            item_class = tf.placeholder(dtype=tf.int32, shape=(self.barch_size,), name="item_class")
            item_nbr = tf.placeholder(dtype=tf.int32, shape=(self.barch_size,), name="item_nbr")
            national_holiday_type = tf.placeholder(dtype=tf.int32, shape=(self.barch_size, self.n_timesteps_past, 1),
                                                   name="national_holiday_type")
            local_holiday_type = tf.placeholder(dtype=tf.int32, shape=(self.barch_size, self.n_timesteps_past, 1),
                                                name="local_holiday_type")
            onpromotion = tf.placeholder(dtype=tf.float32, shape=(self.barch_size, self.n_timesteps_past, 1),
                                         name="onpromotion")
            item_perishable = tf.placeholder(dtype=tf.float32, shape=(self.barch_size,), name="item_perishable")
            national_holiday_transferred = tf.placeholder(dtype=tf.float32,
                                                          shape=(self.barch_size, self.n_timesteps_past, 1),
                                                          name="national_holiday_transferred")
            local_holiday_transferred = tf.placeholder(dtype=tf.float32,
                                                       shape=(self.barch_size, self.n_timesteps_past, 1),
                                                       name="local_holiday_transferred")
            national_holiday = tf.placeholder(dtype=tf.float32, shape=(self.barch_size, self.n_timesteps_past, 1),
                                              name="national_holiday")
            regional_holiday = tf.placeholder(dtype=tf.float32, shape=(self.barch_size, self.n_timesteps_past, 1),
                                              name="regional_holiday")
            local_holiday = tf.placeholder(dtype=tf.float32, shape=(self.barch_size, self.n_timesteps_past, 1),
                                           name="local_holiday")
            dcoilwtico = tf.placeholder(dtype=tf.float32, shape=(self.barch_size, self.n_timesteps_past, 1),
                                        name="dcoilwtico")
            transactions = tf.placeholder(dtype=tf.float32, shape=(self.barch_size, self.n_timesteps_past, 1),
                                          name="transactions")
            local_holiday_fut = tf.placeholder(dtype=tf.float32, shape=(self.barch_size, self.n_timesteps_future, 1),
                                           name="local_holiday_fut")
            national_holiday_fut = tf.placeholder(dtype=tf.float32, shape=(self.barch_size, self.n_timesteps_future, 1),
                                           name="national_holiday_fut")
            regional_holiday_fut = tf.placeholder(dtype=tf.float32, shape=(self.barch_size, self.n_timesteps_future, 1),
                                           name="regional_holiday_fut")
            year = tf.placeholder(dtype=tf.float32, shape=(self.barch_size, self.n_timesteps_past, 1),
                                           name="year")
            month = tf.placeholder(dtype=tf.float32, shape=(self.barch_size, self.n_timesteps_past, 1),
                                           name="month")
            day = tf.placeholder(dtype=tf.float32, shape=(self.barch_size, self.n_timesteps_past, 1),
                                           name="day")
            dow = tf.placeholder(dtype=tf.float32, shape=(self.barch_size, self.n_timesteps_past, 1),
                                           name="dow")
            year_fut = tf.placeholder(dtype=tf.float32, shape=(self.barch_size, self.n_timesteps_future, 1),
                                           name="year_fut")
            month_fut = tf.placeholder(dtype=tf.float32, shape=(self.barch_size, self.n_timesteps_future, 1),
                                           name="month_fut")
            day_fut = tf.placeholder(dtype=tf.float32, shape=(self.barch_size, self.n_timesteps_future, 1),
                                           name="day_fut")
            dow_fut = tf.placeholder(dtype=tf.float32, shape=(self.barch_size, self.n_timesteps_future, 1),
                                           name="dow_fut")
            onpromotion_fut = tf.placeholder(dtype=tf.float32, shape=(self.barch_size, self.n_timesteps_future, 1),
                                         name="onpromotion_fut")

            is_train = tf.placeholder(dtype=tf.bool, shape=None, name="is_train")

            return {"unit_sales": unit_sales,
                    "target": target,
                    "store_nbr": store_nbr,
                    "city": city,
                    "state": state,
                    "store_type": store_type,
                    "store_cluster": store_cluster,
                    "item_family": item_family,
                    "item_class": item_class,
                    "item_nbr": item_nbr,
                    "national_holiday_type": national_holiday_type,
                    "local_holiday_type": local_holiday_type,
                    "onpromotion": onpromotion,
                    "national_holiday_transferred": national_holiday_transferred,
                    "item_perishable": item_perishable,
                    "local_holiday_transferred": local_holiday_transferred,
                    "national_holiday": national_holiday,
                    "regional_holiday": regional_holiday,
                    "local_holiday": local_holiday,
                    "dcoilwtico": dcoilwtico,
                    "transactions": transactions,
                    "local_holiday_fut": local_holiday_fut,
                    "national_holiday_fut": national_holiday_fut,
                    "regional_holiday_fut": regional_holiday_fut,
                    "year": year,
                    "month": month,
                    "day": day,
                    "dow": dow,
                    "year_fut": year_fut,
                    "month_fut": month_fut,
                    "day_fut": day_fut,
                    "dow_fut": dow_fut,
                    "onpromotion_fut": onpromotion_fut,
                    "loss_dev": loss_dev,
                    "mape_dev": mape_dev,
                    "is_train": is_train}

    def define_core_model(self):
        with tf.variable_scope("Core_Model"):
            # Embeddings
            emb_mat_store_nbr = tf.get_variable(shape=self.get_emb_shape("store_nbr"), dtype=tf.float32,
                                                name="emb_mat_store_nbr")
            emb_mat_city = tf.get_variable(shape=self.get_emb_shape("city"), dtype=tf.float32, name="emb_mat_city")
            emb_mat_state = tf.get_variable(shape=self.get_emb_shape("state"), dtype=tf.float32, name="emb_mat_state")
            emb_mat_store_type = tf.get_variable(shape=self.get_emb_shape("store_type"), dtype=tf.float32,
                                                 name="emb_mat_store_type")
            emb_mat_store_cluster = tf.get_variable(shape=self.get_emb_shape("store_cluster"), dtype=tf.float32,
                                                    name="emb_mat_store_cluster")
            emb_mat_item_family = tf.get_variable(shape=self.get_emb_shape("item_family"), dtype=tf.float32,
                                                  name="emb_mat_item_family")
            emb_mat_item_class = tf.get_variable(shape=self.get_emb_shape("item_class"), dtype=tf.float32,
                                                 name="emb_mat_item_class")
            emb_mat_item_nbr = tf.get_variable(shape=self.get_emb_shape("item_nbr"), dtype=tf.float32,
                                               name="emb_mat_item_nbr")
            emb_mat_holiday_type = tf.get_variable(shape=self.get_emb_shape("holiday_type"), dtype=tf.float32,
                                                   name="emb_mat_holiday_type")

            emb_store_nbr = tf.nn.embedding_lookup(emb_mat_store_nbr, self.placeholders.store_nbr,
                                                   name="emb_lookup_store_nbr")
            emb_city = tf.nn.embedding_lookup(emb_mat_city, self.placeholders.city, name="emb_lookup_city")
            emb_state = tf.nn.embedding_lookup(emb_mat_state, self.placeholders.state, name="emb_lookup_state")
            emb_store_type = tf.nn.embedding_lookup(emb_mat_store_type, self.placeholders.store_type,
                                                    name="emb_lookup_store_type")
            emb_store_cluster = tf.nn.embedding_lookup(emb_mat_store_cluster, self.placeholders.store_cluster,
                                                       name="emb_lookup_store_cluster")
            emb_item_family = tf.nn.embedding_lookup(emb_mat_item_family, self.placeholders.item_family,
                                                     name="emb_lookup_item_family")
            emb_item_class = tf.nn.embedding_lookup(emb_mat_item_class, self.placeholders.item_class,
                                                    name="emb_lookup_item_class")
            emb_item_nbr = tf.nn.embedding_lookup(emb_mat_item_nbr, self.placeholders.item_nbr,
                                                  name="emb_lookup_item_nbr")
            emb_national_holiday_type = tf.nn.embedding_lookup(emb_mat_holiday_type,
                                                               self.placeholders.national_holiday_type[:, :, 0],
                                                               name="emb_lookup_national_holiday_type")
            emb_local_holiday_type = tf.nn.embedding_lookup(emb_mat_holiday_type,
                                                            self.placeholders.local_holiday_type[:, :, 0],
                                                            name="emb_lookup_local_holiday_type")

            
            future_data_norm = BatchNorm(name="bn_future")(tf.contrib.layers.flatten(tf.concat(
                                                              [self.placeholders.local_holiday_fut,
                                                               self.placeholders.national_holiday_fut,
                                                               self.placeholders.regional_holiday_fut,
                                                               self.placeholders.year_fut,
                                                               self.placeholders.month_fut,
                                                               self.placeholders.day_fut,
                                                               self.placeholders.dow_fut,
                                                               self.placeholders.onpromotion_fut], axis=2)),
                                                           train=self.placeholders.is_train)


            # Data preparation
            static_data_norm = BatchNorm(name="bn_static")(tf.expand_dims(self.placeholders.item_perishable, 1),
                                                           train=self.placeholders.is_train)

            temporal_data_norm = BatchNorm(name="bn_temporal")(tf.concat([self.placeholders.onpromotion,
                                                                          self.placeholders.national_holiday_transferred,
                                                                          self.placeholders.national_holiday,
                                                                          self.placeholders.regional_holiday,
                                                                          self.placeholders.local_holiday_transferred,
                                                                          self.placeholders.local_holiday,
                                                                          self.placeholders.dcoilwtico,
                                                                          self.placeholders.transactions,
                                                                          self.placeholders.year,
                                                                          self.placeholders.month,
                                                                          self.placeholders.day,
                                                                          self.placeholders.dow], axis=2,
                                                                         name="temporal_data_norm"),
                                                               train=self.placeholders.is_train)

            static_data = tf.concat([static_data_norm,
                                     emb_store_nbr,
                                     emb_item_nbr,
                                     emb_item_family,
                                     emb_item_class,
                                     emb_city,
                                     emb_state,
                                     emb_store_type,
                                     emb_store_cluster,
                                     future_data_norm], axis=1)

            temporal_data = tf.concat([self.placeholders.unit_sales,
                                       temporal_data_norm,
                                       emb_national_holiday_type,
                                       emb_local_holiday_type], axis=2, name="temporal_data")

            # Encoder
            recurrent_cell_encoder = tf.contrib.rnn.CompiledWrapper(
                tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self.n_recurrent_cells),
                                             tf.nn.rnn_cell.LSTMCell(self.n_recurrent_cells),
                                             tf.nn.rnn_cell.LSTMCell(self.n_recurrent_cells)]))
            _, states = tf.nn.dynamic_rnn(recurrent_cell_encoder, temporal_data, dtype=tf.float32)

            # Thought treatment
            states = tf.concat(flatten([[s.c for s in states], [s.h for s in states], [static_data]]), axis=1)
            states = BatchNorm(name="thought_1")(states, train=self.placeholders.is_train)
            states = tf.layers.dense(inputs=states, units=1024, activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(), name="d_thought_1")
            states = BatchNorm(name="thought_2")(states, train=self.placeholders.is_train)
            states = tf.layers.dense(inputs=states, units=1024, activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(), name="d_thought_2")
            states = BatchNorm(name="thought_3")(states, train=self.placeholders.is_train)
            states = tf.layers.dense(inputs=states, units=self.n_recurrent_cells * self.n_recurrent_layers * 2,
                                     activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
            thought_vector = []
            for i in range(self.n_recurrent_layers):
                c = states[:, i * self.n_recurrent_cells:(i + 1) * self.n_recurrent_cells]
                h = states[:, (i + self.n_recurrent_layers) * self.n_recurrent_cells:(i + self.n_recurrent_layers + 1) * self.n_recurrent_cells]
                thought_vector.append(tf.nn.rnn_cell.LSTMStateTuple(c, h))
            thought_vector = tuple(thought_vector)

            # Decoder
            recurrent_cell_decoder = tf.contrib.rnn.CompiledWrapper(
                tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self.n_recurrent_cells),
                                             tf.nn.rnn_cell.LSTMCell(self.n_recurrent_cells),
                                             tf.nn.rnn_cell.LSTMCell(self.n_recurrent_cells)]))

            go = tf.ones([tf.shape(self.placeholders.unit_sales)[0], self.n_timesteps_future, self.n_recurrent_cells])
            outputs, states = decoder(inputs=go, thought_states=thought_vector, cell=recurrent_cell_decoder,
                                      max_ouput_sequence_length=self.n_timesteps_future, name="decoder")

            lstm_stacked_output = tf.reshape(outputs, shape=[-1, outputs.shape[2].value], name="stack_LSTM")
            d = tf.layers.dense(lstm_stacked_output, 64, activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(), name="dense_1")
            d = tf.contrib.layers.layer_norm(d)
            d = tf.layers.dense(d, 32, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="dense_2")
            d = tf.contrib.layers.layer_norm(d)
            d = tf.layers.dense(d, 1, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="dense_3")
            unstacked_output = tf.reshape(d, shape=[-1, self.n_timesteps_future, 1], name="unstack_LSTM")

            return {"output": unstacked_output}

    def define_losses(self):
        with tf.variable_scope("Losses"):
            mae = tf.reduce_mean(tf.abs(self.core_model.output - self.placeholders.target))
            return {"loss": mae}

    def define_optimizers(self):
        with tf.variable_scope("Optimization"):
            op = self.optimizer.minimize(self.losses.loss)
            return {"op": op}

    def define_summaries(self):
        with tf.variable_scope("Summaries"):
            train_final_scalar_probes = {"loss": tf.squeeze(self.losses.loss)}
            final_performance_scalar = [tf.summary.scalar(k, tf.reduce_mean(v), family=self.name)
                                        for k, v in train_final_scalar_probes.items()]
            dev_scalar_probes = {"loss_dev": self.placeholders.loss_dev,
                                 "mape_dev": self.placeholders.mape_dev}
            dev_performance_scalar = [tf.summary.scalar(k, v, family=self.name) for k, v in dev_scalar_probes.items()]
            dev_performance_scalar = tf.summary.merge(dev_performance_scalar)
            return {"scalar_train_performance": tf.summary.merge(final_performance_scalar),
                    "scalar_dev_performance": dev_performance_scalar}
