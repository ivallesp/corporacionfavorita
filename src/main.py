from tqdm import tqdm

from src.architecture import Architecture
from src.common_paths import get_tensorboard_logs_path
from src.data_tools import *
from src.data_tools import preprocess_data
from src.tensorflow_utilities import start_tensorflow_session, get_summary_writer

pd.options.display.max_columns = 100

df, data_cube = preprocess_data()

# Cardinalities
cardinalities = {"date": 1684+1,
                 "store_nbr": 54+1,
                 "item_nbr": 4100+1,
                 "item_family": 33+1,
                 "item_class": 337+1,
                 "city": 22+1,
                 "state": 16+1,
                 "store_type": 5+1,
                 "store_cluster": 17+1,
                 "holiday_type": 7}

embedding_sizes = {"store_nbr": 30,
                   "item_nbr": 100,
                   "item_family": 5,
                   "item_class": 10,
                   "city": 5,
                   "state": 5,
                   "store_type": 3,
                   "store_cluster": 5,
                   "national_holiday_type": 3,
                   "holiday_type": 3}


net = Architecture(n_timesteps_past=n_dates-15, n_timesteps_future=15, cardinalities=cardinalities,
                 embedding_sizes=embedding_sizes, name="cf")

sess = start_tensorflow_session("0")
sess.run(tf.global_variables_initializer())
sw = get_summary_writer(sess, get_tensorboard_logs_path(), "CFavorita", "V2")

batch_size=64
c=0

for epoch in range(1000):
    # TRAIN:
    batcher = get_batcher(data_cube[:160000], batch_size)
    for _batch, params in tqdm(batcher):
        fd = {}
        for key, value in _batch.items():
            fd[getattr(net.placeholders, key)] = value

        _, s = sess.run([net.op.op, net.summaries.scalar_train_performance],
                        feed_dict=fd)
        sw.add_summary(s, c)
        c += 1

    # DEV:
    batcher = get_batcher(data_cube[160000:], batch_size)
    lsd = []
    for _batch, params in batcher:
        fd = {}
        for key, value in _batch.items():
            fd[getattr(net.placeholders, key)] = value

        lsd.append(sess.run([net.losses.loss], feed_dict=fd))
    lsd = np.mean(lsd)

    s = sess.run(net.summ.scalar_dev_performance, feed_dict={net.ph.loss_dev: lsd})
    sw.add_summary(s, c)
