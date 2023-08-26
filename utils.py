import numpy as np
import pandas as pd
import torch

def split_sequences(sequences, n_steps_in, n_steps_out, n_history, historic_sequences=None):
    """transforms target sequence table to historic/real-time/target sequence features)

    Args:
        sequences (np.array): 2-dimensional target feature array (stations on axis0, timestep on axis1)
        n_steps_in (int): length of real-time sequence feature
        n_steps_out (int): length of target sequence
        n_history (int): length of historic sequence

    Returns:
        _type_: _description_
    """
    size = sequences.shape[1]
    rs = np.empty((0,n_steps_in))
    hs = np.empty((0, n_steps_out, n_history))
    ys = np.empty((0,n_steps_out))

    if historic_sequences is None:
        historic_sequences = sequences

    for idx in range(n_history * 504 - n_steps_in, size - (n_steps_in + n_steps_out)):
        r = sequences[:,idx:idx+n_steps_in]
        rs = np.vstack([rs, r])
        
        h = historic_sequences[:, [idx + n_steps_in + out_step - 504*hist_step for out_step in range(n_steps_out) for hist_step in range(n_history, 0, -1)]]
        h = h.reshape(-1,n_steps_out, n_history)
        hs = np.vstack([hs, h])

        y = sequences[:, idx+n_steps_in:idx+n_steps_in+n_steps_out]
        ys = np.vstack([ys, y])

    return rs, hs, ys


def time_features(time_idx, n_steps_in, n_steps_out, n_history, n_stations):
    df = pd.DataFrame(data=pd.to_datetime(time_idx), columns=['time'])
    df['t_index']  = df['time'].dt.hour.multiply(60).add(df['time'].dt.minute).floordiv(20)
    df['dow'] = df['time'].dt.dayofweek
    df['weekend'] = df.dow.isin([5,6]).astype(np.int64)
    df = df[['t_index', 'dow', 'weekend']]

    ts = np.empty((0,n_steps_out,3))
    for idx in range(n_history * 504 - n_steps_in, len(time_idx) - (n_steps_in + n_steps_out)):
        t = df.values[np.newaxis, idx+n_steps_in:idx+n_steps_in+n_steps_out, :]
        ts = np.vstack([ts, t])

    return np.repeat(ts, n_stations, axis=0)


def station_features(station_array, station_df, n_windows, drop_id=False):
    df = pd.DataFrame(data=station_array, columns=['sid']).merge(station_df, how='left', on='sid')

    if drop_id:
        df = df.drop(columns=['sid'])
    return np.tile(df.values, (n_windows,1))


def linear_split(*arrays, n_station, train_frac=0.9):
    data_length = arrays[0].shape[0]
    n_train = int((data_length // n_station) * train_frac) * n_station
    return [arr[:n_train] for arr in arrays], [arr[n_train:] for arr in arrays]


def stationwise_split(*arrays, n_station, train_frac=0.9):
    np.random.seed(42)  # fix random seed
    station_idx = np.random.choice(n_station, int(n_station * train_frac), replace=False)
    data_length = arrays[0].shape[0]
    n_window = data_length // n_station
    train_idx = station_idx[:, np.newaxis] + [i*n_station for i in range(n_window)]
    train_idx = np.sort(train_idx.flatten())
    valid_idx = sorted(set(range(data_length)).difference(train_idx))
    return [arr[train_idx] for arr in arrays], [arr[valid_idx] for arr in arrays]


class EvcFeatureGenerator():
    def __init__(self, sequences, station_attributes, station_embeddings):
        self.realtime_sequences = None
        self.historic_sequences = None
        self.station_attributes = station_attributes  # DataFrame
        self.station_embeddings = station_embeddings  # 향후 임베딩이 주어지는 것이 아니라, train셋에 피팅된 transforming model로부터 생성되도록변경

        sid_encoder = {name:idx for idx, name in enumerate(self.station_embeddings.sid)}
        self.station_embeddings.sid = self.station_embeddings.sid.map(sid_encoder)
        self.station_attributes.sid = self.station_attributes.sid.map(sid_encoder)
        self.umap_embedding_vectors = torch.tensor(self.station_embeddings.drop(columns=['sid']).values).float()

        sequences = sequences.set_index('time').T.reset_index().rename(columns={'index':'sid'})
        sequences.sid = sequences.sid.map(sid_encoder)
        self.realtime_sequences = sequences[sequences.sid.isin(station_attributes.sid)].set_index('sid')  # station feature가 있는 데이터로 한정

    @property
    def n_stations(self):
        return self.realtime_sequences.shape[0]

    def historic_seq_smoothing(self):
        # smoothing
        historic_data = self.realtime_sequences.T
        historic_data.index = pd.to_datetime(historic_data.index)
        historic_data = historic_data.resample(rule='1h').mean().T  # 1시간 단위 smoothing
        self.historic_sequences = historic_data

    def discretize_sequences(self):
        # binary mode만 우선 구현 (1: high avaliability, 0: low availability)
        self.realtime_sequences = self.realtime_sequences.mask(lambda x: x < 0.5,  1).mask(lambda x: x != 1, 0)
        if self.historic_sequences is not None:
            self.historic_sequences = self.historic_sequences.mask(lambda x: x < 0.5,  1).mask(lambda x: x != 1, 0)

    def slice_data(self, prob=0.9):
        get_idx = self.realtime_sequences.mean(axis=1).le(prob)  # 평균 availablity가 일정 확률 이하인 station만 선택
        self.realtime_sequences = self.realtime_sequences.loc[get_idx]
        if self.historic_sequences is not None:
            self.historic_sequences = self.historic_sequences.loc[get_idx]

    def generate_features(self, n_in, n_out, n_hist, pred_step=1):
        print('generating features...')
        n_windows = self.realtime_sequences.shape[1] - (n_out + 504*n_hist)  # 504 -> window size 30분 기준임 (7 * 24 * (60//30))
        R_seq, H_seq, Y_seq = split_sequences(sequences=self.realtime_sequences.values, 
                                              n_steps_in=n_in, n_steps_out=n_out, n_history=n_hist, 
                                              historic_sequences=np.repeat(self.historic_sequences.values, 3, 1) \
                                                  if self.historic_sequences is not None else None)
        T = time_features(time_idx=self.realtime_sequences.columns, n_steps_in=n_in, n_steps_out=n_out, n_history=n_hist, n_stations=self.n_stations)
        S = station_features(station_array=self.realtime_sequences.index, station_df=self.station_attributes, n_windows=n_windows) 

        R_seq = R_seq[:, :, np.newaxis]
        H_seq = H_seq[:, pred_step-1, :, np.newaxis]
        T = T[:,pred_step-1,:]
        Y = Y_seq[:,pred_step-1, np.newaxis]
        print('done!')

        return R_seq, H_seq, T, S, Y


if __name__ == '__main__':
    # # check split sequence function
    # sequences = pd.read_csv('./data/input_table/history_by_station_pub.csv', parse_dates=['time'])
    # station_attributes = pd.read_csv('./data/input_table/pubstation_feature_scaled.csv')
    # station_embeddings = pd.read_csv('./data/input_table/pubstation_umap-embedding.csv')

    # feature_generator = EvcFeatureGenerator(sequences, station_attributes, station_embeddings)
    # feature_generator.historic_seq_smoothing()  # smoothing 적용
    # feature_generator.discretize_sequences()  # 이산화 (binary)
    # feature_generator.slice_data(prob=0.9)  # mean availability 0.9 이하 선택

    # R_seq, H_seq, T, S, Y = feature_generator.generate_features(n_in=12, n_out=6, n_hist=4, pred_step=1)
    # print(R_seq.shape, H_seq.shape, T.shape, S.shape, Y.shape, end='\n')


    # check split sequence function
    sequences = pd.read_csv('./data/input_table/by_test_frac/0.9/train_sequences_multicls_size30_bin3.csv', parse_dates=['time'])
    print(sequences.shape)
    station_attributes = pd.read_csv('./data/input_table/by_test_frac/0.9/station_attributes.csv')
    station_embeddings = pd.read_csv('./data/input_table/by_test_frac/0.9/station_embedding.csv')

    feature_generator = EvcFeatureGenerator(sequences, station_attributes, station_embeddings)
    print(feature_generator.realtime_sequences.shape)
    # feature_generator.historic_seq_smoothing()  # smoothing 적용
    # feature_generator.discretize_sequences()  # 이산화 (binary)
    # feature_generator.slice_data(prob=0.9)  # mean availability 0.9 이하 선택

    R_seq, H_seq, T, S, Y = feature_generator.generate_features(n_in=12, n_out=6, n_hist=4, pred_step=1)
    print(R_seq.shape, H_seq.shape, T.shape, S.shape, Y.shape, end='\n')
