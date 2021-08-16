#%%
from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
from ydata_synthetic.preprocessing.timeseries import processed_stock
from ydata_synthetic.synthesizers.timeseries import TimeGAN
from sklearn.preprocessing import MinMaxScaler
from multiprocessing.pool import Pool

#%%
def train_data(state,df):
    x = np.array(df[df.states==state].Lst_price).reshape(-1,1)
    x = (x[1:,:] - x[:-1,:])/x[:-1,:]
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    seq_length = 24

    # Build dataset
    dataX = []

    # Cut data by sequence length
    for i in range(0, len(x) - seq_length):
        _x = x[i:i + seq_length]
        dataX.append(_x)

    # Mix Data (to make it similar to i.i.d)
    idx = np.random.permutation(len(dataX))

    outputX = []
    for i in range(len(dataX)):
        outputX.append(dataX[idx[i]])

    stock_data = outputX #processed_stock(seq_len=seq_len)
    print(len(stock_data),stock_data[0].shape)
    return stock_data

def gen_data(stock_data):
    #Specific to TimeGANs
    seq_len=24
    n_seq = 1
    hidden_dim=24
    gamma=1

    noise_dim = 32
    dim = 128
    batch_size = 128

    log_step = 100
    learning_rate = 5e-4

    gan_args = [batch_size, learning_rate, noise_dim, 24, 2, (0, 1), dim]
    synth = TimeGAN(model_parameters=gan_args, hidden_dim=24, seq_len=seq_len, n_seq=n_seq, gamma=1)
    synth.train(stock_data, train_steps=2000)
    synth_data = synth.sample(len(stock_data))
    print(synth_data.shape)
    return synth_data

def main(csvname):
    #
    df = pd.read_csv(csvname,delimiter = ",")
    df = df[['states','Lst_price']]
    n_ = len(df.states.unique())
    datalst = []
    for i in range(n_):
        datalst.append(train_data(i,df))
    pool = Pool(processes=n_)
    result = pool.map_async(gen_data,datalst).get()
    pool.close()
    pool.join()
    
    fig_name = csvname.split('.')[0]
    cols = ['state {}'.format(i) for i in range(n_)]
    with pd.ExcelWriter(r"{}_states_results.xlsx".format(fig_name),engine='openpyxl') as writer:
        for i,re in enumerate(result):
            new_re = np.array(re).reshape(len(re),-1)
            new_re = pd.DataFrame(new_re)
            new_re.to_excel(writer,sheet_name=cols[i],header=False,index=False)
        writer.save()

    fig, axes = plt.subplots(nrows=n_, ncols=1, figsize=(10, 12))
    axes=axes.flatten()

    time = list(range(1,25))
    obs = [np.random.randint(len(i)) for i in datalst]

    for j in range(n_):
        df = pd.DataFrame({'Real': datalst[j][obs[j]][:, 0],
                           'Synthetic': result[j][obs[j]][:, 0]})
        df.plot(ax=axes[j],
                    title = cols[j],
                    secondary_y='Synthetic data', style=['-', '--'])
    fig.tight_layout()
    fig.savefig('{}.jpg'.format(fig_name),dpi=400)
    
    return 0

if __name__ == '__main__':
    main('aapl_states.csv')
    main('csco_states.csv')
    main('tnx_states.csv')
    main('hg1_states.csv')
    main('gbpusd_states.csv')
    
    