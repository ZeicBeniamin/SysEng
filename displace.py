import obspy
import h5py
from obspy import UTCDateTime
import numpy as np
from obspy.clients.fdsn.client import Client
import matplotlib.pyplot as plt
import pandas as pd

n = 4

file_name = f"../chunk{n}/chunk{n}.hdf5"
csv_file = f"../chunk{n}/chunk{n}.csv"

traces_list = [
    "GO05.C_20110806162011_EV", ##"3.7"
    "GO05.C_20110830234130_EV", #"4.6"
    "GO05.C_20110831230454_EV", #"3.6"
    "GO05.C_20110920232612_EV", #"4.1"
    "GO05.C_20110921221656_EV", #"3.8"
    # "GO05.C_20110921221657_EV", #"3.8"
    "GO05.C_20110922000302_EV", #"3.7"
    "GO05.C_20111011001103_EV", #"4.5"
    # "GO05.C_20111011001108_EV", #"4.5"
    # "GO05.C_20111031214509_EV" #"2.8"
]

def make_stream(dataset):
    '''
    input: hdf5 dataset
    output: obspy stream

    '''
    data = np.array(dataset)

    tr_E = obspy.Trace(data=data[:, 0])
    tr_E.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_E.stats.delta = 0.01
    tr_E.stats.channel = dataset.attrs['receiver_type']+'E'
    tr_E.stats.station = dataset.attrs['receiver_code']
    tr_E.stats.network = dataset.attrs['network_code']

    tr_N = obspy.Trace(data=data[:, 1])
    tr_N.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_N.stats.delta = 0.01
    tr_N.stats.channel = dataset.attrs['receiver_type']+'N'
    tr_N.stats.station = dataset.attrs['receiver_code']
    tr_N.stats.network = dataset.attrs['network_code']

    tr_Z = obspy.Trace(data=data[:, 2])
    tr_Z.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_Z.stats.delta = 0.01
    tr_Z.stats.channel = dataset.attrs['receiver_type']+'Z'
    tr_Z.stats.station = dataset.attrs['receiver_code']
    tr_Z.stats.network = dataset.attrs['network_code']

    stream = obspy.Stream([tr_E, tr_N, tr_Z])

    return stream

def get_noise(count, all=False):
    n = 1
    noise_csv_file = f"../chunk{n}/chunk{n}.csv"
    noise_file = f"../chunk{n}/chunk{n}.hdf5"

    # reading the csv file into a dataframe:
    df = pd.read_csv(noise_csv_file)
    print(f'total events in csv file: {len(df)}')
    # filterering the dataframe
    df = df[(df.trace_category == 'noise') & (df.receiver_code == 'GO05') ]
    print(f'total events selected: {len(df)}')

    # making a list of trace names for the selected data
    ev_list = df['trace_name'].to_list()[:200]

    # retrieving selected waveforms from the hdf5 file: 
    dtfl = h5py.File(noise_file, 'r')
    if count > len(ev_list):
        raise Exception(f"Noise array only has {len(ev_list)} noise samples")

    for c, evi in enumerate(ev_list):

        if c != count:
            continue
        if c > count:
            break

        dataset = dtfl.get('data/'+str(evi)) 
        # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel 
        
        print(f"Processing {evi} with index {c}")

        client = Client("IRIS")
        inventory = client.get_stations(network=dataset.attrs['network_code'],
                                        station=dataset.attrs['receiver_code'],
                                        starttime=UTCDateTime(dataset.attrs['trace_start_time']),
                                        endtime=UTCDateTime(dataset.attrs['trace_start_time']) + 120,
                                        loc="*", 
                                        channel="*",
                                        level="response")
        
        # converting into displacement
        st = make_stream(dataset)
        st = st.remove_response(inventory=inventory, output="DISP", plot=False)

        return st[2]
    

def make_plot(tr, title='', ylab=''):
    '''
    input: trace
    
    '''
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(tr.times("matplotlib"), tr.data, "k-")
    ax.xaxis_date()
    fig.autofmt_xdate()
    plt.ylabel(ylab)
    plt.title(title)
    plt.show()


def check_data(a, b, title='', ylab=''):
    '''
    input: trace
    
    '''
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(a, b, "k-")
    # ax.xaxis_date()
    # fig.autofmt_xdate()
    plt.ylabel(ylab)
    plt.title(title)
    plt.show()
    
    
if __name__ == '__main__': 

    time = []
    dip = []

    # reading one sample trace from STEAD
    dtfl = h5py.File(file_name, 'r')
    for trace in traces_list:
        dataset = dtfl.get(f'data/{trace}') 

        print(f"Processing trace {trace}")
        # convering hdf5 dataset into obspy sream
        st = make_stream(dataset)
        
        # ploting the verical component of the raw data
        # make_plot(st[2], title='Raw Data', ylab='counts')

        # downloading the instrument response of the station from IRIS
        client = Client("IRIS")
        inventory = client.get_stations(network=dataset.attrs['network_code'],
                                        station=dataset.attrs['receiver_code'],
                                        starttime=UTCDateTime(dataset.attrs['trace_start_time']),
                                        endtime=UTCDateTime(dataset.attrs['trace_start_time']) + 120,
                                        loc="*", 
                                        channel="*",
                                        level="response")  

        # converting into displacement
        st = make_stream(dataset)
        st = st.remove_response(inventory=inventory, output="DISP", plot=False)

        # ploting the verical component
        # make_plot(st[2], title='Displacement', ylab='meters')
        tr = st[2]

        starttime = UTCDateTime(dataset.attrs['trace_start_time'])
        time.append(tr.times() + starttime.timestamp)
        dip.append(tr.data)

    npdata = np.array([],dtype=np.float64)
    
    a = get_noise(5)
    b = get_noise(6)
    c = get_noise(7)
    d = get_noise(8)
    e = get_noise(9)
    # get_noise(10),
    # get_noise(11),
    # get_noise(1),
    # get_noise(2),
    # get_noise(3),
    # get_noise(4),

    noise = [
        a,
        b,
        e,
        a,
        b,
        a,
        d,
        c,
        d,
        e,
        e,
        a,
        d,
        a    
    ]
    
    npdata = np.concatenate( (
        noise[1],
        dip[3],
        noise[2],
        noise[3],
        dip[5],
        noise[4],
        noise[5],
        dip[2],
        dip[4],
        noise[6],
        noise[7],
        noise[3],
        noise[1],
        noise[1],
        noise[8],
        noise[9],
        noise[10],
        dip[6],
        noise[1],
        noise[5],
        noise[3],
    ))

    np.save(file="./quake2.npy", arr=npdata, allow_pickle=False)

    print("Bye")
    print("")