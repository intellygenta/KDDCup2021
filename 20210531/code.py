import numpy as np
import pandas as pd
import datetime as dt
import pathlib
import tqdm
import stumpy

# Path setting
rootpath = pathlib.Path('../dataset')
txt_dirpath = rootpath / 'phase2'  # Place the txt files in this directory

# Parameter setting
min_window_size = 40
max_window_size = 800
growth_rate = 1.1
denom_threshold = 0.1
upper_threshold = 0.75
lower_threshold = 0.25
const_threshold = 0.05
min_coef = 0.5
small_quantile = 0.1
padding_length = 3
train_length = 10
use_gpu = True

# Determine window sizes
size = int(np.log(max_window_size / min_window_size) / np.log(growth_rate)) + 1
rates = np.full(size, growth_rate) ** np.arange(size)
ws = (min_window_size * rates).astype(int)

# Select stump function
if use_gpu:
    stump = stumpy.gpu_stump
else:
    stump = stumpy.stump

# Anomaly score names
names = [
    'orig_p2p',
    'diff_p2p',
    'acc_p2p',
    'orig_p2p_inv',
    'diff_small',
    'acc_std',
    'acc_std_inv',
    'orig_mp_novelty',
    'orig_np_novelty',
    'orig_mp_outlier',
    'orig_np_outlier',
]

# Define function for computing anomaly score with window size w
def compute_score(X, number, split, w):
        
    # original time series (orig)
    seq = pd.DataFrame(X, columns=['orig'])
    
    # velocity (diff) and acceleration (acc)
    seq['diff'] = seq['orig'].diff(1)
    seq['acc'] = seq['diff'].diff(1)
        
    # standard deviation (std)
    for name in ['orig', 'acc']:
        seq[f'{name}_std'] = seq[name].rolling(w).std().shift(-w)
    
    # peak-to-peak (p2p)
    for name in ['orig', 'diff', 'acc']:
        rolling_max = seq[name].rolling(w).max()
        rolling_min = seq[name].rolling(w).min()
        seq[f'{name}_p2p'] = (rolling_max - rolling_min).shift(-w)
    
    # diff small
    diff_abs = seq['diff'].abs()
    cond = diff_abs <= diff_abs.quantile(small_quantile)
    seq['diff_small'] = cond.rolling(w).mean().shift(-w)
    
    # inverse (inv)
    for name in ['orig_p2p', 'acc_std']:
        numer = seq[name].mean()
        denom = seq[name].clip(lower=numer * denom_threshold)
        seq[f'{name}_inv'] = numer / denom
    
    # coef for penalizing subsequences with little change
    name = 'orig_p2p'
    mean = seq[name].mean()
    upper = mean * upper_threshold
    lower = mean * lower_threshold
    const = mean * const_threshold
    seq['coef'] = (seq[name] - lower) / (upper - lower)
    seq['coef'].clip(upper=1.0, lower=0.0, inplace=True)
    cond = (seq[name] <= const).rolling(2 * w).max().shift(-w) == 1
    seq.loc[cond, 'coef'] = 0.0
        
    # matrix profile value (mpv) and index (mpi)
    mpv = {}
    mpi = {}
    for mode in ['train', 'join', 'all']:
        if mode == 'train':
            mp = stump(X[:split], w)
        elif mode == 'join':
            mp = stump(X[split:], w, X[:split], ignore_trivial=False)
        elif mode == 'all':
            mp = stump(X, w)
        mpv[mode] = mp[:, 0].astype(float)
        mpi[mode] = mp[:, 1].astype(int)
        
    # matrix profile (mp) and normalized profile (np) for novelty detection (AB-join)
    numer = mpv['join']
    denom = mpv['train'][mpi['join']]
    begin = split
    end = begin + len(numer) - 1
    numer *= seq.loc[begin:end, 'coef'].values
    seq.loc[begin:end, 'orig_mp_novelty'] = numer
    with np.errstate(all='ignore'):
        seq.loc[begin:end, 'orig_np_novelty'] = numer / denom
    seq['orig_np_novelty'].clip(upper=1 / denom_threshold, inplace=True)

    # matrix profile (mp) and normalized profile (np) for outlier detection (self-join)
    numer = mpv['all']
    denom = mpv['all'][mpi['all']]
    begin = 0
    end = begin + len(numer) - 1
    numer *= seq.loc[begin:end, 'coef'].values
    seq.loc[begin:end, 'orig_mp_outlier'] = numer
    with np.errstate(all='ignore'):
        seq.loc[begin:end, 'orig_np_outlier'] = numer / denom
    seq['orig_np_outlier'].clip(upper=1 / denom_threshold, inplace=True)
    
    # Smooth and mask anomaly score
    padding = w * padding_length
    seq['mask'] = 0.0
    seq.loc[seq.index[w:-w-padding], 'mask'] = 1.0
    seq['mask'] = seq['mask'].rolling(padding, min_periods=1).sum() / padding
    for name in names:
        seq[f'{name}_score'] = seq[name].rolling(w).mean() * seq['mask']
    
    return seq

# Evaluate anomaly score for each time series
results = []
for txt_filepath in sorted(txt_dirpath.iterdir()):
    
    # Load time series
    X = np.loadtxt(txt_filepath)
    number = txt_filepath.stem.split('_')[0]
    split = int(txt_filepath.stem.split('_')[-1])
    print(f'\n{txt_filepath.name} {split}/{len(X)}', flush=True)
    
    # Evaluate anomaly score for each window size w
    for w in tqdm.tqdm(ws):
        
        # Skip long subsequence
        if w * train_length > split:
            continue
            
        # Compute anomaly score
        seq = compute_score(X, number, split, w)
        
        # Skip if coef is small
        if seq['coef'].mean() < min_coef:
            continue
            
        # Evaluate anomaly score
        for name in names:
            
            # Copy anomaly score
            y = seq[f'{name}_score'].copy()
            
            # Find local maxima
            cond = (y == y.rolling(w, center=True, min_periods=1).max())
            y.loc[~cond] = np.nan
            
            # Find 1st peak
            index1 = y.idxmax()
            value1 = y.max()
            
            # Skip if all score is NaN
            if not np.isfinite(value1):
                continue
                
            # Skip if train data has 1st peak
            begin = index1 - w
            end = index1 + w
            if begin < split:
                continue

            # Find 2nd peak
            y.iloc[begin:end] = np.nan
            index2 = y.idxmax()
            value2 = y.max()
            
            # Skip if 2nd peak height is zero
            if value2 == 0:
                continue
            
            # Evaluate rate of 1st peak height to 2nd peak height
            rate = value1 / value2
            results.append([number, w, name, rate, begin, end, index1, value1, index2, value2])

# Display results
results = pd.DataFrame(results, columns=['number', 'w', 'name', 'rate', 'begin', 'end', 'index1', 'value1', 'index2', 'value2'])

# Make submission csv
submission = results.loc[results.groupby('number')['rate'].idxmax(), 'index1']
submission.index = np.arange(len(submission)) + 1
submission.name = 'location'
submission.index.name = 'No.'
submission.to_csv('result.csv')
