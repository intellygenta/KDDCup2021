KDD Cup 2021 Multi-dataset Time Series Anomaly Detection 5th place solution
===

This repository provides the code for the [KDD Cup 2021: Multi-dataset Time Series Anomaly Detection](https://compete.hexagon-ml.com/practice/competition/39/) 5th place solution.

Please watch the [video](https://www.youtube.com/watch?v=J_Ebbql9jCo) for a brief explanation of the solution.

## Requirement

* python 3.8+
* numpy
* pandas
* tqdm
* stumpy

If you have a CUDA-capable GPU and have installed CUDA toolkit, you can accelerate the computation.
If not, please turn off the GPU usage option in `code.py`, i.e., `use_gpu = False`.

## Usage

1. Download [dataset](https://compete.hexagon-ml.com/media/data/multi-dataset-time-series-anomaly-detection-39/data.zip) and place the unzipped `*.txt` files in the `dataset/phase2` directory.
2. Run `20210601/code.py` to get the detection result `20210601/result.csv`.

It takes several days to run because it computes matrix profile with different subsequence lengths for each of the 250 time series.

There are two submission codes in this repository.
* `20210531/code.py` is the final submission code, which achieves 217 / 250 = 86.8% accuracy and results in 5th place in the private leader board.
* `20210601/code.py` is another code I tried to submit on the last day (June 1st), however, I failed to submit it because the time zone of the deadline was unclear (The datetime for each submission was recorded and displayed in UTC-7, but the deadline seemed to be UTC+0).
It turns out that it can achieve 218 / 250 = 87.2% accuracy, which is equivalent to the 3rd place.
