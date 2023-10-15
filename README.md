# Convolution and Cross-Correlation of Count Sketches Enables Fast Cardinality Estimation of Multi-Join Queries

This repository contains the source code, extended results, and cardinality estimates for the experiments of the research paper under review at SIGMOD 2024.

## Requirements

The code is written in Python 3.10. The required packages to run the experiments can be found in `requirements.txt`. To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

### Download the data and queries

The experiments use the IMDB and STATS databases with queries provided by the [End-to-End CardEst Benchmark](https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark). To run the experiments, first download the required data using the following commands:

```bash
curl -L -o End-to-End-CardEst-Benchmark.zip https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark/archive/refs/heads/master.zip
unzip End-to-End-CardEst-Benchmark.zip
rm End-to-End-CardEst-Benchmark.zip

curl -L -o imdb.tgz http://homepages.cwi.nl/~boncz/job/imdb.tgz
mkdir imdb
tar zxvf imdb.tgz -C imdb
rm imdb.tgz
```
This should result in the following file structure:
```
End-to-End-CardEst-Benchmark-master/
imdb/
experiment.py
...
```

## Experiments

Information about the accepted arguments for the experiments can be obtained using the following command:

```bash
python experiment.py --help
```

For example, the following command runs our proposed method with `m=1000000` and takes the median of `l=5` i.i.d. estimates:

```bash
python experiment.py --method count-conv --query stats-7 --bins 1000000 --medians 5
```

### Available queries

The [End-to-End CardEst Benchmark](https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark) provides queries with sub-queries for the STATS and IMDB databases. The following are the available options: `stats-[1-146]`, `stats_sub-[1-2603]`, `job_light-[1-70]`, and `job_light_sub-[1-696]`, where the brackets are inclusive ranges.


### Speed-up data loading

Loading the data from csv files for each experiment can incur significant overhead. To alleviate this, one can cache the loaded tables as pickle files using `python cache_tables.py`. After this finishes, the data loading time during the experiments should be reduced by roughly a factor of 10.


## Extended results

The absolute relative error plots, in addition to the timing plots of each stage (initialization, sketching, and inference) for all 216 queries are provided in [`/figures`](figures).


## Cardinality estimates

The cardinality estimates for all the sub-queries of both the STATS and IMDB databases are provided in [`/estimates`](estimates), which follows the same format as the estimates provided by the [End-to-End CardEst Benchmark](https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark).