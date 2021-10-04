# Multistream Classification

Classification (class label prediction) over two non-stationary data streams, one with labeled data (source) and the other with unlabeled data (target). Covariate shift is assumed between the source and target streams.

The problem is to predict the class label data on target stream using labeled data from the source stream, both of which can have concept drift asynchronously. More details in the publication at [CIKM 2016](http://www.utdallas.edu/~swarup.chandra/papers/multistream_cikm16.pdf)

# Environment

1. Java code for change point detection is based from [this](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12335/11786) paper.
2. We use the instance weighted libSVM code from [here](https://www.csie.ntu.edu.tw/~cjlin/libsvm/).
3. config.properties file specifies data path and other configurable items.
4. Python v2.7

# Execution
```
$ python multistream.py <dataset_name>
```
