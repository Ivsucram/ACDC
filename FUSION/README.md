# FUSION
Efficient Multistream Classification using Direct DensIty Ratio Estimation

## Synopsis
Traditional data stream classification assumes that data is generated from a single non-stationary process. On the contrary, multistream classification problem involves two independent non-stationary data generating processes. One of them is the source stream that continuously generates labeled data. The other one is the target stream that generates unlabeled test data from the same domain. The distributions represented by the source stream data is biased compared to that of the target stream. Moreover, these streams may have asynchronous concept drifts between them. The multistream classification problem is to predict the class labels of target stream instances, while utilizing labeled data available on the source stream. This kind of scenario is often observed in real-world applications due to scarcity of labeled data. FUSION provides an efficient solution for multistream classification by fusing drift detection into online data adaptation. Theoretical analysis and experiment results show its effectiveness. Please refer to the paper (mentioned in the reference section) for further details. 

## Requirements
FUSION requires that-
* Input file will be provided in a ARFF/CSV format.
* All the features need to be numeric. If there is a non-numeric featues, those can be converted to numeric features using standard techniques.
* Features should be normalized to get better performance. 

## Environment
* Python 2.7
* Scipy, sklearn
* numpy, math

## Execution
To execute the program:
1. First set properties in the config.properties file. Available options have been discussed later in this file.
2. Call the main function in the multistream.py file with two parameters. The first parameter is the path to the dataset file without extension. Extension is automatically appended from the corresponding property in the config.property file. The second parameter is the probability that the next instance will come from the source stream. As an example, the second parameter value 0.1 means that the next instance will come from the source stream with 10% probability and from the target stream with 90% probability. 
 
## Properties:
* baseDir
  * Path to the base directory, which contains the input file(s). This will be appended to the name of the input file for getting the input file path.
* srcfileAppend
  * This string is appeneded after the name of input file supplied as the first parameter to get the file name for the source stream in the baseDir location.
* trgfileAppend
  * This string is appeneded after the name of input file supplied as the first parameter to get the file name for the target stream in the baseDir location.
* useKliepCVSigma
  * 1: Use the cross-validated value for sigma; 0: Use a fixed value for sigma.
* kliepDefSigma
  * In case useKliepCVSigma=0 was used, the value for sigma is specified in this property.
* kliepParEta
  * Value for the parameter Eta.
* kliepParLambda
  * Value for the parameter lambda.
* kliepParB
  * Value for the parameter B.
* kliepParThreshold
  * Value for the threshold used in the change detection algorithm.
* useSvmCVParams
  * If set, find the parameters for SVM using cross-validation.
* svmDefGamma
  * Default value for the gamma parameter in SVM.
* svmDefC
  * Default value for the parameter "C" in SVM.
* kernel
  * Type of kernel used in the svm algorithm.
* cushion
  * The value of cushion for the change detection algorithm if not calculated by gamma.
* sensitivity
  * Sensitivity of the change detection algorithm.
* maxWindowSize
  * Size of the source and target sliding window.
* initialDataSize
  * Size of the initial/warm-up training data.
* enableForceUpdate
  * If set, update the classifier after a long period of time even if there is no change detected.
* forceUpdatePeriod
  * If enableForceUpdate is set, the classifier is updated after this many instances even if there is no change detected.
* ensemble_size
  * Size of the ensemble.
* output_file_name
  * Path to the output file.
* logfile
  * Path to the log file.
* tempDir
  * Path to the directory containing all the temporary files.

## Output
### Console output
* The program shows progress or any change point detected in console. 
* At the end, it reports the overall accuracy.

### File output
1. A log file is generated in the location specified by "logfile" property, which contains important debug information.
2. The output file contains the running average accuracy.

## Reference
[FUSION: An Online Method for Multistream Classification](https://dl.acm.org/citation.cfm?id=3132886&dl=ACM&coll=DL&CFID=1020200191&CFTOKEN=12773057)
