ProbFire model input datasets and pre-trained model weights with feature scaling parameters.

Author: Tadas Nikonovas (tadas.nik@gmail.com)

The following datasets were generated as part of research which was published in:
Nikonovas et al. ProbFire: a probabilistic fire early warning system for Indonesia, NHESS, 2021.

ProbFire model source code and description can be found at https://github.com/ToFEWSI/ProbFire

The input features (era5_features.parquet) is stored in binary 'parquet' file format. To read the datasets use Appache Arrow libraries available for C, C++, C#, Go, Java, JavaScript, MATLAB, Python, R, Ruby, and Rust (see https://arrow.apache.org/).

The model weights and scaler parameters are stored as Python objects using the standard Joblib library. File names indicate validation year (data from that year is not used for ProbFire training)
