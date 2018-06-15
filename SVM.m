clear all
%load data from images and labels respectively
TrainingSample = loadMNISTImages('train-images.idx3-ubyte')';
TestingSample = loadMNISTImages('t10k-images.idx3-ubyte')';
TrainingLabel = loadMNISTLabels('train-labels.idx1-ubyte');
TestingLabel = loadMNISTLabels('t10k-labels.idx1-ubyte');

%use svm to classify raw data with linear_function (different c parameter)
raw_model_N2_t0_h0 = svmtrain(TrainingLabel,TrainingSample,'-c 0.01 -h 0 -t 0' ); %svm model
[raw_predict_label_N2_t0_h0, raw_accuracy_N2_t0_h0, raw_dec_values_N2_t0_h0] =svmpredict(TestingLabel, TestingSample, raw_model_N2_t0_h0); % test the trainingdata
raw_model_N1_t0_h0 = svmtrain(TrainingLabel,TrainingSample,'-c 0.1 -h 0 -t 0' );
[raw_predict_label_N1_t0_h0, raw_accuracy_N1_t0_h0, raw_dec_values_N1_t0_h0] =svmpredict(TestingLabel, TestingSample, raw_model_N1_t0_h0); % test the trainingdata
raw_model_1_t0_h0 = svmtrain(TrainingLabel,TrainingSample,'-c 1 -h 0 -t 0' );
[raw_predict_label_1_t0_h0, raw_accuracy_1_t0_h0, raw_dec_values_1_t0_h0] =svmpredict(TestingLabel, TestingSample, raw_model_1_t0_h0); % test the trainingdata
raw_model_10_t0_h0 = svmtrain(TrainingLabel,TrainingSample,'-c 10 -h 0 -t 0' );
[raw_predict_label_10_t0_h0, raw_accuracy_10_t0_h0, raw_dec_values_10_t0_h0] =svmpredict(TestingLabel, TestingSample, raw_model_10_t0_h0); % test the trainingdata

%apply PCA to reduce data dimension
mean_TrainingSample = mean(TrainingSample); %calculate mean of training sample
train_stddr = bsxfun(@minus,TrainingSample,mean_TrainingSample);   %standardize training sample data
test_stddr = bsxfun(@minus,TestingSample,mean_TrainingSample);  %standardize testing sample data
[U,V]= eig(cov(train_stddr));   %calculate the eigenvectors and eigenvalues of training data
eigenvectors = fliplr(U);   %change its frequency
eigenvalues = sort(diag(V,0),'descend');
p_d_40=40; %reduce dimension to 40
train_pca_40 = train_stddr*eigenvectors(:,1:p_d_40);  %project training data in the direction of eigenvectors
test_pca_40 = test_stddr*eigenvectors(:,1:p_d_40); %project testing data in the direction of eigenvectors

%use svm to classify data reduced dimensions to 40 with linear_function (different c parameter)
pca40_model_N2_t0_h0 = svmtrain(TrainingLabel,train_pca_40,'-c 0.01 -h 0 -t 0' );
[pca40_predict_label_N2_t0_h0, pca40_accuracy_N2_t0_h0, pca40_dec_values_N2_t0_h0] =svmpredict(TestingLabel, test_pca_40, pca40_model_N2_t0_h0); % test the trainingdata
pca40_model_N1_t0_h0 = svmtrain(TrainingLabel,train_pca_40,'-c 0.1 -h 0 -t 0' );
[pca40_predict_label_N1_t0_h0, pca40_accuracy_N1_t0_h0, pca40_dec_values_N1_t0_h0] =svmpredict(TestingLabel, test_pca_40, pca40_model_N1_t0_h0); % test the trainingdata
pca40_model_1_t0_h0 = svmtrain(TrainingLabel,train_pca_40,'-c 1 -h 0 -t 0' );
[pca40_predict_label_1_t0_h0, pca40_accuracy_1_t0_h0, pca40_dec_values_1_t0_h0] =svmpredict(TestingLabel, test_pca_40, pca40_model_1_t0_h0); % test the trainingdata
pca40_model_10_t0_h0 = svmtrain(TrainingLabel,train_pca_40,'-c 10 -h 0 -t 0' );
[pca40_predict_label_10_t0_h0, pca40_accuracy_10_t0_h0, pca40_dec_values_10_t0_h0] =svmpredict(TestingLabel, test_pca_40, pca40_model_10_t0_h0); % test the trainingdata

 
p_d_80=80; %reduce dimension to 80
train_pca_80 = train_stddr*eigenvectors(:,1:p_d_80);  %project training data in the direction of eigenvectors
test_pca_80 = test_stddr*eigenvectors(:,1:p_d_80); %project testing data in the direction of eigenvectors
%use svm to classify data reduced dimensions to 80 with linear_function (different c parameter)
pca80_model_N2_t0_h0 = svmtrain(TrainingLabel,train_pca_80,'-c 0.01 -h 0 -t 0' );
[pca80_predict_label_N2_t0_h0, pca80_accuracy_N2_t0_h0, pca80_dec_values_N2_t0_h0] =svmpredict(TestingLabel, test_pca_80, pca80_model_N2_t0_h0); % test the trainingdata
pca80_model_N1_t0_h0 = svmtrain(TrainingLabel,train_pca_80,'-c 0.1 -h 0 -t 0' );
[pca80_predict_label_N1_t0_h0, pca80_accuracy_N1_t0_h0, pca80_dec_values_N1_t0_h0] =svmpredict(TestingLabel, test_pca_80, pca80_model_N1_t0_h0); % test the trainingdata
pca80_model_1_t0_h0 = svmtrain(TrainingLabel,train_pca_80,'-c 1 -h 0 -t 0' );
[pca80_predict_label_1_t0_h0, pca80_accuracy_1_t0_h0, pca80_dec_values_1_t0_h0] =svmpredict(TestingLabel, test_pca_80, pca80_model_1_t0_h0); % test the trainingdata
pca80_model_10_t0_h0 = svmtrain(TrainingLabel,train_pca_80,'-c 10 -h 0 -t 0' );
[pca80_predict_label_10_t0_h0, pca80_accuracy_10_t0_h0, pca80_dec_values_10_t0_h0] =svmpredict(TestingLabel, test_pca_80, pca80_model_10_t0_h0); % test the trainingdata

p_d_200=200; %reduce dimension to 200
train_pca_200 = train_stddr*eigenvectors(:,1:p_d_200);  %project training data in the direction of eigenvectors
test_pca_200 = test_stddr*eigenvectors(:,1:p_d_200); %project testing data in the direction of eigenvectors
%use svm to classify data reduced dimensions to 200 with linear_function (different c parameter)
pca200_model_N2_t0_h0 = svmtrain(TrainingLabel,train_pca_200,'-c 0.01 -h 0 -t 0' );
[pca200_predict_label_N2_t0_h0, pca200_accuracy_N2_t0_h0, pca200_dec_values_N2_t0_h0] =svmpredict(TestingLabel, test_pca_200, pca200_model_N2_t0_h0); % test the trainingdata
pca200_model_N1_t0_h0 = svmtrain(TrainingLabel,train_pca_200,'-c 0.1 -h 0 -t 0' );
[pca200_predict_label_N1_t0_h0, pca200_accuracy_N1_t0_h0, pca200_dec_values_N1_t0_h0] =svmpredict(TestingLabel, test_pca_200, pca200_model_N1_t0_h0); % test the trainingdata
pca200_model_1_t0_h0 = svmtrain(TrainingLabel,train_pca_200,'-c 1 -h 0 -t 0' );
[pca200_predict_label_1_t0_h0, pca200_accuracy_1_t0_h0, pca200_dec_values_1_t0_h0] =svmpredict(TestingLabel, test_pca_200, pca200_model_1_t0_h0); % test the trainingdata
pca200_model_10_t0_h0 = svmtrain(TrainingLabel,train_pca_200,'-c 10 -h 0 -t 0' );
[pca200_predict_label_10_t0_h0, pca200_accuracy_10_t0_h0, pca200_dec_values_10_t0_h0] =svmpredict(TestingLabel, test_pca_200, pca200_model_10_t0_h0); % test the trainingdata

%use svm to classify data reduced dimensions to 40 with kernel_function (tune different c and g parameters to find the optimal result)
pca40_model_RBF_g2c4 = svmtrain(TrainingLabel,train_pca_40,'-t 2 -g 0.015625 -c 19.3' );
[pca40_predict_label_RBF_g2c4, pca40_accuracy_RBF_g2c4, pca40_dec_values_RBF_g2c4] =svmpredict(TestingLabel, test_pca_40, pca40_model_RBF_g2c4); % test the trainingdata
%above is the optimal parameters for data reduced dimensions to 40, to
%tune other parameters, just change the relative values of c and g 