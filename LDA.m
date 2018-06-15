clear all
%load data from images and labels respectively
TrainingSample = loadMNISTImages('train-images.idx3-ubyte')';
TestingSample = loadMNISTImages('t10k-images.idx3-ubyte')';
TrainingLabel = loadMNISTLabels('train-labels.idx1-ubyte');
TestingLabel = loadMNISTLabels('t10k-labels.idx1-ubyte');

mean_TrainingSample = mean(TrainingSample); %calculate mean of training sample
%initiate SW and SB
SW = 0; %within-class covariance
SB = 0; %between-class covariance
for i = 1:10 %calculate 10 classes respectively
   n(i,1) =  length(find(TrainingLabel == i-1));
   mu(i,:) = mean(TrainingSample((TrainingLabel == i-1),:))';
   train_stddr = bsxfun(@minus,TrainingSample((TrainingLabel == i-1),:),mu(i,:)); %normalize training data
   S(:,:)= train_stddr'*train_stddr/n(i,1);%cov(TrainingSample)
   P(i,1) = n(i,1)/60000; %coefficient
   SW = SW + P(i,1)*S(:,:); %calculate within-class covariance
   SB = SB + P(i,1)*(bsxfun(@minus,mu(i,:),mean_TrainingSample))'*(bsxfun(@minus,mu(i,:),mean_TrainingSample)); %calculate between-class covariance
end
A = repmat(0.001,[1,size(SW,1)]); %add a very small matrix to make SW can be reversible
B = diag(A);
[V,D] = eig(inv(SW + B)*SB); %calculate the eigenvectors and eigenvalues

%reduce dimension to 2
W_2 = V(:,1:2);
train_LDA_2 = TrainingSample*W_2;
test_LDA_2 = TestingSample*W_2;
mdl_2 = ClassificationKNN.fit(train_LDA_2,TrainingLabel,'NumNeighbors',1);  %apply knn to projected data
predict_label_knn_2=predict(mdl_2, test_LDA_2);
accuracy_knn_2=length(find(predict_label_knn_2 == TestingLabel))/length(TestingLabel)*100;  
%plot 2d data
figure(1);
for i = 1:10
    scatter(train_LDA_2((TrainingLabel == i-1),1),train_LDA_2((TrainingLabel == i-1),2),'o');
    hold on
end
title('distribution of data in two-dimension');

%reduce dimension to 3
W_3 = V(:,1:3);
train_LDA_3 = TrainingSample*W_3;
test_LDA_3 = TestingSample*W_3;
mdl_3 = ClassificationKNN.fit(train_LDA_3,TrainingLabel,'NumNeighbors',1);  %apply knn to projected data
predict_label_knn_3=predict(mdl_3, test_LDA_3);
accuracy_knn_3=length(find(predict_label_knn_3 == TestingLabel))/length(TestingLabel)*100; 
%plot 3d data
figure(2);
for i = 1:10
    scatter3(train_LDA_3((TrainingLabel == i-1),1),train_LDA_3((TrainingLabel == i-1),2),train_LDA_3((TrainingLabel == i-1),3),'o');
    hold on
end
title('distribution of data in three-dimension');

%reduce dimension to 9
W_9 = V(:,1:9);
train_LDA_9 = TrainingSample*W_9;
test_LDA_9 = TestingSample*W_9;
mdl_9 = ClassificationKNN.fit(train_LDA_9,TrainingLabel,'NumNeighbors',1);  %apply knn to projected data
predict_label_knn_9=predict(mdl_9, test_LDA_9);
accuracy_knn_9=length(find(predict_label_knn_9 == TestingLabel))/length(TestingLabel)*100;  

%reduce dimension to 10
W_10 = V(:,1:10);
train_LDA_10 = TrainingSample*W_10;
test_LDA_10 = TestingSample*W_10;
mdl_10 = ClassificationKNN.fit(train_LDA_10,TrainingLabel,'NumNeighbors',1);  %apply knn to projected data
predict_label_knn_10=predict(mdl_10, test_LDA_10);
accuracy_knn_10=length(find(predict_label_knn_10 == TestingLabel))/length(TestingLabel)*100;  