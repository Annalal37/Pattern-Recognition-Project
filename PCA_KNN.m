clear all
%load data: images and labels respectively
TrainingSample = loadMNISTImages('train-images.idx3-ubyte')';
TestingSample = loadMNISTImages('t10k-images.idx3-ubyte')';
TrainingLabel = loadMNISTLabels('train-labels.idx1-ubyte');
TestingLabel = loadMNISTLabels('t10k-labels.idx1-ubyte');

mean_TrainingSample = mean(TrainingSample); %calculate mean of training sample
train_stddr = bsxfun(@minus,TrainingSample,mean_TrainingSample);   %standardize training sample data
test_stddr = bsxfun(@minus,TestingSample,mean_TrainingSample);  %standardize testing sample data
[U,V]= eig(cov(train_stddr));   %calculate the eigenvectors and eigenvalues of training data
eigenvectors = fliplr(U);   %rotate eigenvectors
eigenvalues = sort(diag(V,0),'descend'); %sort eigenvalues according to their significance

train_pca_2 = train_stddr*eigenvectors(:,1:2);  %project training data in the direction of eigenvectors
train_pca_3 = train_stddr*eigenvectors(:,1:3);  %project training data in the direction of eigenvectors

%plot projected data vector in 2d
figure(1);
for i = 1:10
    scatter(train_pca_2((TrainingLabel == i-1),1),train_pca_2((TrainingLabel == i-1),2),'o');
    hold on
end
title('Projected data vector by PCA in 2d plots');

%plot projected data vector in 3d
figure(2);
for i = 1:10
    scatter3(train_pca_3((TrainingLabel == i-1),1),train_pca_3((TrainingLabel == i-1),2),train_pca_3((TrainingLabel == i-1),3),'o');
    hold on
end
title('Projected data vector by PCA in 3d plots');

%plot eigenvectors
figure(3);
subplot(1,3,1)
b1 = reshape(eigenvectors(:,1),28,28);
imshow(b1,[])
title('eigenvector1')
subplot(1,3,2)
b2 = reshape(eigenvectors(:,2),28,28);
imshow(b2,[])
title('eigenvector2')
subplot(1,3,3)
b3 = reshape(eigenvectors(:,3),28,28);
imshow(b3,[])
title('eigenvector3')

%apply knn to projected data in different dimensions
p_d_40=40; %reduce dimension to 40
train_pca_40 = train_stddr*eigenvectors(:,1:p_d_40);  %project training data in the direction of eigenvectors
test_pca_40 = test_stddr*eigenvectors(:,1:p_d_40); %project testing data in the direction of eigenvectors
mdl_40 = ClassificationKNN.fit(train_pca_40,TrainingLabel,'NumNeighbors',1);  %apply knn to projected data
predict_label_knn_40=predict(mdl_40, test_pca_40);
accuracy_knn_40=length(find(predict_label_knn_40 == TestingLabel))/length(TestingLabel)*100; %calculate testing accuracy 

p_d_80=80; %reduce dimension to 80
train_pca_80 = train_stddr*eigenvectors(:,1:p_d_80);  %project training data in the direction of eigenvectors
test_pca_80 = test_stddr*eigenvectors(:,1:p_d_80); %project testing data in the direction of eigenvectors
mdl_80 = ClassificationKNN.fit(train_pca_80,TrainingLabel,'NumNeighbors',1);  %apply knn to projected data
predict_label_knn_80=predict(mdl_80, test_pca_80);
accuracy_knn_80=length(find(predict_label_knn_80 == TestingLabel))/length(TestingLabel)*100;  

p_d_200=200; %reduce dimension to 200
train_pca_200 = train_stddr*eigenvectors(:,1:p_d_200);  %project training data in the direction of eigenvectors
test_pca_200 = test_stddr*eigenvectors(:,1:p_d_200); %project testing data in the direction of eigenvectors
mdl_200 = ClassificationKNN.fit(train_pca_200,TrainingLabel,'NumNeighbors',1);  %apply knn to projected data
predict_label_knn_200=predict(mdl_200, test_pca_200);
accuracy_knn_200=length(find(predict_label_knn_200 == TestingLabel))/length(TestingLabel)*100;  

%calculate energy to choose dimensionality automatically
i = 1;
energy = 0;
while(energy < 0.95)
    i = i+1;
    energy = sum(eigenvalues(1:i))/sum(eigenvalues);
end
train_pca_i = train_stddr*eigenvectors(:,1:i);  %project training data in the direction of eigenvectors
test_pca_i = test_stddr*eigenvectors(:,1:i); %project testing data in the direction of eigenvectors
mdl_i = ClassificationKNN.fit(train_pca_i,TrainingLabel,'NumNeighbors',1);  %apply knn to projected data 
predict_label_knn_i=predict(mdl_i, test_pca_i);
accuracy_knn_i=length(find(predict_label_knn_i == TestingLabel))/length(TestingLabel)*100;  
