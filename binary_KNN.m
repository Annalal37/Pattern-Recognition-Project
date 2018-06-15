clear all
clc
load('spamData');
%binarize data from Xtrain to XbinomialTrain     
XbinomialTrain = Xtrain;
XbinomialTrain(Xtrain ~= 0) = 1;
XbinomialTrain(Xtrain == 0) = 0;
%binarize data from Xtest to XbinomialTrain 
XbinomialTest = Xtest;
XbinomialTest(Xtest ~= 0) = 1;
XbinomialTest(Xtest == 0) = 0;

%count the Hamming distances mapping from each feature in each row of Xtest data to each feature in each row of Xtrain data 
for h = 1:1536
    for i = 1:3065
        distance_test(i,h) = 0;
        for j = 1:57
        distance_test(i,h) = distance_test(i,h)+abs(XbinomialTest(h,j)-XbinomialTrain(i,j));
        end
    end
 end
[num_test,val_test] = sort(distance_test);%sort the distances and store their number and label sequences using matrices
%count the Hamming distances mapping from each feature in each row of Xtrain data to each feature in each row of Xtrain data 
 for h = 1:3065
    for i = 1:3065
        distance_train(i,h) = 0;
        for j = 1:57
        distance_train(i,h) = distance_train(i,h)+(XbinomialTrain(h,j)-XbinomialTrain(i,j)).^2;
        end
    end
 end 
 [num_train,val_train] = sort(distance_train);
 
%caculate test error rates with different 'k' value in KNN algorithm.
 for K = [1:1:10,15:5:100]
    test_error_count(K)= 0;
     for j = 1:1536
         K0(j) = 0;%count the number of Ytrain=0.
         K1(j) = 0;%count the number of Ytrain=1.
         for i = 1:K
             if(ytrain(val_test(i,j),1)==0)
                 K0(j) = K0(j)+1;
             else
                 K1(j) = K1(j)+1;
             end
         end
         if(K0(j) >= K1(j))%classify the Xtest data to have actual ytest results.
             yactual_test(j,1) = 0;
         else
             yactual_test(j,1) = 1;
         end
         if(yactual_test(j,1) ~= ytest(j,1))
             test_error_count(K) = test_error_count(K) + 1;
         end
         test_error_rate(K) = test_error_count(K)/1536;
     end
 end
%caculate train error rates with different 'k' value in KNN algorithm.
for K = [1:1:10,15:5:100]
    train_error_count(K)= 0;
     for j = 1:3065
         K0(j) = 0;
         K1(j) = 0;
         for i = 1:K
             if(ytrain(val_train(i,j),1)==0)
                 K0(j) = K0(j)+1;
             else
                 K1(j) = K1(j)+1;
             end
         end
         if(K0(j) >= K1(j))
             yactual_train(j,1) = 0;
         else
             yactual_train(j,1) = 1;
         end
         if(yactual_train(j,1) ~= ytrain(j,1))
             train_error_count(K) = train_error_count(K) + 1;
         end
         train_error_rate(K) = train_error_count(K)/3065;
     end
end
%plot two error rates' trend with different k.
plot([1:1:10,15:5:100],test_error_rate([1:1:10,15:5:100]));
hold
plot([1:1:10,15:5:100],train_error_rate([1:1:10,15:5:100]),'r');
axis([1 100 0 0.16]);
title('Relationship between error rate and k through binarization strategy');
xlabel('k');
ylabel('error rate');
legend('test error rate','train error rate');
