clear all
clc
load('spamData');
%log-transform data
XlogTrain = log(Xtrain+0.1);
XlogTest = log(Xtest+0.1);
%count the Euclidean distances mapping from each feature in each row of Xtest data to each feature in each row of Xtrain data 
for h = 1:1536
    for i = 1:3065
        distance_test(i,h) = 0;
        for j = 1:57
        distance_test(i,h) = distance_test(i,h)+(XlogTest(h,j)-XlogTrain(i,j)).^2;
        end
    end
 end
[num_test,val_test] = sort(distance_test);%sort the distances and store their number and label sequences using matrices
%count the Euclidean distances mapping from each feature in each row of Xtrain data to each feature in each row of Xtrain data 
 for h = 1:3065
    for i = 1:3065
        distance_train(i,h) = 0;
        for j = 1:57
        distance_train(i,h) = distance_train(i,h)+(XlogTrain(h,j)-XlogTrain(i,j)).^2;
        end
    end
 end 
 [num_train,val_train] = sort(distance_train);
 
%caculate test error rates with different 'k' value in KNN algorithm.
 for K = [1:1:10,15:5:100]
    test_error_count(K)= 0;
     for j = 1:1536
         K0(j) = 0;
         K1(j) = 0;
         for i = 1:K
             if(ytrain(val_test(i,j),1)==0)
                 K0(j) = K0(j)+1;
             else
                 K1(j) = K1(j)+1;
             end
         end
         if(K0(j) >= K1(j))
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
axis([1 100 0 0.1]);
title('Relationship between error rate and k through log-transformed strategy');
xlabel('k');
ylabel('error rate');
legend('test error rate','train error rate');