clear all
clc
load('spamData');
%apply z normalization to Xtrain and Xtest data
znorm_Xtrain = zscore(Xtrain);
znorm_Xtest = zscore(Xtest);
%generate two new matrices of Xtrain whose ytrain is 1 and another ytrain is 0 respectively
for j = 1:57
    count_Y1 = 0;%count the number of ytrain=1
    count_Y0 = 0;%count the number of ytrain=0
    for i = 1:3065
        if(ytrain(i,1) == 1)
            count_Y1 = count_Y1+1;
            Y1_XTrain(count_Y1,j) = znorm_Xtrain(i,j);
        else 
            count_Y0 = count_Y0+1;
            Y0_XTrain(count_Y0,j) = znorm_Xtrain(i,j);
        end
    end     
end
P_Y1 = count_Y1/3065;
P_Y0 = count_Y0/3065;
%count the means and standard deviations of two new matrices respectively
mu_Y1_XTrain = mean(Y1_XTrain);
sigma_Y1_XTrain = std(Y1_XTrain);
mu_Y0_XTrain = mean(Y0_XTrain);
sigma_Y0_XTrain = std(Y0_XTrain);

%apply normpdf function to normalize Xtrain data.
for j = 1:57
    for i = 1:3065
        PXj_Y1_train(i,j) = normpdf(znorm_Xtrain(i,j),mu_Y1_XTrain(j),sigma_Y1_XTrain(j));
        PXj_Y0_train(i,j) = normpdf(znorm_Xtrain(i,j),mu_Y0_XTrain(j),sigma_Y0_XTrain(j));
    end
end
%apply normpdf function to normalize Xtest data.
for j = 1:57
    for i = 1:1536
        PXj_Y1_test(i,j) = normpdf(znorm_Xtest(i,j),mu_Y1_XTrain(j),sigma_Y1_XTrain(j));
        PXj_Y0_test(i,j) = normpdf(znorm_Xtest(i,j),mu_Y0_XTrain(j),sigma_Y0_XTrain(j));
    end
end

%caculate the train error rate
train_error_count = 0;
for i=1:3065
    AcculumP_train = P_Y1/P_Y0;%P_Y1/P_Y0*(the product of each feature's value in each row).
    for j = 1:57 
        AcculumP_train = AcculumP_train * PXj_Y1_train(i,j)/PXj_Y0_train(i,j);
    end
        if(AcculumP_train > 1) %product of P(Y=1|Xj) > that of P(Y=0|Xj)
        ytrain_actual(i,1) = 1;
    else
        ytrain_actual(i,1) = 0;
        end
    if(ytrain_actual(i,1) ~= ytrain(i))
        train_error_count = train_error_count+1;
    end
    train_error_rate =  train_error_count/3065;
end

%caculate the test error rate
test_error_count = 0;
for i=1:1536
    AcculumP_test = P_Y1/P_Y0;%P_Y1/P_Y0*(the product of each feature's value in each row).
    for j = 1:57 
        AcculumP_test = AcculumP_test * PXj_Y1_test(i,j)/PXj_Y0_test(i,j);
    end 
       if(AcculumP_test > 1) %product of P(Y=1|Xj) > that of P(Y=0|Xj)
        ytest_actual(i,1) = 1;
    else
        ytest_actual(i,1) = 0;
       end
    if(ytest_actual(i,1) ~= ytest(i))%compare two test results
        test_error_count = test_error_count+1;
    end
    test_error_rate =  test_error_count/1536;
end
