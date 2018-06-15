clear all
clc
load('spamData');
%binarize data from Xtrain to XbinomialTrain      
XbinomialTrain = Xtrain;
XbinomialTrain(Xtrain ~= 0) = 1;
XbinomialTrain(Xtrain == 0) = 0;
%insert one volume to the original XbinomialTrain
Xtrain_first = ones(3065,1);
%binarize data from Xtest to XbinomialTest
XbinomialTest = Xtest;
XbinomialTest(Xtest ~= 0) = 1;
XbinomialTest(Xtest == 0) = 0;
%insert one volume to the original XbinomialTest
Xtest_first = ones(1536,1);
%generate a 57*57 unit matrix to match the equation's matrix dimension
I = eye(57);
for lambda = [1:1:10,15:5:100]  %change lambda according to the CA1 requirement.
     if(lambda <= 10)
     i = lambda;
    else
        i = 8+lambda/5;
    end
    W0_first = 0;   %initate the first feature element to be 0
    mu_first = 1./(1+exp(-W0_first*Xtrain_first));  %using formula to iterate the first feature element (bias) for the first time
    S_first = diag(mu_first.*(1-mu_first));
    g_first = Xtrain_first'*(mu_first - ytrain);    %note: don't apply lambda to the bias
    H_first = Xtrain_first'*S_first*Xtrain_first;
    b = W0_first;   %store the previous W0_first element
    W0_first=b-inv(H_first)*g_first;    %iterate

    Wk = zeros(57,1);   %initate the first Wk from 1 to D(57) dimensions to be 0
    mu = 1./(1+exp(-Wk'*XbinomialTrain'));
    S = diag(mu.*(1-mu));
    g_reg = XbinomialTrain'*(mu' - ytrain) + lambda*Wk;  %apply lambda to iterate 
    H_reg = XbinomialTrain'*S*XbinomialTrain + lambda* I;
    a = Wk; %store the previous Wk
    Wk = a - inv(H_reg)*g_reg;    %iterate

    while(max(abs(W0_first-b)>0.01))    %set a theshold 0.01 to detect the last W0 in the iteration
    mu_first = 1./(1+exp(-W0_first*Xtrain_first));  %iterate without using lambda
    S_first = diag(mu_first.*(1-mu_first));
    g_first = Xtrain_first'*(mu_first - ytrain);
    H_first = Xtrain_first'*S_first*Xtrain_first;
    b = W0_first;
    W0_first=b-inv(H_first)*g_first;
    end

    while(max(abs(Wk-a)>0.01))  %set a theshold 0.01 to detect the last Wk in the iteration
    mu = 1./(1+exp(-Wk'*XbinomialTrain'));   %iterate using lambda
    S = diag(mu.*(1-mu));
    g_reg = XbinomialTrain'*(mu' - ytrain) + lambda*Wk;
    H_reg = XbinomialTrain'*S*XbinomialTrain + lambda* I;
    a = Wk;
    Wk = a - inv(H_reg)*g_reg;
    end
    %calculate train error rate 
    result_train = W0_first*Xtrain_first'+Wk'* XbinomialTrain';  %calculate log(P(Y=1|Xj)/P(Y=0|Xj))
    yactual_train(result_train > 0) = 1;    %classify 
    yactual_train(result_train <= 0) = 0;

    L_train = yactual_train' - ytrain;  %compare two ytrain matrices and count the different number
    train_error_count = 3065-sum(~L_train(:))+1;
    train_error_rate(i) = train_error_count/3065;

    %calculate test error rate 
    result_test = W0_first*Xtest_first'+Wk'* XbinomialTest';
    yactual_test(result_test > 0) = 1;
    yactual_test(result_test <= 0) = 0;

    L_test = yactual_test' - ytest;
    test_error_count = 1536-sum(~L_test(:))+1;
    test_error_rate(i) = test_error_count/1536;
end
%plot error rate with different lambda
plot([1:1:10,15:5:100],train_error_rate(1:28));
hold on
plot([1:1:10,15:5:100],test_error_rate(1:28));
axis([1 100 0.07 0.13])
title('Relationship between error rate and lambda through binarized data');
xlabel('lambda');
ylabel('error rate');
legend('train error rate','test error rate');
