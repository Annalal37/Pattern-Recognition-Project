clear all
clc
load('spamData');
%binarize data from Xtrain to XbinomialTrain     
XbinomialTrain = Xtrain;
XbinomialTrain(Xtrain ~= 0) = 1;
XbinomialTrain(Xtrain == 0) = 0;
%binarize data from Xtest to XbinomialTest
XbinomialTest = Xtest;
XbinomialTest(Xtest ~= 0) = 1;
XbinomialTest(Xtest == 0) = 0;

%train data
%calculate P(Xj=1|Y=1),P(Xj=1|Y=0),P(Xj=0|Y=1),P(Xj=0|Y=0) from Xtrain data
%calculate the number of 'yTrain = 1' and 'yTrain = 0'
N_1 = length(find(ytrain == 1));
N_0 = 3065 - N_1;
P_Y1 = N_1/3065;
P_Y0 = N_0/3065;

%calculate the numerators (case:Nj_11: when ytrain=1, calculate the number of Xbinomial train = 1.)
Nj_11 = double(ytrain==1)'*double(XbinomialTrain==1);
Nj_10 = double(ytrain==0)'*double(XbinomialTrain==1);
Nj_01 = double(ytrain==1)'*double(XbinomialTrain==0);
Nj_00 = double(ytrain==0)'*double(XbinomialTrain==0);

%calculate the test error rate
 for k = 1:201
    alpha = (k-1)/2;%the actual alpha values range from 0:0.5:100
    test_error_count(k) = 0;
    for i=1:1536
        AcculumP(k) = P_Y1/P_Y0;%P_Y1/P_Y0*(the product of each feature's value in each row).
        for j = 1:57 
            PXj_1_Y_1(j)= (Nj_11(j)+alpha)/(N_1+2*alpha);%P(Xj=1|Y=1)
            PXj_0_Y_1(j)= (Nj_01(j)+alpha)/(N_1+2*alpha);%P(Xj=0|Y=1)
            PXj_1_Y_0(j)= (Nj_10(j)+alpha)/(N_0+2*alpha);%P(Xj=1|Y=0)
            PXj_0_Y_0(j)= (Nj_00(j)+alpha)/(N_0+2*alpha);%P(Xj=0|Y=0)
            
            if(XbinomialTest(i,j) == 0)
               AcculumP(k) = AcculumP(k) * PXj_0_Y_1(j)/PXj_0_Y_0(j);
            else
               AcculumP(k) = AcculumP(k) * PXj_1_Y_1(j)/PXj_1_Y_0(j);
            end
        end
            if(AcculumP(k) > 1)%product of P(Y=1|Xj) > that of P(Y=0|Xj)
                ytest_actual(i,k) = 1;
             else
                ytest_actual(i,k) = 0;
             end
           
            if(ytest_actual(i,k) ~= ytest(i,1))%compare two test results
                test_error_count(k) = test_error_count(k) + 1;
            end
    end
            test_error_rate(k) =  test_error_count(k)/1536;
    
 end

 %calculate the train error rate
 for k = 1:201
    alpha = (k-1)/2;
    train_error_count(k) = 0;
    for i=1:3065
        AcculumP(k) = P_Y1/P_Y0;
        for j = 1:57 
            PXj_1_Y_1(j)= (Nj_11(j)+alpha)/(N_1+2*alpha);%P(Xj=1|Y=1)
            PXj_0_Y_1(j)= (Nj_01(j)+alpha)/(N_1+2*alpha);%P(Xj=0|Y=1)
            PXj_1_Y_0(j)= (Nj_10(j)+alpha)/(N_0+2*alpha);%P(Xj=1|Y=0)
            PXj_0_Y_0(j)= (Nj_00(j)+alpha)/(N_0+2*alpha);%P(Xj=0|Y=0)
            
            if(XbinomialTrain(i,j) == 0)
               AcculumP(k) = AcculumP(k) * PXj_0_Y_1(j)/PXj_0_Y_0(j);
            else
               AcculumP(k) = AcculumP(k) * PXj_1_Y_1(j)/PXj_1_Y_0(j);
            end
        end             
               if(AcculumP(k) > 1) 
                ytrain_actual(i,k) = 1;
             else
                ytrain_actual(i,k) = 0;
             end
             if(ytrain_actual(i,k) ~= ytrain(i,1))
                train_error_count(k) = train_error_count(k)+1;
             end
            train_error_rate(k) =  train_error_count(k)/3065;
     end
 end
 
%plot two error rates' trend with different alpha.
axis([0 100 0 0.14])
plot(0:0.5:100,train_error_rate(1:201));
hold
plot(0:0.5:100,test_error_rate(1:201),'r');
title('Relationship between error rate and alpha');
xlabel('alpha');
ylabel('error rate');
legend('train error rate','test error rate');
