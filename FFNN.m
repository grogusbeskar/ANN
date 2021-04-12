x = fitur';
t = target';
hiddenLayerSize = 5;
net = feedforwardnet(hiddenLayerSize, 'trainscg');
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'softmax';
net.input.processFcns = {'removeconstantrows','mapminmax'}; % Memilih Pre Processing Input & Output
net.output.processFcns = {'removeconstantrows','mapminmax'};
net.divideFcn = 'dividerand';  % Fungsi Pembagian Data
net.divideMode = 'sample';  % Mendefinisikan Dimensi Data Target
net.divideParam.trainRatio = 50/100;
net.divideParam.valRatio = 25/100;
net.divideParam.testRatio = 25/100;
net.performFcn = 'crossentropy';  % Memilih Fungsi Performance
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotconfusion', 'plotroc'};

% Inisiasi Melatih Jaringan
[net,tr] = train(net,x,t);

% Menguji Jaringan
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y)
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);

% Menghitung Kembali Training, Validation and Test Performance
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,y)
valPerformance = perform(net,valTargets,y)
testPerformance = perform(net,testTargets,y)

% Menunjukkan Struktur Jaringan
view(net);
% Plot Confusion All
plotconfusion(t,y)
% Plot Confusion Training
yTrn=net(x(:,tr.trainInd)); 
tTrn=t(:,tr.trainInd);
plotconfusion(tTrn,yTrn);
% Plot Confusion Validasi
yVal=net(x(:,tr.valInd)); 
tVal=t(:,tr.valInd);
plotconfusion(tVal,yVal);
% Plot Confusion Testing
yTst=net(x(:,tr.testInd));
tTst=t(:,tr.testInd);
plotconfusion(tTst,yTst);


