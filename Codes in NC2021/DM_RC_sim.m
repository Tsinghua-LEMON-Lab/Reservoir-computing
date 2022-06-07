%% I-V
clc;
clear;
% ----------------------Model Parameters----------------------
para.r = 0.99;
para.G0 = 0.5;
para.Kp = 9.13;
para.Kn = 0.32;
para.alpha = 0.23;

% ----------------------Voltage Sequence----------------------
Vmin = -3;
Vmax = 3;
inv = 0.099;
V = -[0:inv:Vmax-inv, Vmax:-inv:Vmin+inv, Vmin:inv:0];

% ----------------------Start Simulation----------------------
step = length(V);
I = zeros(1, step);
G = para.G0;
for i = 1:step
    [I(i), G] = DynamicMemristor(V(i), G, para);
end

% ----------------------Experimental Data----------------------
load('exdata.mat');

% ----------------------Plot----------------------
figure;
semilogy(V, abs(I)+10^-5, 'b');
hold on;
semilogy(Vex, Iex, 'r');
str1 = '\color{blue}Simulation';
str2 = '\color{red}Experiment';
lg = legend(str1, str2);
set(lg, 'box', 'off');
xlabel('Voltage (V)');
ylabel('Current (μA)');
axis([-3, 3, -inf, inf]);
set(gca, 'FontName', 'Arial', 'FontSize', 20);
set(gcf, 'unit', 'normalized', 'position', [0.2,0.2,0.3,0.45]);
%% Waveform classification
clear;clc;
% ----------------------Model Parameters----------------------
para.r = 0.99;
para.G0 = 0.5;
para.Kp = 9.13;
para.Kn = 0.32;
para.alpha = 0.23;

% ----------------------DM_RC Parameters----------------------
ML = 4;
N = 10;
Vmax = 3;
Vmin = -2;

% ----------------------DATASET----------------------
sample = 8;
step = 2000;
Data = zeros(1, 2*step);
p1 = sin(pi*2*(0:sample-1)/sample);
p2(1:sample/2) = 1;
p2(sample/2+1:sample) = -1;
for i = 1:2*step/sample
    q = unidrnd(2);
    if q == 1
        Data(sample*(i-1)+1:sample*i) = p1;
        Label(sample*(i-1)+1:sample*i) = 0;
    else
        Data(sample*(i-1)+1:sample*i) = p2;
        Label(sample*(i-1)+1:sample*i) = 1;
    end
end

% ----------------------TRAIN----------------------
% initialize input stream
Input = Data(1:step);

% generate target
Target = Label(1:step);

% mask process
Mask = 2*unidrnd(2, N, ML)-3;
Input_ex = [];
for j = 1:N
    for i = 1:step
        Input_ex(j, (i-1)*ML+1:ML*i) = Input(i)*Mask(j, :);
    end
end
UL = max(max(Input_ex));
DL = min(min(Input_ex));
Input_ex = (Input_ex-DL)/(UL-DL)*(Vmax - Vmin)+Vmin;

% memristor output
memout = [];
G = para.G0;
for i = 1:length(Input_ex(1, :))
    [memout(:,i), G] = DynamicMemristor(Input_ex(:,i), G, para);
    sprintf('%s', ['train:', num2str(i), ', Vmax:', num2str(Vmax), ', ML:', num2str(ML)])
end

% states collection
states = [];
for i = 1:step
    a = memout(:, ML*(i-1)+1:ML*i);
    states(:, i) = a(:);
end
X = [ones(1,step); states];

% linear regression
Wout = Target*pinv(X);

% ----------------------TEST----------------------
% initialize input stream
Input = Data(step+1:end);

% generate target
Target = Label(step+1:end);

% mask process
Input_ex = [];
for j = 1:N
    for i = 1:step
        Input_ex(j, (i-1)*ML+1:ML*i) = Input(i)*Mask(j, :);
    end
end
UL = max(max(Input_ex));
DL = min(min(Input_ex));
Input_ex = (Input_ex-DL)/(UL-DL)*(Vmax - Vmin)+Vmin;

% memristor output
memout = [];
states = [];
G = para.G0;
for i = 1:length(Input_ex(1, :))
    [memout(:, i), G] = DynamicMemristor(Input_ex(:,i),G,para);
    sprintf('%s',['test:', num2str(i), ', Vmax:', num2str(Vmax), ', ML:', num2str(ML)])
end

% states collection
for i = 1:step
    a = memout(:, ML*(i-1)+1:ML*i);
    states(:,i) = a(:);
end
X = [ones(1,step);states];

% system output
Out = Wout*X;
NRMSE = sqrt(mean((Out(10:end)-Target(10:end)).^2)./var(Target(10:end)));
sprintf('%s',['NRMSE:',num2str(NRMSE)])

% ----------------------PLOT----------------------
figure;
subplot(2, 1, 1);
plot(Input, 'b', 'linewidth', 1);
hold on;
plot(Input, '.r');
axis([0, 400, -1.2, 1.2])
ylabel('Input')
set(gca,'FontName', 'Arial', 'FontSize', 20);
subplot(2, 1, 2);
plot(Target, 'k', 'linewidth', 2);
hold on;
plot(Out, 'r', 'linewidth',1);
axis([0, 400, -0.2, 1.2])
str1 = '\color{black}Target';
str2 = '\color{red}Output';
lg = legend(str1, str2);
set(lg, 'Orientation', 'horizon');
ylabel('Prediction')
xlabel('Time (\tau)')
set(gca,'FontName', 'Arial', 'FontSize', 20);
set(gcf, 'unit', 'normalized', 'position', [0.2, 0.2, 0.6, 0.35]);
%% Spoken-digit recognition
clc;
clear;
addpath('Auditory Toolbox\');

% ----------------------Model Parameters----------------------
para.r = 0.99;
para.G0 = 0.5;
para.Kp = 9.13;
para.Kn = 0.32;
para.alpha = 0.23;

% ----------------------DM_RC Parameters----------------------
ML = 10;
N = 40;
Vmax = 3;
Vmin = 0;
Mask = 2*randi([0,1],64,ML,N)-1;

% ----------------------DATASET----------------------
for i = 1:5
    for j = 1:10
        for k = 1:10
            filename(k+(i-1)*10,j) = {['Voice Data\train\f',num2str(i),'\','0',num2str(j-1),'f',num2str(i),'set',num2str(k-1),'.wav']};
        end
    end
end

% ----------------------CROSS-VALIDATION----------------------
WRR = 0;
TF=zeros(10,10);
for u=1:10
% SHUFFLE DATASET
S = [];
for i = 1:10
    r = randperm(size(filename,1));
    res = filename(:,i);
    res = res(r,:);
    S = [S,res];
end

% TRAIN
words = 450;
VL = zeros(words);
Target = [];X = [];
q = 0;p = 1;
for j = 1:words   
    q = q+1;
    if q > 10
        q = 1;
        p = p+1;
    end
    
    % data preprocess
    a = audioread(S{p,q});
    a = resample(a,8000,12500);
    f = LyonPassiveEar(a,8000,250);
    L = zeros(10,length(f(1,:)));
    L(q,:) = ones(1,length(f(1,:)));
    VL(j) = length(f(1,:));
    Target(:,sum(VL(1:j))-VL(j)+1:sum(VL(1:j))) = L;
    
    % mask process
    Input = [];
    for k = 1:N
        for i = 1:VL(j)
            Input(k, ML*(i-1)+1:ML*i) = abs(f(:,i))'*Mask(:,:,k);            
        end
    end
    UL = max(max(Input));
    DL = min(min(Input));
    Input = (Input-DL)/(UL-DL)*(Vmax - Vmin)+Vmin;
    
    % memristor output
    memout = [];
    G = para.G0;
    for i = 1:length(Input(1, :))
        [memout(:,i), G] = DynamicMemristor(Input(:,i), G, para);
    end
    
    % states collection
    for i = 1:VL(j)
        a = memout(:, ML*(i-1)+1:ML*i);
        X(:,sum(VL(1:j))-VL(j)+i) = a(:);
    end
    
    sprintf('%s',['loop:',num2str(u),',train:',num2str(j),',',num2str(u-1),'acc:',num2str(WRR)])
end

% linear regression
Wout = Target*X'*pinv(X*X');

% TEST
clc;
VL = zeros(words);
Target = [];X=[];
words = 50;q = 0;p = 46;
for j=1:words
    q = q+1;
    if q > 10
        q = 1;
        p = p+1;
    end

    % data preprocess
    a = audioread(S{p,q});
    a = resample(a,8000,12500);
    f = LyonPassiveEar(a,8000,250);
    L = zeros(10,length(f(1,:)));
    L(q,:) = ones(1,length(f(1,:)));
    VL(j) = length(f(1,:));
    Target(:,sum(VL(1:j))-VL(j)+1:sum(VL(1:j))) = L;
    
    % mask process
    Input = [];
    for k=1:N
        for i=1:VL(j)
            Input(k, ML*(i-1)+1:ML*i) = abs(f(:,i))'*Mask(:,:,k);            
        end
    end
    UL = max(max(Input));
    DL = min(min(Input));
    Input = (Input-DL)/(UL-DL)*(Vmax - Vmin)+Vmin;
    
    % memristor output
    memout = [];
    G = para.G0;
    for i = 1:length(Input(1, :))
        [memout(:,i), G] = DynamicMemristor(Input(:,i), G, para);
    end
    
    % states collection
    for i = 1:VL(j)
        a = memout(:, ML*(i-1)+1:ML*i);
        X(:,sum(VL(1:j))-VL(j)+i) = a(:);
    end

    sprintf('%s',['loop:',num2str(u),',test:',num2str(j)])
end

% system output
Y = Wout*X;

% accuracy calculation
Mout = [];
rl = zeros(10,10);
real = zeros(10,words);
for i=1:words
    Mout(:,i) = mean(Y(:,sum(VL(1:i))-VL(i)+1:sum(VL(1:i))),2);
    [~,id] = max(Mout(:,i));
    real(id,i) = 1;
    if mod(i,10) == 0
        rl = rl+real(:,(i/10-1)*10+1:i);
    end
end
WRR = 100*sum(sum(rl.*eye(10,10)))/words;
TF= TF+rl;
end
WRR = 100*sum(sum(TF.*eye(10,10)))/(u*words);

% ----------------------PLOT----------------------
figure(1);
x = [0 9];y = [0 9];
imagesc(x, y, TF);
ylabel('Predicted output digit')
xlabel('Correct output digit')
title(['Acc: ',num2str(WRR),'%'])
colorbar;
colormap(flipud(hot)); 
set(gca,'FontName', 'Arial', 'FontSize', 15);

figure(2);
subplot(2,1,1)
plot(Input(1, :));
ylabel('Input (V)')
axis([0,inf,-inf,inf]);
set(gca,'FontName', 'Arial', 'FontSize', 15);
subplot(2,1,2)
plot(memout(1, :), 'r');
xlabel('Time step')
ylabel('Output (μA)')
axis([0,inf,-inf,inf]);
set(gca,'FontName', 'Arial', 'FontSize', 15);
%% Henon Map prediction
clc;
clear;
% ----------------------Model Parameters----------------------
para.r = 0.99;
para.G0 = 0.5;
para.Kp = 9.13;
para.Kn = 0.32;
para.alpha = 0.23;

% ----------------------DM_RC Parameters----------------------
ML = 4;
N = 25;
Vmax = 2.5;
Vmin = 0;

% ----------------------DATASET----------------------
step = 1000;
dataset = HenonMap(2*step+1);

% ----------------------TRAIN----------------------
% initialize input stream
Input = dataset(1:step+1);

% generate target
Target = Input(2:end);

% mask process
Mask = 2*unidrnd(2, N, ML)-3;
Input_ex = [];
for j = 1:N
    for i = 1:step
        Input_ex(j, (i-1)*ML+1:ML*i) = Input(i)*Mask(j, :);
    end
end
UL = max(max(Input_ex));
DL = min(min(Input_ex));
Input_ex = (Input_ex-DL)/(UL-DL)*(Vmax - Vmin)+Vmin;

% memristor output
memout = [];
states = [];
G = para.G0;
for i = 1:length(Input_ex(1, :))
    [memout(:,i), G] = DynamicMemristor(Input_ex(:,i), G, para);
    sprintf('%s', ['train:', num2str(i), ', Vmax:', num2str(Vmax), ', ML:', num2str(ML)])
end

% linear regression
for i = 1:step
    a = memout(:, ML*(i-1)+1:ML*i);
    states(:, i) = a(:);
end
X = [ones(1, step); states];
Wout = Target*pinv(X);

% ----------------------TEST----------------------
% initialize input stream
Input = dataset(step+1:2*step+1);
% generate target
Target = Input(2:end);

% mask process
Input_ex = [];
for j = 1:N
    for i = 1:step
        Input_ex(j, (i-1)*ML+1:ML*i) = Input(i)*Mask(j, :);
    end
end
UL = max(max(Input_ex));
DL = min(min(Input_ex));
Input_ex = (Input_ex-DL)/(UL-DL)*(Vmax - Vmin)+Vmin;

% memristor output
memout = [];
states = [];
G = para.G0;
for i = 1:length(Input_ex(1, :))
    [memout(:, i), G] = DynamicMemristor(Input_ex(:,i),G,para);
    sprintf('%s',['test:', num2str(i), ', Vmax:', num2str(Vmax), ', ML:', num2str(ML)])
end

% system output
for i = 1:step
    a = memout(:, ML*(i-1)+1:ML*i);
    states(:,i) = a(:);
end
X = [ones(1, step); states];
Out = Wout*X;
NRMSE = sqrt(mean((Out(10:end)-Target(10:end)).^2)./var(Target(10:end)));
sprintf('%s',['NRMSE:',num2str(NRMSE)])

% ----------------------PLOT----------------------
% time series
figure(1);
plot(Target(1:200), 'k', 'linewidth', 2);
hold on;
plot(Out(1:200), 'r', 'linewidth',1);
axis([0, 200, -2, 2])
str1 = '\color{black}Target';
str2 = '\color{red}Output';
lg = legend(str1, str2);
set(lg, 'Orientation', 'horizon', 'box', 'off');
ylabel('Prediction')
xlabel('Time (\tau)')
set(gca,'FontName', 'Arial', 'FontSize', 20);
set(gcf, 'unit', 'normalized', 'position', [0.2, 0.2, 0.6, 0.35]);

% 2D map
figure(2);
plot(Target(2:end), 0.3*Target(1:end-1), '.k', 'markersize', 12);
hold on;
plot(Out(2:end), 0.3*Out(1:end-1), '.r', 'markersize', 12);
str1 = '\color{black}Target';
str2 = '\color{red}Output';
lg = legend(str1,str2);
set(lg, 'box', 'off');
ylabel('{\ity} (n)');
xlabel('{\itx} (n)');
axis([-2, 2, -0.4, 0.4]);
set(gca, 'FontName', 'Arial', 'FontSize', 20);
set(gcf, 'unit', 'normalized', 'position', [0.2,0.2,0.3,0.45]);
