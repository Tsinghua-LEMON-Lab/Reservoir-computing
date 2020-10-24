%% I-V
clc;
clear;
% ----------------------Simulation Parameters----------------------
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
ylabel('Current (¦ÌA)');
axis([-3, 3, -inf, inf]);
set(gca, 'FontName', 'Arial', 'FontSize', 20);
set(gcf, 'unit', 'normalized', 'position', [0.2,0.2,0.3,0.45]);
%% Henon Map prediction
clc;
clear;
% ----------------------Simulation Parameters----------------------
para.r = 0.99;
para.G0 = 0.5;
para.Kp = 9.13;
para.Kn = 0.32;
para.alpha = 0.23;

% ----------------------DM_RC Parameters----------------------
ML = 4;
N = 25;
Vmax = 1;
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
%% Waveform classification
clear;clc;
% ----------------------Simulation Parameters----------------------
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
        Label(sample*(i-1)+1:sample*i) = 1;
    else
        Data(sample*(i-1)+1:sample*i) = p2;
        Label(sample*(i-1)+1:sample*i) = 0;
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
X = [ones(1,step); states];
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

% system output
for i = 1:step
    a = memout(:, ML*(i-1)+1:ML*i);
    states(:,i) = a(:);
end
X = [ones(1,step);states];
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
