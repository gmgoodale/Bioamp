file = '01_2022_02_24 04_15.csv';

% Import data
time = readmatrix(file, 'Range', 'A10:A38000');
X = readmatrix(file, 'Range', 'E10:E38000');

% FFT Section
Fs = 1000;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = length(X);        % Length of signal
t = (0:L-1)*T;        % Time vector

FFTX = fft(X);

P2 = abs(FFTX/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f = Fs*(0:(L/2))/L;

Xfilt = lowpass(X, 10, Fs);
Xsmooth = smooth(X, 100);

% Begin Plotting section
tiledlayout(2,1)
start = 2;
stop = 3700;

% Top plot
nexttile
plot(time(start:stop),X(start:stop),time(start:stop),Xfilt(start:stop),time(start:stop),Xsmooth(start:stop),'LineWidth', 1.5)
title('X Accelerometer');
legend('Origirnal','Filtered','Smoothed','Location','NorthEast');
xlabel('Time')
ylabel('Signal Values')

% Bottom plot
nexttile
plot(f,P1) 
title('')
xlabel('f (Hz)')
ylabel('|P1(f)|')