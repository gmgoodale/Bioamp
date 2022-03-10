clear all
file = '01_2022_02_24 04_15.csv';
ECG = readmatrix(file, 'Range', 'E2:E38000');
ECG(1:10000,1:1);
fs=1000
d = lowpass(ECG(1:10000,1:1),10,fs);
smooth(d, 10);
r=snr(d);
r

