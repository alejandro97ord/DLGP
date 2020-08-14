
if select == 1
    fileName = 'Real_Sarcos';
elseif select == 2
    fileName = 'Real_Barrett';
elseif select == 3
    fileName = 'Real_Sarcos_long';
elseif select == 4
    fileName = 'SL_Sarcos';
elseif select == 5
    fileName = 'SL_Barrett';
elseif select == 6
    fileName = '2DoFData_large';
elseif select == 7
    fileName = 'KUKA_flask';
end

load(['C:\Users\alejandro\Desktop\P10\Datasets\',fileName,'.mat']);
try
    load(['C:\Users\alejandro\Desktop\P10\Datasets\hyps_',fileName,'.mat']);
end