clc
clear
close all


%% Metadata

n.stimsites = 60;
n.subs = 28;
direct.dataroot = [pwd '/Data/'];

%% Loop through subjects
for sub = 1: n.subs
    % Setup directory structure
    direct.datasub = [direct.dataroot 'S' num2str(sub) '/'];
    mkdir(direct.datasub)

    % Load data 
    load([direct.dataroot 'data_s' num2str(sub) '.mat'])
    
    % Save data
    for site = 1: n.stimsites
        dat = data{site};
        save([direct.datasub 'data_s' num2str(sub) '_site' num2str(site) '.mat'], 'dat')
    
    end
end


return

%% Metadata

n.stimsites = 60;
n.subs = 28;
direct.dataroot = [pwd '/Data/Datasample/'];

%% Loop through subjects
sub = 1;

% Setup directory structure
direct.datasub = [direct.dataroot 'S' num2str(sub) '/'];
mkdir(direct.datasub)

% Load data 
load([direct.dataroot 'dataclean.mat'])

% Save data
for site = 1: n.stimsites
    dat = data{site};
    save([direct.datasub 'data_s' num2str(sub) '_site' num2str(site) '.mat'], 'dat')

end

