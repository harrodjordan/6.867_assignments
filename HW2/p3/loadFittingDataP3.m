function [X,Y] = loadFittingDataP3(ifPlotData)

if nargin < 1
    ifPlotData = 1;
end

data = importdata('p3curvefitting.txt');

X = data(1,:); Y = data(2,:);

if ifPlotData
    figure;
    plot(X, Y, 'o', 'MarkerSize', 8);
    xlabel('x'); ylabel('y');
end