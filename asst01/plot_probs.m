% Greg Guyles
% Machine learning
% Asst 1
% 1-24-2014

function hndl = plot_probs()

% make histogram using 100 samples of the 
% normal distribution, mu = 55, sigma = 12
figure;
hist(normrnd(55, 12, 1000, 1));

% add annotations
xlabel('Sampled Value');
ylabel('Frequency');

end