% Greg Guyles
% Machine learning
% Asst 1
% 1-24-2014

function hndl = plot_error_curves(test_errs,n_vals)

% make plot and return handle
figure;
hndl = plot(test_errs');

% add annotations
% I'm not quite sure what a better way to do this is
legend(num2str(n_vals(1)), num2str(n_vals(2)), num2str(n_vals(3)), num2str(n_vals(4)), num2str(n_vals(5)));
xlabel('Number of Neighbors');
ylabel('Mean Squared Error');
set(gca,'xTick',0:2:10);

end