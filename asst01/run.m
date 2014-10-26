% Machine Learning -- Assignment 1
%
% This simple script should be run to test all components of the assigment.
% Comment out any calls that aren't implemented when you are working on the
% first parts

clear;

do_dist_test = 1;
do_knn_reg = 1;
do_knn_class = 1;
do_prob_plots = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%
% Test distance functions
if (do_dist_test)
  
  % Use these numbers for testing speed. 
  n = 10000;
  m = 500;
  d = 100;

  % Use these for debugging and testing other code
  % n = 200;
  % m = 50;
  % d = 10;

  % Some random vectors (in matrix format)
  a = randn(n,d);
  b = randn(m,d);

  tic;
  dist1 = dist_1(a,b);
  fprintf('Finished dist_1 in %0.3f seconds\n',toc);
  tic;
  dist2 = dist_2(a,b);
  fprintf('Finished dist_2 in %0.3f seconds\n',toc);

  % check should be close to 0
  check = max(abs(dist1(:)-dist2(:)));
  fprintf('Max difference between distance matrices is %d\n', check);

end


%%%%%%%%%%%%%%%%%%%%%%
% Test k-NN Regression
if (do_knn_reg)
  
  % Try different training set sizes
  n_vals = [100 500 1000 5000 10000];
  nk = 10;

  m = 100;
  d = 8;
  gauss_err_std = 0.05;

  test_errs = zeros(length(n_vals),nk);

  for i=1:length(n_vals)

    n = n_vals(i);

    % Create synthetic data using a linear model
    weights = rand(d,1);
    X_trn = rand(n,d);  % feature data for training set
    X_tst = rand(m,d);  % feature data for test set
    y_trn = X_trn * weights;  % apply linear model
    y_trn = y_trn + gauss_err_std*randn(n,1);  % add some gaussian noise
    y_tst = X_tst * weights;

    % Now test k-nn with different values of k

    for k=1:nk

      preds = knn(k, X_trn, y_trn, X_tst, 1);
      mse = mean((preds - y_tst).^2);
      test_errs(i,k) = mse;

    end

  end

  plot_error_curves(test_errs,n_vals);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test k-NN Classification

if (do_knn_class)
  
  % Load some data build in to Matlab
  
  % you may need the Matlab Statistics toolbox for the next two lines of
  % code. If you don't have it, figure out some other data to use for
  % testing your classification algorithm
  load fisheriris;
  figure;
  gscatter(meas(:,1), meas(:,2), species,'rgb','osd');
  
  [n_total d] = size(meas);
  y_vals = zeros(n_total,1);
  grp1_inds = find(strcmp(species,'setosa'));
  grp2_inds = find(strcmp(species,'versicolor'));
  grp3_inds = find(strcmp(species,'virginica'));
  y_vals(grp1_inds) = 1;
  y_vals(grp2_inds) = 2;
  y_vals(grp3_inds) = 3;
  
  nk = 10;
  n_runs = 100;  % try this many different scrambles of the data
  test_errs = zeros(n_runs,nk);
  
  for i=1:n_runs
  
    % First scramble the data to mix up classes
    new_order = randperm(n_total);
    new_meas = meas(new_order,:);
    new_y_vals = y_vals(new_order);

    X_trn = new_meas(1:100,:);
    y_trn = new_y_vals(1:100);
    X_tst = new_meas(101:end,:);
    y_tst = new_y_vals(101:end);

    n_tst = length(y_tst);

    for k=1:nk
      preds = knn(k, X_trn, y_trn, X_tst, 0);
      num_corr = length(find(preds == y_tst));
      accuracy = 100*num_corr/n_tst;
      test_errs(i,k) = accuracy;
    end
    
  end
  mean_errs = mean(test_errs);  % average each column (i.e., over runs)
  figure;
  plot(mean_errs);
  
end


%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test k-NN Classification

if (do_prob_plots)
  
  plot_probs();
  
end






