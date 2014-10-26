% Greg Guyles
% Machine learning
% Asst 1
% 1-24-2014

function preds = knn(k, X_trn, y_trn, X_tst, do_regression)

% find distance matrix
Dist = dist_2(X_tst, X_trn);

% find k closest values and indices
[DistSort, DistInd] = sort(Dist, 2);
DistSort = DistSort(1:end, 1:k);
DistInd = DistInd(1:end, 1:k);
    
if (do_regression)
    
    % find inverse of normalized distances
    DNorm = DistSort .^ -1;
    DNorm = sum(DNorm, 2);
    DNorm = bsxfun(@times, DNorm, DistSort);
    DNorm = DNorm .^ -1;
    
    % use indices to get y training values
    y2 = y_trn(DistInd);
    
    % perform element-by-element multiplication
    y3 = bsxfun(@times, y2, DNorm);
    
    % sum rows
    preds = sum(y3, 2);
end


if (~do_regression)
    [rows, cols] = size(DistSort);
    y2 = y_trn(DistInd);
    for row = 1:rows
        for col = 1:cols
            
            % initialize temp vars for each category
            temp = [0 0 0];
            
            % increment appropriate temp var by 1/dist
            temp(y2(row, col)) = temp(y2(row, col)) + DistSort(row, col) ^ -1;
        end
        
        % find the max temp value and append predes with its index
        [max1, maxInd] = max(temp);
        preds(row, 1) = maxInd;
    end
end

end