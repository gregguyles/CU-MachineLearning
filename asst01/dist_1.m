% Greg Guyles
% Machine learning
% Asst 1
% 1-24-2014

function dist = dist_1(a,b)
% This function computes the Euclidean distance between vector a and b. If
% a or b is a matrix, all combinations of distances are computed where
% each row corresponds to a different vector. For example if a is an n x d
% matrix and b is an m x d matrix, the output will be an n x m matrix where
% the i,j entry is the distance between row i in a and row j in b. a and b
% must have the same number of columns.

[aRows, aCols] = size(a);
[bRows, bCols] = size(b);

% Check if input columns are equal
if (aCols ~= bCols)
    disp('ERROR: input matrices must have equal number of Columns')
    return
end

% initialize output matrix aRows x bRows
dist = zeros(aRows, bRows);

for i = 1:aRows
    for j = 1:bRows
        % find difference of each corresponding element
        % raise to power 2 and take root
        dist(i, j) = sqrt(sum((a(i,:)-b(j,:)) .^ 2));
    end                                  % end j loop
end                                      % end i loop

end                                      % end function