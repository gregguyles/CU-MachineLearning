% Greg Guyles
% Machine learning
% Asst 1
% 1-24-2014

function dist = dist_2(a,b)

aCols = length(a(1,:));
bCols = length(b(1,:));

% Check if input columns are equal
if (aCols ~= bCols)
    disp('ERROR: input matrices must have equal number of Columns')
    return
end

% square and sum terms by row
% add rows of a & b element-by-element
% subtract product of element-by-element * 2
% take root
dist = sqrt(bsxfun(@plus, dot(a, a, 2), dot(b, b, 2)') - (a * b') * 2);

end