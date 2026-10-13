function write_csv(filename, header, data)
%WRITE_CSV  A plain CSV with one header line, readable by pgfplots.
%
%   write_csv(filename, header, data)
%
%   HEADER is a cell array of column names, DATA a numeric matrix with as
%   many columns.  Empty entries are written as NaN, which pgfplots skips
%   when the axis is given "unbounded coords=jump".  Nothing else is
%   produced: no macros, no LaTeX.  Every number quoted in the paper is
%   read off this file by hand, on purpose.

assert(numel(header) == size(data,2), ...
    'write_csv: %d names for %d columns', numel(header), size(data,2));

fid = fopen(filename, 'w');
fprintf(fid, '%s', header{1});
fprintf(fid, ',%s', header{2:end});
fprintf(fid, '\n');
for i = 1:size(data,1)
    fprintf(fid, '%.10g', data(i,1));
    fprintf(fid, ',%.10g', data(i,2:end));
    fprintf(fid, '\n');
end
fclose(fid);
fprintf('  written %s  (%d rows x %d columns)\n', filename, size(data,1), size(data,2));
end