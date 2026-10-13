function setup()
%SETUP  Put this repository on the MATLAB path.
%
%   Run once per session, from the repository root:
%
%       >> setup
%
%   Adds src/ and utils/ to the path and checks that the TT-Toolbox is
%   reachable.  Nothing is written outside the current directory.

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'src'));
addpath(fullfile(here, 'utils'));

if exist('tt_tensor', 'class') ~= 8 && exist('tt_tensor', 'file') == 0
    warning('setup:ttToolbox', ...
        ['The TT-Toolbox was not found on the path.  It is required: the\n' ...
         '         routines here take and return tt_tensor and tt_matrix objects.\n' ...
         '         Clone https://github.com/oseledets/TT-Toolbox and run its\n' ...
         '         setup_tt.m, then run this script again.']);
else
    fprintf('  src/ and utils/ added to the path; TT-Toolbox found.\n');
end
end
