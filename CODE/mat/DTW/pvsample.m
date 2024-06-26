function c = pvsample(b, t, hop)
% c = pvsample(b, t, hop)   Interpolate an STFT array according to the 'phase vocoder'
%     b is an STFT array, of the form generated by 'specgram'.
%     t is a vector of (real) time-samples, which specifies a path through 
%     the time-base defined by the columns of b.  For each value of t, 
%     the spectral magnitudes in the columns of b are interpolated, and 
%     the phase difference between the successive columns of b is 
%     calculated; a new column is created in the output array c that 
%     preserves this per-step phase advance in each bin.
%     hop is the STFT hop size, defaults to N/2, where N is the FFT size
%     and b has N/2+1 rows.  hop is needed to calculate the 'null' phase 
%     advance expected in each bin.
%     Note: t is defined relative to a zero origin, so 0.1 is 90% of 
%     the first column of b, plus 10% of the second.

if nargin < 3
  hop = 0;
end

[rows,cols] = size(b);

N = 2*(rows-1);

if hop == 0
  % default value
  hop = N/2;
end

% Empty output array
c = zeros(rows, length(t));

% Expected phase advance in each bin
dphi = zeros(1,N/2+1);
dphi(2:(1 + N/2)) = (2*pi*hop)./(N./(1:(N/2)));

% Phase accumulator
% Preset to phase of first frame for perfect reconstruction
% in case of 1:1 time scaling
ph = angle(b(:,1));

% Append a 'safety' column on to the end of b to avoid problems 
% taking *exactly* the last frame (i.e. 1*b(:,cols)+0*b(:,cols+1))
b = [b,zeros(rows,1)];

ocol = 1;
for tt = t
  % Grab the two columns of b
  bcols = b(:,floor(tt)+[1 2]);
  tf = tt - floor(tt);
  bmag = (1-tf)*abs(bcols(:,1)) + tf*(abs(bcols(:,2)));
  % calculate phase advance
  dp = angle(bcols(:,2)) - angle(bcols(:,1)) - dphi';
  % Reduce to -pi:pi range
  dp = dp - 2 * pi * round(dp/(2*pi));
  % Save the column
  c(:,ocol) = bmag .* exp(1i*ph);
  % Cumulate phase, ready for next frame
  ph = ph + dphi' + dp;
  ocol = ocol+1;
end