function [mel]=f2mel(hz)
mel=2595 * log10(1+hz/700);
end