function f_tilda = th_p_of_Lamda1(alpha,f)

f_zero = 0.7;
[n,m] = size(f);
for i=1:m
    for j=1:n
        freq = f(i,j);
        if (freq > f_zero)
            f_tilda(i,j) = alpha*f_zero + ((1-alpha*f_zero)/(1-f_zero)) *...
                (freq - f_zero);

        elseif (freq == 1)
             f_tilda(i,j) = 1;
        elseif (freq == 0)
             f_tilda(i,j) = 0;
        else
             f_tilda(i,j) = alpha * freq;
        end
    end
end
end