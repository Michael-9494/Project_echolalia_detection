function omega_tilda = th_p_of_Lamda(alpha,f,f_max)


if alpha<1
    f_zero = (7/8)*f_max;
elseif  alpha>1
    f_zero = (7/(8*alpha))*f_max;
elseif alpha==1
    f_zero =    f_max ;
    omega_tilda = f;
    return
end

if (f > f_zero)
    omega_tilda = alpha*f_zero + ((f_max-alpha*f_zero)/(f_max-f_zero)) *...
        (f - f_zero);

elseif (f == f_max)
    omega_tilda = 1;
elseif (f == 0)
    omega_tilda = 0;
else
    omega_tilda = alpha * f;
end

end