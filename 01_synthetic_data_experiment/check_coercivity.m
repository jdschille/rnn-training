function RNN_test = check_coercivity(RNN_test,RNN_opti,m_set,data,epsilon)

    S = length(RNN_opti{1}.Y);
    N = length(RNN_test{1}.Y{1}.Y);

    for m_idx = 1:length(m_set)

        coerc_test_H = 0;
        coerc_test_J = 0;
    
        for j = 1:S
            for i = m_set(m_idx)+1:N
                y_str = RNN_test{m_idx}.Y{j}.Y(i);
                y_inf = RNN_opti{m_idx}.Y{j}.Y(i);
                coerc_test_H = coerc_test_H + (y_str-y_inf)'*(y_str-y_inf);
                coerc_test_J = coerc_test_J + 2*(y_inf-data.Y(j+i-1))'*(y_str-y_inf);
            end
        end
        

        optimality_gap = max(0,RNN_opti{m_idx}.obj-RNN_test{m_idx}.obj);

        if optimality_gap > 0, warning('optimality gap detected'); end

        if coerc_test_J >= -1/epsilon*(coerc_test_H) - optimality_gap
            max_c_possible = -(coerc_test_H-optimality_gap)/coerc_test_J;
            if max_c_possible<=0
                max_c_possible = inf;
            end
            msg = ['Coercivity property holds, max_c_possible = ' num2str(max_c_possible)];
        else
            msg = ('Coercivity property invalid');
            warning(msg)
        end
        
        RNN_test{m_idx}.coercivity = msg;
        RNN_test{m_idx}.coercivity_optimality_gap = optimality_gap;
    end
end
