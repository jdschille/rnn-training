classdef RNN_
    methods(Static)
        
        function RNN = learn(T,N,b,m,RNN,data,initial_constraint,rand_setup,custom_warmstart,delay_idx)
            % initial constraint
            % 0 = TBPTT RNN, fix initial conditions to zero
            % 1 = benchmark RNN, coupled initial conditions
            % 3 = free initial conditions
            
            % ensure same seed
            if rand_setup.fix_seed
                rng(rand_setup.seed)
            end

            print_string = ['Training: ' ...
                ' case: ' num2str(initial_constraint) ...
                ', m: ' num2str(m)];
            disp(print_string)

            % NLP solver
            import casadi.*
            solver = NLP.create_mini(N,b,m,RNN,initial_constraint,delay_idx);
            disp('... NLP created, start solving')

            % construct data batch
            SI = 1:T-N+1;
            for j = 1:b
                data_mini{j}.X = data.X(:,SI(j):SI(j)+N-1);
                data_mini{j}.Y = data.Y(:,SI(j):SI(j)+N-1);
            end

            % warmstart and mini-batch segmentation
            MINI_H0 = zeros(RNN.n_hidden,1);
            if initial_constraint==3
                warmstart.theta = custom_warmstart.theta;
                warmstart_H = custom_warmstart.H;
            else
                warmstart.theta = 1e-1*randn(RNN.n_learnables,1);
                warmstart.weights = theta2weights(warmstart.theta,RNN);
                for j = 1:b
                    warmstart_H{j}.H = zeros(RNN.n_hidden,N+1);
                    if initial_constraint==1 && j>1
                        warmstart_H{j}.H(:,1) = warmstart_H{j-1}.H(:,2);
                    end
                    for i = 1:N
                        warmstart_H{j}.H(:,i+1) = RNN.f(warmstart_H{j}.H(:,i),data_mini{j}.X(:,i),...
                            warmstart.weights.W_hh,warmstart.weights.W_xh,warmstart.weights.b_h);
                    end
                end
            end


            % solve NLP
            solution = NLP.solve_mini(solver,warmstart_H,warmstart.theta,data_mini,MINI_H0);

            % generate output predictions
            RNN.SI = SI;
            RNN.obj = solution.f;
            RNN.H = solution.H;
            RNN.theta = solution.theta;
            RNN.weights = theta2weights(solution.theta,RNN);
            RNN.Y = cell(b,1);
            for j=1:b
                RNN.Y{j}.Y = zeros(size(data_mini{j}.Y));

                for i = 1:N
                    % compute output predictions
                    RNN.Y{j}.Y(:,i) = RNN.g(RNN.H{j}.H(:,i+1),RNN.weights.W_hy,RNN.weights.b_y);
                end
            end

            disp('--- Training successfull')
        end

        function RNN_T = predict(RNN,initial,data,t1,t2,t3)
            T = length(data.Y);
            RNN_T.H = zeros(RNN.n_hidden,T+1);
            RNN_T.H(:,1) = initial;
            RNN_T.Y = zeros(RNN.n_output,T);
            for i = 1:T
                RNN_T.H(:,i+1) = RNN.f(RNN_T.H(:,i),data.X(:,i),...
                    RNN.weights.W_hh,RNN.weights.W_xh,RNN.weights.b_h);
            
                RNN_T.Y(:,i) = RNN.g(RNN_T.H(:,i+1),...
                    RNN.weights.W_hy,RNN.weights.b_y);
            end
            RNN_T.E_Y = (RNN_T.Y-data.Y).^2;
            RNN_T.MSE_Y_t1t2 = sum(RNN_T.E_Y(t1:t2-1))/length(RNN_T.E_Y(t1:t2-1));
            RNN_T.MSE_Y_t2t3 = sum(RNN_T.E_Y(t2:t3))/length(RNN_T.E_Y(t2:t3));
            RNN_T.MSE_Y_t1t3 = sum(RNN_T.E_Y(t1:t3))/length(RNN_T.E_Y(t1:t3));
        end
    end
end