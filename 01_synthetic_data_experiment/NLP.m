classdef NLP
    methods(Static)

        function solver = create_mini(N,b,m,RNN,H0_coupling,delay_idx)

            import casadi.*

            % NLP shooting
            for i = 1:b
                H{i} = SX.sym(['H' num2str(i)],RNN.n_hidden,N+1);
            end
            W_hh = SX.sym('W_hh',RNN.n_hidden,RNN.n_hidden);
            W_xh = SX.sym('W_xh',RNN.n_hidden,RNN.n_input);
            b_h = SX.sym('b_h',RNN.n_hidden,1);
            W_hy = SX.sym('W_hy',RNN.n_output,RNN.n_hidden);
            b_y = SX.sym('b_y',RNN.n_output,1);
            theta = vertcat(...
                reshape(W_hh,numel(W_hh),1),reshape(W_xh,numel(W_xh),1),b_h,...
                reshape(W_hy,numel(W_hy),1),b_y);

            % parameters
            for i = 1:b
                X{i} = SX.sym(['X' num2str(i)],RNN.n_input,N);
                Y{i} = SX.sym(['Y' num2str(i)],RNN.n_output,N);
                W{i} = SX.sym(['W' num2str(i)],1,N);
            end
            H0 = SX.sym('H0',RNN.n_hidden,1);

            % general note:
            % H = [h0,h1,h2,...,hT] -> H(1) = h0
            % X = [x1,x2,...,xT] -> X(1) = x1

            % build stage cost and equality constraints
            obj = 0;
            g = [];
            for i = 1:b
                switch H0_coupling
                    case 0 % TBPTT RNN
                        % fix all initial conditions to zero
                        g = [g; H{i}(:,1)-H0];
                    case 1 % benchmark RNN
                        if i<=delay_idx
                            % first initial condition free
                            g = [g; zeros(RNN.n_hidden,1)];
                        else
                            % couple all remaining initial conditions
                            g = [g; H{i}(:,1)-H{i-delay_idx}(:,delay_idx+1)];
                        end
                    otherwise
                        % all initial conditions are free
                        g = [g; zeros(RNN.n_hidden,1)];
                end

                % cost function and dynamic constraints
                for j = 1:N
                    Dy = RNN.g(H{i}(:,j+1),W_hy,b_y) - Y{i}(:,j);

                    if j>=m+1 % enforce burn-in phase   
                        obj = obj + Dy'*Dy;
                    end
    
                    g = [g; H{i}(:,j+1) - RNN.f(H{i}(:,j),X{i}(:,j),W_hh,W_xh,b_h)];
                end
            end

            % normalize cost
            obj = obj/((N-m)*b);

            % Formulate NLP and solver
            opt_var_H = [];
            opt_par_X = [];
            opt_par_Y = [];
            for i = 1:b
                opt_var_H = vertcat(opt_var_H,reshape(H{i},numel(H{i}),1));
                opt_par_X = vertcat(opt_par_X,reshape(X{i},numel(X{i}),1));
                opt_par_Y = vertcat(opt_par_Y,reshape(Y{i},numel(Y{i}),1));
            end
            opt_var = vertcat(opt_var_H,theta);
            opt_params = vertcat(opt_par_X,opt_par_Y,H0);
            nlp = struct('x',opt_var,'p',opt_params,'f',obj,'g',g);
            opts = struct;
            opts.ipopt.print_level =0;
            opts.print_time = 0;
            solver = nlpsol('solver', 'ipopt', nlp, opts);            
        end


        function solution = solve_mini(solver,H,theta,DATA,H0)

            % Read inputs
            N = size(H{1}.H,2)-1;
            Nh = size(H{1}.H,1);
            b = length(H);

            % Formulate NLP and solver
            opt_var_H = [];
            opt_par_X = [];
            opt_par_Y = [];
            for j = 1:b
                opt_var_H = vertcat(opt_var_H,reshape(H{j}.H,numel(H{j}.H),1));
                opt_par_X = vertcat(opt_par_X,reshape(DATA{j}.X,numel(DATA{j}.X),1));
                opt_par_Y = vertcat(opt_par_Y,reshape(DATA{j}.Y,numel(DATA{j}.Y),1));
            end
            opt_var = vertcat(opt_var_H,theta);
            opt_params = vertcat(opt_par_X,opt_par_Y,H0);

            % constraints   
            bg = zeros(b*(N+1)*Nh,2); % dynamic equality constraints, lb=ub
            b_H = [-inf(size(opt_var_H)),inf(size(opt_var_H))];
            b_theta = [-inf(size(theta)),inf(size(theta))];
            b_Whh = [-reshape(eye(Nh),[],1),reshape(eye(Nh),[],1)]*0.999;
            b_theta(1:Nh*Nh,:) = b_Whh;
            bx = [b_H;b_theta];

            % Solve NLP
            res = solver(...
                'p',opt_params,...
                'x0',opt_var,...
                'lbg',bg(:,1),...
                'ubg',bg(:,2),...
                'lbx',bx(:,1),...
                'ubx',bx(:,2));

            % Get solution
            sol_x = full(res.x);
            solution.f = full(res.f);
            for j = 1:b
                solution_H{j}.H = reshape(sol_x((j-1)*Nh*(N+1)+1:j*Nh*(N+1)),Nh,N+1);
            end
            solution.H = solution_H;
            solution.theta = sol_x(b*Nh*(N+1)+1:end);
        end


    end
end