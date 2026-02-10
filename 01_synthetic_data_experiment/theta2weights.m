function RNN_weights = theta2weights(theta,RNN)
%THETA2WEIGHTS Summary of this function goes here
%   Detailed explanation goes here

    n_W_hh = RNN.n_hidden^2;
    n_W_xh = RNN.n_hidden*RNN.n_input;
    n_b_h = RNN.n_hidden;
    n_W_hy = RNN.n_hidden*RNN.n_output;
    n_b_y = RNN.n_output;
    
    idx = 0;
    W_hh = reshape(theta(idx+1:idx+n_W_hh),RNN.n_hidden,RNN.n_hidden);
    idx = idx + n_W_hh;
    W_xh = reshape(theta(idx+1:idx+n_W_xh),RNN.n_hidden,RNN.n_input);
    idx = idx+n_W_xh;
    b_h = theta(idx+1:idx+n_b_h);
    idx = idx+n_b_h;
    W_hy = reshape(theta(idx+1:idx+n_W_hy),RNN.n_output,RNN.n_hidden);
    idx = idx+n_W_hy;
    b_y = theta(idx+1:idx+n_b_y);

    RNN_weights.W_hh = W_hh;
    RNN_weights.W_xh = W_xh;
    RNN_weights.b_h = b_h;
    RNN_weights.W_hy = W_hy;
    RNN_weights.b_y = b_y;
end

