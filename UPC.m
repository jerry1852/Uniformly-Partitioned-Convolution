L=64;
h = randn(500,1);
x = randn(5000,1);

st = upfir_init(h, L);

y = zeros(size(x));
n = numel(x);
t = 1;


while t <= n
    xblk = x(t:min(t+L-1,n));
    if numel(xblk) < L
        xblk(end+1:L) = 0;   % pad last block
    end

    [yblk, st] = upfir_process(st, xblk);

    % write back (truncate last block if needed)
    len = min(L, n - t + 1);
    y(t:t+len-1) = yblk(1:len);

    t = t + L;
end

% Reference
y_ref = filter(h, 1, x);

% Relative error
rel_err = norm(y - y_ref) / (norm(y_ref) + 1e-12);
fprintf("rel err = %.3e\n", rel_err);

figure();
subplot(2, 1, 1);
plot(y);
subplot(2,1,2);
plot(y_ref);



function st = upfir_init(h, L)
    %PFIR_INIT  Partitioned FIR (time-domain) state init (uniform partitions).
    %   h : impulse response (vector)
    %   L : block size (new samples per process call)

    %   h(n) = sum_{k=0}^{P-1} h_k(n - kL),  with h_k length-L segments
    %   y(n) = sum_{k=0}^{P-1} x(n - kL) * h_k(n)

    h=h(:);

    p=ceil(numel(h)/L);% caculate the number of block
    hpad=[h; zeros(p*L-numel(h), 1)]; % here is special for the last block, make sure that the length L

    % 这里2l是filter和input的规定长度，包含现在和过去的数据。
    n=2*L;
    h_matrix=complex(zeros(n, p));

    

    for i=1:p
        idx=(i-1)*L+(1:L);
        h=hpad(idx);

        % 这里进行了修改，我们不再需要简单的时域上的信号，而是频域上的。
        hn_matrix=[h; zeros(L, 1)];
        
        h_matrix(:, i)=fft(hn_matrix, n);
    end

    st.h=h;
    st.h_matrix=h_matrix;
    st.L=L;

    st.xhist=complex(zeros(n, p)); % here is the input x signal.
    st.p=p;

    st.pos   = 1;                       % write pointer

    st.prev   = zeros(L, 1);           % overlap-add tail
    st.n=n;
end


function [yblk, st] = upfir_process(st, xblk)
    %PFIR_PROCESS  Process one block (length L) using partitioned convolution.
    %   Input:
    %     st   : state from pfir_init
    %     xblk : new input samples (length L)
    %   Output:
    %     yblk : output samples for this block (length L)
    %     st   : updated state

    l=st.L;
    p=st.p;
    n=st.n;
    
    % 这里进行了修改，把input补充成2l的长度
    xblk=xblk(:);
    x_in=[st.prev; xblk];
    x=fft(x_in, st.n);

    st.xhist(:, st.pos)=x;
    
    
    % 需要注意的一点是，尽管公式中是x(n-kl)但实际上因为是实时计算，所以每次传入的x并不是全部，而是一个一个的区块，长度和分块滤波器的
    %长度相同。因此，这个式子也可以理解为，xt，xt-1,xt-2这样的的区块。在这里，引入了p个区块，并且是循环loop采纳并计算。
    
    y=complex(zeros(n, 1));

    for k=0:p-1
        idx=st.pos-k;
        while idx<1
            idx=idx+p;
        end
        
        % 频域乘积然后再累加 Y(ω) = Σ X_{k−l}(ω) · H_l(ω)

        y=y + st.xhist(:, idx) .* st.h_matrix(:, k+1);

         

    end

    % get the time domain output, OLS "save": discard first L (time-aliased), keep last L
    yblk = y2L(l+1:end);
    st.prev = xblk;


    % 找好位置放入输入块
    st.pos = st.pos + 1;
    if st.pos > p
        st.pos = 1;
    end
end





