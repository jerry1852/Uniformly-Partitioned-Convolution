L=64;
h = randn(500,1);
x = randn(5000,1);

st = pfir_init(h, L);

y = zeros(size(x));
n = numel(x);
t = 1;


while t <= n
    xblk = x(t:min(t+L-1,n));
    if numel(xblk) < L
        xblk(end+1:L) = 0;   % pad last block
    end

    [yblk, st] = pfir_process(st, xblk);

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



function st = pfir_init(h, L)
    %PFIR_INIT  Partitioned FIR (time-domain) state init (uniform partitions).
    %   h : impulse response (vector)
    %   L : block size (new samples per process call)

    %   h(n) = sum_{k=0}^{P-1} h_k(n - kL),  with h_k length-L segments
    %   y(n) = sum_{k=0}^{P-1} x(n - kL) * h_k(n)

    h=h(:);

    p=ceil(numel(h)/L);% caculate the number of block
    hpad=[h; zeros(p*L-numel(h), 1)]; % here is special for the last block, make sure that the length L

    h_matrix=zeros(L, p);

    for i=1:p
        idx=(i-1)*L+(1:L);
        h_matrix(:, i)=hpad(idx);
    end

    st.h=h;
    st.h_matrix=h_matrix;
    st.L=L;

    st.xhist=zeros(L, p); % here is the input x signal.
    st.p=p;

    st.pos   = 1;                       % write pointer
    % 为什么 ola 长度是 L-1：因为每次短卷积 conv(L, L) 产生 2L-1 点，
    % 本块只输出前 L 点，剩下 L-1 点对应未来时间，需要存起来下次叠加。
    st.ola   = zeros(L-1, 1);           % overlap-add tail

end


function [yblk, st] = pfir_process(st, xblk)
    %PFIR_PROCESS  Process one block (length L) using partitioned convolution.
    %   Input:
    %     st   : state from pfir_init
    %     xblk : new input samples (length L)
    %   Output:
    %     yblk : output samples for this block (length L)
    %     st   : updated state

    l=st.L;
    p=st.p;
    xblk=xblk(:);

    st.xhist(:, st.pos)=xblk;
    acc=zeros(2*l-1, 1);  %(L+L-1)
    
    % 需要注意的一点是，尽管公式中是x(n-kl)但实际上因为是实时计算，所以每次传入的x并不是全部，而是一个一个的区块，长度和分块滤波器的
    %长度相同。因此，这个式子也可以理解为，xt，xt-1,xt-2这样的的区块。在这里，引入了p个区块，并且是循环loop采纳并计算。

    for k=0:p-1
        idx=st.pos-k;
        while idx<1
            idx=idx+p;
        end
        
        % hk[n]和x(n-kl)，当前时间点是pos
        h_k=st.h_matrix(:, k+1);
        x_k=st.xhist(:, idx);

        acc=acc+conv(x_k, h_k);% do the convution

         

    end

    % 这里需要注意的是，卷积输出的是2l-1长度的信号，而我们应该需要的是前半部分，l长度的信号。但是后面的我们也需要，因为
    % 当有新的块进来的时候，旧的块会被挤出。过去的信号会对未来的输出产生影响，尽管我们始终只有p个块，但ola却可以计算出
    % 考虑到过去影响的输出。旧块在内存里被覆盖，但它对未来输出的最后 L-1 个样本贡献必须先存进 ola，保证卷积等价。
    % acc(1:L)   -> 对应当前块时间 [tL, tL+L-1] 的输出
    % acc(L+1:end)-> 对应未来时间 [tL+L, tL+2L-2]，不能现在输出，但不能丢
    yblk=acc(1:l);

    yblk(1:l-1) = yblk(1:l-1) + st.ola;
    st.ola = acc(l+1:end);  

    % 找好位置放入输入块
    st.pos = st.pos + 1;
    if st.pos > p
        st.pos = 1;
    end
end





