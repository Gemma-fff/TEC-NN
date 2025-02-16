%测试案例：设计要求 (Tc=-8.1,) DT=36, P=49.5, Qc=256.2; 应给结果：N=127, I=3.8, L=1.39, W=1.39, H=1;
%测试案例：设计要求 (Tc=-7.3,) DT=31.65, P=15, Qc=202.78; 应给结果：N=71, I=3.75, L=2, W=2.05, H=1.5;
%测试案例：设计要求 (Tc=3.46,) DT=22.7, P=32.7, Qc=642.2; 应给结果：N=71, I=8.2, L=2.05, W=2.1, H=0.5;
%测试案例：设计要求 (Tc=0.29,) DT=23.9 P=13.17, Qc=246.15; 应给结果：N=127, I=1.88, L=1.36, W=1.36, H=1;
%测试案例：设计要求 (Tc=-16.3,) DT=40.7 P=15.2, Qc=101; 应给结果：N=127, I=1.52, L=1.36, W=1.39, H=2.5;
%测试案例：设计要求 (Tc=4.22,) DT=24.8 P=63.55, Qc=490; 应给结果：N=127, I=4.9, L=1.3, W=1.36, H=0.5;
clc;
clear;
close all;

load SJWL.mat net;
load inputs.mat inputs;
load outputs.mat outputs;

%交互输入设计目标

DT = input('请输入期望温差');
P = input('请输入期望消耗电能上限');
Qc = input('请输入期望制冷量');

wdt = input('请输入期望温差所占百分比');
wp = input('请输入期望消耗电能所占百分比');
wqc = input('请输入期望制冷量所占百分比');

goal(1,:) = [1, DT, P, Qc];

goal_1(1,:) = mapminmax('apply', goal' ,outputs );         %输出量归一化目标

%% 初始化种群

N = 20;                                                    % 初始种群个数

d = 5;                                                     % 空间维数

ger = 100;                                                  % 最大迭代次数 

w=0.9;                                                     % 惯性权重

%limit = [71,127 ; 1,10 ; 1.3,2 ; 1.3,2 ; 0.5,2 ];        % 设置位置参数限制,依次表示 N I L W H

limit = [0,1 ; 0,1 ; 0,1 ; 0,1 ; 0,1 ];                    %直接当做x归一化后的值

vlimit = [-1,1 ; -1,1; -1,1 ; -1,1 ; -1,1 ];               % 设置速度限制
                     
c_2 = 0.5;                                                 % 自我学习因子

c_3 = 0.5;                                                 % 群体学习因子 

%初始种群的位置

for i = 1:d     

    x(:,i) = limit(i, 1) + (limit(i, 2) - limit(i, 1)) * rand(N, 1);  

end    

% for i = 1:N
% 
%      x1(i,:) = mapminmax('apply', x(i,:)' ,inputs );           %归一化后的x
% 
% end

%% 固定变量定义

outd = 4;                        %输出变量维数

iter = 1;                        %记录迭代到第几次

hang2 = 1;                       %记录recordx（各粒子初始位置）的行数

hang3 = 0;                       %记录recordfxm（个体最佳结果）行数

%v = rand(N, d);                 %这种每个速度都在[-1,1]    

% 初始种群的速度

for i = 1:d

    v(:,i) = vlimit(i, 1) + 2 * vlimit(i, 2) * rand(N, 1);% 初始种群的速度

end

xm = x;                          % 每个个体的历史最佳位置（归一化x的值）

ym = zeros(1, d);                % 种群的历史最佳位置（归一化x的值）

fxm = ones(N, 1) *inf;           % 每个个体的历史最佳适应度

fym = inf;                       % 种群历史最佳适应度

tic;

%% 记录器初始化

recordfym = zeros(ger, 1);         %用于记录每一代适应度(差值)最优结果

recordfxm = zeros(ger*N,1);        %用于记录每个粒子适应度(差值)

% recordym  = zeros(ger,d);          %用于记录每一代种群最优位置(4个变量值)
% 
% recordxm  = zeros(ger*N,d);        %用于记录每个粒子历史最佳位置(4个变量值)
 
recordx = zeros(ger*N,d);          % 用于记录每个粒子初始化时位置（输入量）

recordfx = zeros(ger*N,outd);       %记录每个粒子经过神经网络后的归一化性能结果

%recordffxm = zeros(ger*N,outd);        %记录每个粒子的反归一后的输出量

%% 迭代求解

while iter <= ger

     for i=1:N
   
        recordx(hang2,:)=x(i,:);                           %记录每个粒子的输入量（初始位置）

        fx(i,:)=sim(net,x(i,:)');                          %得到经过神经网络后的归一化性能结果
      
 %       ffx(i,:)=mapminmax('reverse',fx(i,:)',outputs);     %将结果反归一化
      
        recordfx(hang2,:)=fx(i,:);                         %记录每个粒子经过神经网络后的归一化性能结果
          
        hang2=hang2+1;
     end

%记录每个粒子适应度值

     for i=1:N

         minus(i,:) = wdt * abs(goal_1(1,2)-fx(i,2)) + wp * abs(goal_1(1,3)-fx(i,3)) + wqc * abs(goal_1(1,4)-fx(i,4));    %计算每个粒子归一化输出量与归一化设计要求之间的差值
         
         recordfxm(hang3+i,:) = minus(i,:);              %记录每个粒子的归一化差值（最佳适应度）

     end

     hang3 = hang3 +N;

     for i = 1:N      
        
%         recordx(hang2,:)=x(i,:);
% 
%         hang2=hang2+1;

%         ffx(i)=mapminmax('reverse',fx(i),ps_output);
%        
%         recordfxm(hang,:)=ffx(i);
        
        if  minus(i) < fxm(i) 

            fxm(i) = minus(i);     % 更新个体历史最佳适应度

            xm(i,:) = x(i,:);   % 更新个体历史最佳位置

        end 

%         recordxm(hang,:)=xm(i,:);

%         hang=hang+1;

     end

if min(fxm) < fym

        [fym, nmax] = min(fxm);   % 更新群体历史最佳适应度

        ym = xm(nmax, :);         % 更新群体历史最佳位置

        outym = fx(nmax,:);       % 更新群体最佳输入量时对应的归一化性能输出量

 end

    v = v * w + c_2 * rand *(xm - x) + c_3 * rand *(repmat(ym, N, 1) - x);% 速度更新

    % 边界速度处理

    for i=1:d 

        for j=1:N

        if  v(j,i)>vlimit(i,2)

            v(j,i)=vlimit(i,2);

        end

        if  v(j,i) < vlimit(i,1)

            v(j,i)=vlimit(i,1);

        end

        end

    end       

    x = x + v;% 位置更新

    % 边界位置处理（记得先反归一再比较）

    for i=1:d 

        for j=1:N

        if  x(j,i)>limit(i,2)

            x(j,i)=limit(i,2);

        end

        if  x(j,i) < limit(i,1)

            x(j,i)=limit(i,1);

        end

        end

    end
    %ffym=mapminmax('reverse',fym,ps_output);

    recordfym(iter) = fym;         %记录每一代最好结果

%   recordym(iter,:)=ym;
%     plot(recordfym);
% 
%     title('最优适应度进化过程')  

    iter = iter+1;

   % times=times+1;

end
%fym=postmnmx(fym,ps_output); 
%ym=postmnmx(ym,ps_input);
 plot(recordfym);
 
 title('最优适应度进化过程')  

%my_fym=mapminmax('reverse',fym,ps_output);

bestout = mapminmax('reverse',outym',outputs);
disp(['设计结构性能：',num2str(bestout')]);

bestin = mapminmax('reverse',ym',inputs);
disp(['输入变量取值：',num2str(bestin')]);

toc;