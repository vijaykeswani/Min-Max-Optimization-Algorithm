clear

eta = 0.05;

x0 = 5.5;
y0 = 5.5;

%%%To choose the loss function, go to the botton of the file.







%Window size
B=20





v =(-B:0.5:B); Y = v'*ones(1,length(v)); X = Y'; Z = value(X,Y);surfc(X,Y,Z-2)
set(gca,'Xlim',[-B,B], 'Ylim', [-B,B],'Zlim', [-30,30])
view(43,40)


%curve_globalcurve_global = animatedline([0,0],[-B,B],[0-0.5,0-0.5],'Color','r','LineWidth',5)

i=0
title(['i = ' num2str(i)], 'FontSize', 30, 'FontName', 'Times','FontWeight', 'normal')


curve1 = animatedline('LineWidth',4)
curve2 = animatedline('LineWidth',4)

%pause



T = 401

x1 = x0
y1 = y0
z1= value(x1,y1)

%x2 = 0.6
%y2 = 0.6
%z2= x2*y2

hold on
for t = 1:T
    i=t-1
   addpoints(curve1,x1(t),y1(t),z1(t))
   %addpoints(curve2,x2(t),y2(t),z2(t))
    title(['i = ' num2str(i)], 'FontSize', 30, 'FontName', 'Times','FontWeight', 'normal')

   drawnow
   pause(0.05)
  % plot(x(t),y(t),z(t)))
   
   y1(t+1) = y1(t) + eta*yGrad(x1(t),y1(t));
   x1(t+1) = x1(t) - eta*xGrad(x1(t),y1(t));
   z1(t+1) = value(x1(t),y1(t));

   [x1; y1]
   
   %x2(t+1) = x2(t) - eta*y2(t);
   %y2(t+1) = y2(t) + eta*x2(t);
   %z2(t+1) = x2(t+1)*y2(t+1);

   
   
   %
%    frame = getframe(h);
%    im = frame2im(frame);
%    [imind,cm] = rgb2ind(im,256);
%    if t==1
%        imwrite(imind,cm,filename,'gif','Loopcount',inf);
%    else
%        imwrite(imind,cm,filename,'gif','Writemode','append');
%    end

    fid = fopen('b_gda_acc.txt', 'w');
    fprintf(fid, '%d,%d ', [x1; y1]);
    fclose(fid);



end
hold off


%%%Input a formular for the loss function and its gradients

%%%Loss function formula
function z = value(x, y)
%%%%F1
%    z = -3*x.^2 - y.^2 +4*x.*y;
%%%%F2
%     z =  3*x.^2 + y.^2 +4*x.*y; 
%%%%F3
  z= (4*x.^2 - (y - 3*x +0.05*x.^3).^2 - 0.1*y.^4).*exp(-0.01*(x.^2+y.^2));
end

%%%formula for x-gradient
function g = xGrad(x, y)
%%%nabla_x F1
%    g = -6*x + 4*y;
%%%nabla_x F2
%     g = 6*x + 4*y;
%%%nabla_x F3
   g = -0.02*x*(4*x.^2 - (y - 3*x +0.05*x.^3).^2 - 0.1*y.^4).*exp(-0.01*(x.^2+y.^2)) +(8*x-2*(y - 3*x +0.05*x.^3)*(-3+0.15*x.^2)).*exp(-0.01*(x.^2+y.^2));
end

%formula for y-gradient
function g = yGrad(x, y)
%%%nabla_y F1
%    g = -2*y + 4*x;
%%%nabla_y F2
%      g = 2*y + 4*x;
%%%nabla_y F3
   g = -0.02*y*(4*x.^2 - (y - 3*x +0.05*x.^3).^2 - 0.1*y.^4).*exp(-0.01*(x.^2+y.^2)) + (-2*(y - 3*x +0.05*x.^3) -0.4*y^3)*exp(-0.01*(x.^2+y.^2));
end