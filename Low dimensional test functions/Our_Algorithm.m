clear


eta = 0.05
sigma = 0.5

%Initial points
x0 = 5.5;
y0 = 5.5;




delta = 0.03
eps = 0.001

max_rejects = 20;

%%%To choose the loss function, go to the botton of the file.





%plot window size
B = 20


h = figure
filename = 'test2.gif'

v =(-B:0.5:B); Y = v'*ones(1,length(v)); X = Y'; 

Z = value(X,Y);

surfc(X,Y,Z-2)
set(gca,'Xlim',[-B,B], 'Ylim', [-B,B],'Zlim', [-30,30])
view(43,40)
title(['i = ' num2str(i)], 'FontSize', 30, 'FontName', 'Times','FontWeight', 'normal')


curve1 = animatedline('LineWidth',6)
curve2 = animatedline('LineWidth',6)

%curve_global = animatedline([0,0],[-5,5],[0-0.5,0-0.5],'Color','r','LineWidth',5)

i=0
title(['i = ' num2str(i)], 'FontSize', 30, 'FontName', 'Times','FontWeight', 'normal')







%max number of iterations
T = 20000





%%%%%%%%%%%%%%%%%
x1 = x0;
y1 = y0;
z1= value(x1,y1);
x_outer = x1
y_outer = y1

x2 = 0.6
y2 = 0.6
z2= value(x2,y2)





x_old=x1;
y_old = y1;
z_old = 1000;


x1_accepted = [];
y1_accepted = [];
z1_accepted = [];

min_propose = false;
accepted = true;
inner_loop_beginning=false;
t_inner_start=1;
reject_count = 0;



for t = 1:T
   inner_loop_beginning=false;
    

   
   addpoints(curve1,x1(t),y1(t),z1(t))
   
   drawnow
   if reject_count > max_rejects
      return    
   end

      % pause(0.1)

   x1(t);
   y1(t);
   z1(t);

  
  if min_propose == true
    x_old=x1(t);
    y_old = y1(t);
    z_old = z1(t);



if t>1
i = i+1;
end
title(['i = ' num2str(i)], 'FontSize', 30, 'FontName', 'Times','FontWeight', 'normal')

    x1(t+1)= x1(t) - sigma*randn;
    min_propose = false;
    
    inner_loop_beginning=true;
    t_inner_start = t;
  else
      x1(t+1)= x1(t);
  end
  
  y1(t+1) = y1(t) + eta*yGrad(x1(t),y1(t));
  z1(t+1) = value(x1(t+1), y1(t+1));

  
 stop_inner_loop =false;
  %if  z1(t+1) <=  z1(t) +0.001*eps && inner_loop_beginning==false
  if  norm(yGrad(x1(t),y1(t))) <=  eps && inner_loop_beginning==false
     stop_inner_loop =true; 
     %pause
  end
  
  
  
      
   pauser = stop_inner_loop;
  if stop_inner_loop ==true 
      
      
      if z1(t+1) >= z_old-delta;
      %reject
      x1(t+1) = x_old;
      y1(t+1) = y_old;
      z1(t+1) = z_old;
      
      %save the values of x and y at the end of the outer loop  
      %
clear curve1
%
v =(-B:0.5:B); Y = v'*ones(1,length(v)); X = Y'; Z = value(X,Y);surfc(X,Y,Z-2)
set(gca,'Xlim',[-B,B], 'Ylim', [-B,B],'Zlim', [-30,30])
view(43,40)
title(['i = ' num2str(i)], 'FontSize', 30, 'FontName', 'Times','FontWeight', 'normal')


curve1 = animatedline('LineWidth',5)
addpoints(curve1,x1_accepted,y1_accepted,z1_accepted)



      accepted = false;
      reject_count = reject_count + 1;
      else
          x1_accepted = [x1_accepted, x1(t_inner_start:(t+1))];
          y1_accepted = [y1_accepted, y1(t_inner_start:(t+1))];
          z1_accepted = [z1_accepted, z1(t_inner_start:(t+1))];
          reject_count = 0;

            fid = fopen('b_mx_acc.txt', 'w');

            % fprintf(fid, '%d,%d ', [x1_accepted; y1_accepted]);
            fprintf(fid, '%d,%d ', [x1_accepted; y1_accepted]);
            fclose(fid);
      

      end
      
           
      
      drawnow    
      min_propose = true;
      accepted = true;
  end

   
  z1(t+1) = value(x1(t+1),y1(t+1));
  
   x_outer(i+2) = x1(t+1);
   y_outer(i+2) = y1(t+1);
   z_outer(i+2) = z1(t+1);   
  
    fid = fopen('b_mx_outer.txt', 'w');

    fprintf(fid, '%d,%d ', [x_outer; y_outer]);
    fclose(fid);

  

   

   
   
 
   if pauser == true
   end
   
   

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