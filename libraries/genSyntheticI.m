
%% syntheticI
%
function [data] = genSyntheticI(nSamples)
    theta = [-inf, -.5, 0.25, inf];
    dimension = 2;
    features = rand(nSamples, dimension);
           
    data = [];
    for i = 1:nSamples
        d = features(i,:);
        x = d - 0.5;
        x = prod (x);
        x = x*10;
        stdDev = 0.125;
        epsilon = stdDev*randn; 
        x = x+epsilon;
        class = x;
        class = theta > class;
        class = find(~class);
        class = size(class, 2);
        if ( class == 2 )        
            z = rand;
            xx = (x-theta(2))/(theta(3)-theta(2));
            if (z < xx)
                class = 3;
            else
                class = 1;   
            end
        end
        if class == 3
            class = 2;
        end
        data = [data; d class];
    end 
    
    % dlmwrite('data.csv',data)
    return