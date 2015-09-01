function str = genSVMstring(options,withprob)
    switch ( options.kernel )
      case 'rbf'
        str = sprintf(' -t 2 -g %f ',options.gamma);
        
      case 'polynomial'
        str = sprintf(' -t 1 -d %f -g %f -r %f ',options.degree,...
                      options.gamma,...
                      options.coef);
        
        
      case 'linear'
        str = sprintf(' -t 0 ');

      case 'intersection'
        str = sprintf(' -t 5 ');

      case 'chisquared'
        str = sprintf(' -t 6 ');

      otherwise
        kakakakkaak
    end
    
    str = sprintf('-s 0 %s -e %f -c %f -q', str, options.epsilon,options.C);

    if nargin == 2
        if withprob
            str = sprintf('%s -b 1',str);
        end
    end
    
    return