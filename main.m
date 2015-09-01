
function main(method,prune,datasetID,givenval,degree,nround,gamma,randomset,D)

    if nargin == 8
        D = 0;
    end
    
    % method:
    %     - standard (libsvm, no pruning)
    %     - weights
    %     - threshold
    %     - sca_flip
    %     - sca_del
    %     - reducoSVM
    
    basepath =  pwd;

    
    general_opt = struct();
    general_opt.method      = method;
    general_opt.prune       = prune;
    general_opt.datasetID   = datasetID; 
    general_opt.givenval    = givenval;
    general_opt.nrounds     = nround;
    general_opt.project_lib_path = fullfile(basepath,'libraries/');
    % learning options
    general_opt.kernel      = 'polynomial';
    general_opt.degree      = degree;

    general_opt.C           = 100; %[2.^(-5:2:5)]; % [7];
    general_opt.gamma       = gamma; % [2^-1 2];
    general_opt.h           = 1; %[10];
    general_opt.s           = [3];  % [];  % 2;
    general_opt.ssca_D      = D;
    general_opt.randomset   = randomset;
    % --------------------------------------------------------------
    % changes after this line is of your responsability
    % --------------------------------------------------------------
    general_opt.trial       = '1';
    general_opt.learningrate  = ''; 
    
    reject(general_opt);
    
return