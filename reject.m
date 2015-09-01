
%% Reject main code file 
% 
% Developed by
% - Ricardo Sousa
%   Researcher at INEB
%   rsousa @ rsousa.org
% - Jaime S. Cardoso
%   Professor at FEUP and Researcher at INESC Porto
%   jsc @ inesctec.pt
%  
% Code License (GNU v3)
function reject(general_opt)
    
    if nargin ~= 1
        % argument sanity check
        usage();
        return;
    end

    % max num of threads
    % MaxNumCompThreads(1);

    warning off all;
    
    % debug info will be written to this file
    % global filename

    
    datasetID = general_opt.datasetID;
    method    = general_opt.method;
    % ----------------------------------------------------------------------------------
    % CONFIGURATION SECTION

    % C-values
    if ~isfield( general_opt, 'C' ) 
        Cvalue = -5:2:3;    %-5:2:10; %-5:2:15;
        Cvalue = 2.^Cvalue;
    else
        Cvalue = general_opt.C;
    end
    
    % Gamma values
    if ~isfield ( general_opt, 'gamma' )
        gamma  = -3:2:-1;
        gamma  = 2.^gamma; 
    else
        gamma  = general_opt.gamma;
    end

    % osvm specific configuration values
    if ~isfield ( general_opt, 'h' )
        h = 1:4; %3:7 - syntheticI
    else
        h = general_opt.h;
    end
    
    if ~isfield ( general_opt, 's' )
        s = [2,4];
    else
        s = general_opt.s;
    end

    % weights 
    wr = 0.04:.2:.48;

    if strcmp(general_opt.method,'sca_flip') == 1 | ...
            strcmp(general_opt.method,'sca_del') == 1 | ...
            strcmp(general_opt.method,'ssca_del') == 1 | ...
            strcmp(general_opt.method,'standard') == 1
        wr = 0.04; % this value does not matter
    end

    if strcmp(general_opt.method,'sca_flip') == 1 | ...
            strcmp(general_opt.method,'sca_del') == 1 | ...
            strcmp(general_opt.method,'ssca_del') == 1 | ...
            strcmp(general_opt.method,'standard') == 1
        general_opt.prune = true;
    end
    
    % precision 
    epsilon = 1e-5;
    
    % kernel degree
    degree  = general_opt.degree;

    % folds
    foldsx  = .05:.05:.9;
    folds   = [foldsx' 1-foldsx'];
    nfolds  = 3;
    
    %pause
    paramRange = struct('C',Cvalue,'gamma',gamma)
    
    parameters = struct();
    %parameters.k         = k;
    parameters.Cvalue    = Cvalue;
    parameters.gamma     = gamma;
    parameters.h         = h;
    parameters.s         = s;
    
    % number of rounds
    nrounds = general_opt.nrounds; 

    % kernel type
    kerneltype = general_opt.kernel;

    switch (method)
      case {'standard','sca_flip','sca_del','ssca_del','threshold','weights'}
        kernel = kerneltype;
        
      otherwise
        fprintf(1,'(reject.m) error: method ''%s'' unknown\n',method);
        usage();
        return
    end

    if ( strcmp(method,'threshold') == 1 )
        probability = 1;
    else
        probability = 0;
    end
    
    % -------------------------------------------------------
    % lets combine all them  
    combinations = combine_parameters( general_opt, parameters);
    method_parameter = 1;
    
    % ----------------------------------------------------------------------------------
    % specific options for the my_svm_dual_train
    options = struct('trial',general_opt.trial,'epsilon',epsilon,'method',method,'method_parameter',method_parameter); 
    options.project_lib_path = general_opt.project_lib_path; %
    options.workmem      = 1024;
    options.test         = false;
    % dataset 
    options.givenval     = general_opt.givenval;
    options.randomset    = general_opt.randomset;
    % pruning specific options
    options.prune        = general_opt.prune;
    options.reduce_now   = false;
    options.submethod    = 'libsvm';
    % bag of features
    options.usenclusters = 1000; %2^2;
    % SVM Specific Options
    options.coef         = 1;
    options.kernel       = kernel;
    options.degree       = degree;
    options.weights      = 1;
    options.wr           = wr;
    options.folds        = folds;
    options.nfolds       = nfolds;
    options.nrounds      = nrounds;
    options.probability  = probability;
    options.maxiter      = 15000;
    options.optimization = 'cplex';
    options.C        = [];
    options.gamma    = [];
    options.ssca_D   = general_opt.ssca_D;
    
    if ( method_parameter == 0 && ...
         ( strcmp(options.method,'threshold') == 1 ) )
        options.nbins = 1;
    end

    switch (options.submethod)      
      case 'libsvm'
        addpath(fullfile(general_opt.project_lib_path, 'libsvm-3.17/matlab/'))

      otherwise
        which_method_are_you_going_to_run
    
    end
    options

    fprintf(1,'Using dataset ''%s'' on method ''%s'' \n', datasetID, method );
    % ----------------------------------------------------------------------------------

    best_options = reject_run( options, combinations, datasetID);

    return

%%                                                                                         
%
function usage
    fprintf(1,'You must identify the method and datasetID\n\n');
    fprintf(1,'Usage: ./reject method datasetID\n');
    fprintf(1,'method: \n');
    fprintf(1,['\t- threshold\n'...
               '\t- weights\n']);
    fprintf(1,'datasetID: \n');
    fprintf(1,['\t- syntheticI\n'...
               '\t- others'...
               '\n\n']);

    return

    
%%                                                                                         
%
function combinations = combine_parameters( options, parameters)

    switch( options.method )        
      case {'threshold', 'weights', 'sca_flip', 'sca_del', 'ssca_del', 'standard'}
        switch( options.kernel )
          case 'linear'
            combinations = parameters.Cvalue;
          otherwise
            combinations = combvec( parameters.Cvalue, parameters.gamma );
        end

      otherwise
        str = sprintf('Method ''%s'' unknown.\n',options.method);
        error(str);
    end
    
    return