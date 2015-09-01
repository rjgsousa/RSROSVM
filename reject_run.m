
%% Reject run
function best_options = reject_run( options, combinations, datasetID )
    
% needed files
    addpath(genpath(options.project_lib_path))
    
    % --------------------------------------------------------------------------------
    % screen size
    % scrsz = get(0,'ScreenSize');
    
    wr      = options.wr;
    nrounds = options.nrounds;
    folds   = options.folds;
    method  = options.method;
    
    global datafeatures
    global dataclasses
    global STREAM
    global IDX
    global IDS
    global UNIQUE_IDS
    global UNIQUE_CLS
    
    first = true;

    options.datasetID = datasetID;
    options.givenIDS  = false;

    pruning = 'false';
    if options.prune
        pruning = 'true';
    end

    % 2 - 10%
    % 4 - 20%
    % 8 - 40%
    for i = 14
        if ~options.givenval
            fprintf(1,'Training with %3d%% of data.\n',folds(i,1)*100);
        else
            fprintf(1,'Using train/val data.\n');
        end
        
        
        %% resets rand seed
        STREAM = RandStream('mrg32k3a');
        RandStream.setDefaultStream(STREAM);

        switch( datasetID )
          case 'NBI_sift'
            vl_twister('state',0);
        end
        
        all_roc1 = zeros( length(wr), nrounds );
        all_roc2 = zeros( length(wr), nrounds );
        all_roc3 = zeros( length(wr), nrounds );
        all_roc4 = zeros( length(wr), nrounds );
        all_roc5 = zeros( length(wr), nrounds );
        
        for k = 1:nrounds
            roc_data = [];
            
            % filename_error  = sprintf('%s_%s_%03d_%s_%c' ,method,options.datasetID,folds(i,1)*100,pruning,options.trial);
            % filename_reject = sprintf('%s_%s_%03d_%s_%c',method,options.datasetID,folds(i,1)*100,pruning,options.trial);

            % switch( options.method )
            %   case 'ssca_del'
            %     filename_error  = sprintf('%s_%01.1f',filename_error,options.ssca_D);
            %     filename_reject = sprintf('%s_%01.1f',filename_reject,options.ssca_D);
                
            % end
            
            % switch( options.datasetID )
            %   otherwise
            %     filename_error  = sprintf('%s_error_tmp_results.mat',filename_error);
            %     filename_reject = sprintf('%s_reject_tmp_results.mat',filename_reject);
            % end
            % -------------------------------------------------------------------------
            % load datasets
            [ datafeatures, dataclasses, IDS, UNIQUE_IDS, UNIQUE_CLS ] = loadDataSets( options, datasetID ); 
            
            if iscell( dataclasses )
                options.nclasses = length(unique(dataclasses{1}));
            else
                options.nclasses = length(unique(dataclasses));
            end
            first = false;
            
            % ----------------------------------------------------------
            if options.givenIDS
                n     = size(UNIQUE_CLS,1);
                IDX   = randperm(STREAM,n);
            else
                n     = length(dataclasses);
                IDX   = randperm(STREAM,n);
            end
            % ----------------------------------------------------------
            
           
            fprintf(1,'------------- %d round ----------------\n',k);

            %t0 = cputime;
            [best_options roc_data] = run( folds(i,:), combinations, wr, options);

            %roc_data
            
            %filename = ['results' options.trial '/best_options_' num2str(i)  '-'  num2str(k) '.mat'];
            %fprintf(1,'Saved best options in %s\n',filename);
            %save(filename,'best_options')

            %fprintf(1,'Method ''%s'' took %f seconds.\n',method,cputime-t0);

            %all_roc1(:,k) = roc_data(:,1);
            all_roc1(:,k) = roc_data(:,2); % performance
            all_roc2(:,k) = roc_data(:,3); % support vector
            all_roc3(:,k) = roc_data(:,4); % threshold
            all_roc4(:,k) = roc_data(:,5); % time
            
            %save(filename_error , 'all_roc1')
            %save(filename_reject, 'all_roc2')

        end
        
        m_roc12=std(all_roc1,0,2);
        m_roc22=std(all_roc2,0,2);
        m_roc32=std(all_roc3,0,2);
        m_roc42=std(all_roc4,0,2);

        m_roc11=mean(all_roc1,2);
        m_roc21=mean(all_roc2,2);
        m_roc31=mean(all_roc3,2);
        m_roc41=mean(all_roc4,2);
        
        roc_data = zeros(length(wr),7);
        roc_data(:,1)=m_roc11; % Perf
        roc_data(:,2)=m_roc12;
        roc_data(:,3)=m_roc21; % SV
        roc_data(:,4)=m_roc22;
        roc_data(:,5)=m_roc31; % threshold
        roc_data(:,6)=m_roc32;
        roc_data(:,7)=wr';
        roc_data(:,8)=m_roc41; % time
        roc_data(:,9)=m_roc42;

        roc_data
        filenameroc  = sprintf('%s_%s_%05d_%03d_%s_%c' ,method,options.datasetID,options.randomset,folds(i,1)*100,pruning,options.trial);

        switch( options.method )
          case 'ssca_del'
            filenameroc = sprintf('%s_%01.1f',filenameroc,options.ssca_D);
        end      
        
        switch( options.datasetID )
          otherwise
            filenameroc  = sprintf('%s_error_vs_reject.txt',filenameroc);
        end
        
        dlmwrite(filenameroc,roc_data);
        
    end

    return

