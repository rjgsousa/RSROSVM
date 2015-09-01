
function [best_options, roc_data] = run( folds, combinations, wr, options)
    
    options.trainsize = folds(1);

    data = struct();
    [ idxTrain, idxVal, idxTest ] = divide_data( options );
    
    fprintf(1,'Doing a %d-fold Cross Validation\n',options.nfolds);
    fprintf(1,'Train size: %d\n',length(idxTrain));
    
    data.idxTrain = idxTrain;
    data.idxTest  = idxTest;
    data.idxVal   = idxVal;
    
    fprintf(1,'Training with %d instances.\n',length(idxTrain));
    
    % ---------------------------------------------------------------------------------
    global datafeatures
    global dataclasses
    switch( options.method )
        % ---------------------------------------------------------------------------------
      case {'threshold','weights'}
        % 1,2,3,.. -> 1,3,5,...
        dataclasses = (dataclasses-1)*2+1;

        options.reduce_now = false;
        options.nclasses   = max(unique(dataclasses));
    end
    truedim = size (datafeatures, 2);
    options = setfield(options, 'trueDim', truedim);
    % ---------------------------------------------------------------------------------

    
    % run models
    fprintf(1,'(START) Time: %s\n',datestr(now,'HH:MM:SS'));
    
    [ best_options, roc_data ] = ...
        runModels1( folds, combinations, wr, options, data );

    return
    
    