function [trainIDX, toRemove ] = changeDataset( toRemove, trainIDX, options )
    %% function to modify the dataset    
    
    global dataclasses

    % nothing to remove
    if sum(toRemove) == 0
        return;
    end
    
    % there is at least two classes after removal?
    tmp = unique( dataclasses(trainIDX(~toRemove)) );

    % small hack to avoid reducing the whole training set
    % or having one-class problem
    % preserves 10% of the subset
    if ( length( tmp ) < 2 )
        for k=1:options.nclasses
            idxToRemove = find( toRemove == 1 & dataclasses(trainIDX) == k );
            
            idx = 1:.1*length(idxToRemove);
            
            toRemove(idxToRemove(idx)) = 0;
        end
    end
    
    switch( options.method )
      case {'sca_del','ssca_del','threshold','weights'}
        % cleans datasets with rejected/misclassified instances
        trainIDX(toRemove) = []; 
        
      case 'sca_flip'
        ;
        
      otherwise
        myerror(' (changeDataset) method unknown ');
    end
    
    
    return
