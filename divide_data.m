function [idxTrain, indexVal, idxTest ] = divide_data( options )
    global datafeatures
    global dataclasses
    global IDX
    global IDS
    global UNIQUE_CLS
    global UNIQUE_IDS

    range = 1:options.nclasses;
    
    %% training and testing data indexes
    % idxTrain = 1:trainSize;
    % idxTest  = trainSize+1:nelements;
    % split data
    
    if options.givenIDS
        totalIMGS = 0;
        idxTrain  = [];

        for ki=1:length(range)
            k       = range(ki);
            idx     = find( UNIQUE_CLS == k );
            % permutation on the ids
            idx     = idx(randperm(length(idx)));
            
            nelem   = round( length(idx)*options.trainsize );

            totalIMGS = totalIMGS+nelem;
            TRAIN_UNIQUE_IDS = UNIQUE_IDS(idx(1:nelem));
            
            idxTrain = [idxTrain; TRAIN_UNIQUE_IDS];
        
        end
        
        remain      = mod(length(idxTrain),options.nfolds);
        quoc        = floor(length(idxTrain)/options.nfolds);
        itemsremain = options.nfolds-((quoc+remain)*options.nfolds-length(idxTrain));

        if itemsremain<0
            itemsremain = options.nfolds+itemsremain;
        end
        
        idxTrain = idxTrain(1:end-itemsremain);
        indexVal = reshape(idxTrain,options.nfolds,floor(length(idxTrain)/options.nfolds));

        if length( unique(indexVal) ) ~= length( indexVal(:))
            myerror( 'non-unique items' );
        end
        
        idxTest  = setxor(idxTrain,UNIQUE_IDS);

        % idxTrain
        % idxTest
        
        % length(idxTrain)
        % length(idxTest)

        % kk
        
        fprintf(1,'\n\nTraining with %d images.\n',totalIMGS);
    else
        if options.givenval
            dataclasses_tmp = dataclasses;
            
            if isempty( dataclasses{2} ) % only training data given
                dataclasses = dataclasses{1};                
            else
                dataclasses = dataclasses{2};
            end

            IDX = 1:length(dataclasses);
            options.trainsize = 1;
        end
        
        
        idxTrain2 = [];
        for ki=1:length(range)
            k = range(ki);
            idx       = find( dataclasses(IDX) == k );
            nelem     = round(length(idx)*options.trainsize);
            idxTrain2 = [idxTrain2, IDX(idx(1:nelem))];
        end

        % put at all classes in each fold
        remain  = mod(length(idxTrain2),options.nfolds);
        quoc    = floor(length(idxTrain2)/options.nfolds);
        itemsremain = (quoc+remain)*options.nfolds-length(idxTrain2);
        if remain ~= 0
            idxTrain2 = [idxTrain2, idxTrain2(end-itemsremain+1:end)];
        end
        indexVal  = reshape(idxTrain2,options.nfolds,length(idxTrain2)/options.nfolds);
        
        % If does not have an example per class in each fold, include:
        maxnumbermissing = 0;
        for nfolds=1:options.nfolds
            nclassesval      = unique(dataclasses(indexVal(nfolds,:)));
            classesmissing   = setxor(nclassesval,range);
            maxnumbermissing = max(maxnumbermissing,length(classesmissing));
        end
        nitems   = size(indexVal,2);
        indexVal = [indexVal, ones(options.nfolds,maxnumbermissing)];
        
        for nfolds=1:options.nfolds
            nclassesval      = unique(dataclasses(indexVal(nfolds,:)));
            classesmissing   = setxor(nclassesval,range);
            step = 1;
            for n=1:length(classesmissing)
                I = find( dataclasses == classesmissing(n) );
                indexVal(nfolds,nitems+step) = I(1);
                step = step+1;
            end
        end
        idxTrain  = indexVal(:);
        idxTest   = setxor(idxTrain,IDX);

        if options.givenval
            nTrain = length( dataclasses_tmp{1} );
            nVal   = length( dataclasses_tmp{2} );
            nTest  = length( dataclasses_tmp{3} );

            idxTrain = 1:(nTrain+nVal);
            if nVal ~= 0 % if validation data was given
                indexVal = indexVal+nTrain;
            else
                datafeatures = [datafeatures(1); datafeatures(3)];
                dataclasses_tmp = [dataclasses_tmp(1); dataclasses_tmp(3)];
            end
            idxTest  = (nTrain+nVal+1):(nTrain+nVal+nTest);
            
            % [min(idxTrain), max(idxTrain)]
            % [min(indexVal(:)), max(indexVal(:))]
            % [min(idxTest), max(idxTest)]
            % kk
            
            datafeatures = cell2mat(datafeatures);
            dataclasses  = cell2mat(dataclasses_tmp);
        end
    end
    
    return
