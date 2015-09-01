
function [best_options, roc_data] = runModels1(folds, combinations, wr, options,data)
    global datafeatures
    global dataclasses
    global IDS
    global flag 
    flag   = false;
    
    nfolds = options.nfolds;
    
    % n-fold cross validation
    foldsIDX = 1:nfolds;
    
    idxTrain  = data.idxTrain;
    idxTest   = data.idxTest;
    idxVal    = data.idxVal;

    options.nclassesOrig = options.nclasses;
    
    % wr for non reduced set methods is already controlled in the
    % reject.m code
    switch( options.method )
      case {'weights'}
        wrprime = wr;

      case {'threshold','sca_flip','sca_del','ssca_del','standard'}
        wrprime = 1;
    end
    
    roc_data = [];
    fprintf(1,'(START) Date: %s\n',datestr(now,'yyyy/mm/dd'));
    fprintf(1,'(START) Time: %s\n',datestr(now,'HH:MM:SS'));
    
    for z=1:length(wrprime)
        % some initializations
        best_error = inf; %*ones(options.nensemble,1);
        
        fprintf(1,'wr = %.3f\n',wrprime(z));
        
        if size(combinations,2) > 1
            for i=1:size(combinations,2)
                % set parameters values
                options = setParameters(i,combinations,options);
                
                % options.SOMconfig
                % error 
                error   = zeros(1,nfolds);
                
                for k=1:nfolds
                    % k-cross validation
                    idx = setxor(1:options.nfolds,k);
                    
                    % get folds indexes
                    trainIDX = idxVal(idx,:);
                    trainIDX = trainIDX(:)';
                    
                    if options.randomset > 0
                        myrandomsetidx = randperm(length(trainIDX));
                        trainIDX = trainIDX(myrandomsetidx(1:options.randomset));
                    end
                    valIDX   = idxVal(k,:);
                        
                    switch( options.method )                  
                        % ---------------------------------------------------------------------------------
                      case {'weights','sca_flip','sca_del','ssca_del','standard'}
                        [predict, ytrue, ~, ~, fail] = runSpecificModelsI(trainIDX,valIDX,wrprime(z),options);
                        
                      case 'threshold'
                        [predict ytrue prob net] = runSpecificModelsII(trainIDX,valIDX,wrprime(z),options);
                        
                    end
                    
                    if options.prune && ( strcmp(options.method,'threshold') ~= 1 && ...
                                          strcmp(options.method,'sca_flip') ~= 1 && ...
                                          strcmp(options.method,'sca_del') ~= 1 &&...
                                          strcmp(options.method,'ssca_del') ~= 1 &&...
                                          strcmp(options.method,'standard') ~= 1 )
                        error(k) = calcErrorReject(predict,ytrue,valIDX,wrprime(z),options);
                    else
                        error(k) = mean( predict ~= ytrue );
                    end
                    
                end
                
                % average fold errors
                error( isinf(error) ) = [];
                perror = mean(error);
                
                if perror < best_error
                    best_options  = options;
                    best_error    = perror;
                end
            end
        
        else % not cross-val needed
            options = setParameters(1,combinations,options);
            best_options = options;
        end
        
        fprintf(1,'Performing final train/test.\n');
        if options.randomset > 0
            myrandomsetidx = randperm(length(idxTrain));
            idxTrain = idxTrain(myrandomsetidx(1:options.randomset));
        end
        fprintf(1,'\t%d training points.\n',length(idxTrain));

        
        %% Results for Test data.
        
        switch( options.method )
          case {'weights','sca_flip','sca_del','ssca_del','standard'}
            if best_options.prune
                best_options.reduce_now = true;
            end
            
            [ predict, ytrue, totalSV, testtime, fail ] = runSpecificModelsI(idxTrain,idxTest,wr(z),best_options);
            
            if best_options.prune
                perf = mean( predict == ytrue );
            else
                irej = ~mod(predict,2);
                RR   = sum(irej)/length(idxTest);
                idx  = ~irej;
                
                trueClass          = ytrue(idx);
                predictOutOfReject = predict(idx);
                
                erro = find(trueClass ~= predictOutOfReject);
                ER   = length(erro)/ length(idxTest);
                
                perf = 1 - (ER + wr(z)*RR);
            end
            
            best_options.reduce_now = false;
            
            if ~fail
                roc_data = [roc_data; wrprime(z) perf mean(totalSV) 0 testtime];
            else
                fprintf(1,'!!Fail\n');
            end
          
          case 'threshold'
            roc_data = runSpecificModelsIII( idxTrain, idxTest, wr, best_options, roc_data);
            
          otherwise
            myerror ( '(runmodels) method unknown' );
        end
        options.test = false;
    end

    % fprintf(1,'(END) Time: %f\n',cputime);
    
    fprintf(1,'wr   perf SVs\n');
    for item=1:size(roc_data,1)
        fprintf(1,'%.2f %.2f %.2f\n',roc_data(item,1),roc_data(item,2), ...
                roc_data(item,3));
    end
    fprintf(1,'\n\n');
    return;
    
%%                                                                                                     
%
function [predict ytrue totalSVs testtime fail] = runSpecificModelsI(trainIDX,valIDX,wr,options)
    global dataclasses
    global datafeatures
    fail = 0;  predict = 0;

    switch( options.method )
      % ----------------------------------------------------------------------
      case 'standard'
        model            = trainModels(trainIDX, wr, options);
        
        starttime = tic;
        [predict, ytrue] = testModels(valIDX, model, options);
        testtime   = toc(starttime);

        totalSVs = obtainNumberSVs(model,options);
        
        % ----------------------------------------------------------------------
      case 'sca_del'

        model    = trainModels(trainIDX, wr, options);
        [predict, ytrue, idxAll]  = testModels(trainIDX, model, options);

        ind = predict ~= ytrue;
        trainIDX = changeDataset(ind,trainIDX,options);

        model    = trainModels(trainIDX, wr, options);
        starttime = tic;
        [predict, ytrue]  = testModels(valIDX, model, options);
        testtime   = toc(starttime);

        totalSVs = obtainNumberSVs(model,options);

      case 'ssca_del'

        model    = trainModels(trainIDX, wr, options);
        [predict, ytrue, idxAll, sign]  = testModels(trainIDX, model, options);
                
        ind = ytrue ~= predict;

        fprintf(1,'\n\t num train. instances: %.1f\n',length(trainIDX));
        fprintf(1,'\t rejected: %.1f\n',sum( sign < options.ssca_D ));
        fprintf(1,'\t misclassified: %.1f\n',sum(ind));

        ind = ind | ( sign < options.ssca_D ) ; 
        
        trainIDX = changeDataset(ind,trainIDX,options);
                
        % % --------------------------------------------------------
        % SVs1     = full(model.SVs);
        % SVs1cls  = (model.sv_coef > 0)+1;
        % modelaux = model;
        % % --------------------------------------------------------

        model    = trainModels(trainIDX, wr, options);
        starttime = tic;
        [predict, ytrue]  = testModels(valIDX, model, options);
        testtime   = toc(starttime);

        % % ---------------------------------------
        % options
        % SVs2 = full(model.SVs);
        % SVs2cls  = (model.sv_coef > 0)+1;

        % indplot = dataclasses(trainIDX_tmp)<2;
        % h0 = figure;
        % plot(datafeatures(trainIDX_tmp( indplot),1),datafeatures(trainIDX_tmp( indplot),2),'ro')
        % hold on
        % plot(datafeatures(trainIDX_tmp(~indplot),1),datafeatures(trainIDX_tmp(~indplot),2),'g+')

        % h1 = figure;
        % plot(datafeatures(trainIDX_tmp( indplot),1),datafeatures(trainIDX_tmp( indplot),2),'ro')
        % hold on
        % plot(datafeatures(trainIDX_tmp(~indplot),1),datafeatures(trainIDX_tmp(~indplot),2),'g+')
        % plot(SVs1(:,1),SVs1(:,2),'ko','MarkerSize',10)

        % X = linspace(0,1);
        % [X0 X1] = meshgrid(X,X);
        % X00 = X0(:);
        % X11 = X1(:);
        % predict1 = testModels([],modelaux,options,[X00 X11]);
        % contour(X0,X1,reshape(predict1,100,100))
        % unique(predict1)
        
        % h2 = figure;
        % plot(datafeatures(trainIDX_tmp( indplot),1),datafeatures(trainIDX_tmp( indplot),2),'ro')
        % hold on
        % plot(datafeatures(trainIDX_tmp(~indplot),1),datafeatures(trainIDX_tmp(~indplot),2),'g+')
        % plot(SVs2(:,1),SVs2(:,2),'ko','MarkerSize',10)
        
        % predict2 = testModels([],model,options,[X00 X11]);
        % contour(X0,X1,reshape(predict2,100,100))

        % % saveas(h0,'fig0.png','png')
        % % saveas(h1,'fig1.png','png')
        % % saveas(h2,'fig2.png','png')

        % size(SVs1)
        % size(SVs2)
        
        % dlmwrite('synthetic_train.csv',[datafeatures(trainIDX_tmp,:),dataclasses(trainIDX_tmp)]);
        % dlmwrite('synthetic_svs1.csv',[SVs1,SVs1cls]);
        % dlmwrite('synthetic_svs2.csv',[SVs2,SVs2cls]);
        % dlmwrite('contour_svs1.csv',[X00,X11,predict1]);
        % dlmwrite('contour_svs2.csv',[X00,X11,predict2]);
        
        
        totalSVs = obtainNumberSVs(model,options);

        % predict
        % ytrue
        % totalSVs
        
        % ----------------------------------------------------------------------
      case 'sca_flip'
        model    = trainModels(trainIDX, wr, options);
        [predict, ytrue, idxAll]  = testModels(trainIDX, model, options);
        
        % save previous state
        dataclasses_tmp = dataclasses;
        
        ind = predict ~= ytrue;
        
        trainIDX_tmp = trainIDX;
        [trainIDX, toRemove] = changeDataset(ind,trainIDX,options);

        trainidxMisClassified = trainIDX(toRemove); 
        ind1 = dataclasses(trainidxMisClassified) == 1;
        
        dataclasses(trainidxMisClassified( ind1)) = 2;
        dataclasses(trainidxMisClassified(~ind1)) = 1;
        
        model    = trainModels(trainIDX, wr, options);
        starttime = tic;
        [predict, ytrue, ~]  = testModels(valIDX, model, options);
        testtime   = toc(starttime);

        % restore previous state
        dataclasses = dataclasses_tmp;
        
        %totalSVs = length(model.Alpha);
        totalSVs = obtainNumberSVs(model,options);
      % ----------------------------------------------------------------------
      case 'weights'
        
        if options.prune & options.reduce_now
            dataclasses_tmp    = dataclasses;
            
            options.reduce_now = false;

            % fprintf(1,'3.1 %d\n', length(trainIDX));

            weights_model            = trainModels(trainIDX, wr, options);
            [predict, ytrue, idxAll] = testModels(trainIDX, weights_model, options);
            irej                     = ~mod(predict,2); % items to be removed

            misclassified = predict ~= ytrue;

            fprintf(1,'\n\t num train. instances: %.1f\n',length(trainIDX));
            fprintf(1,'\t rejected: %.1f\n',sum(irej));
            fprintf(1,'\t misclassified: %.1f\n',sum(misclassified));

            irej = irej | misclassified;
            trainIDX_tmp = changeDataset(irej,trainIDX,options);

            dataclasses = (dataclasses-1)/2+1;
            options.nclasses = length(unique(dataclasses));
            options.reduce_now = true;

            options.method = 'standard'; % standard
            
            % fprintf(1,'3.2 %d\n', length(trainIDX_tmp));
            
            weights_model    = trainModels(trainIDX_tmp, wr, options);
            starttime = tic;
            [predict, ytrue] = testModels( valIDX, weights_model, options);
            testtime   = toc(starttime);

            % weights_model
            totalSVs = obtainNumberSVs(weights_model,options);
            
            best_options.nclasses = length(unique(dataclasses));
            dataclasses = dataclasses_tmp;
        
        else
            
            weights_model    = trainModels(trainIDX, wr, options);
            starttime = tic;
            [predict, ytrue] = testModels(valIDX, weights_model, options);
            testtime   = toc(starttime);


            totalSVs = [];
            for k = 1:length(weights_model)
                totalSVs = [totalSVs, obtainNumberSVs(weights_model{k},options)];
            end

        end
        
    end
    
    return

%%                                                                                                     
%
function [predict ytrue prob net idxAll] = runSpecificModelsII(trainIDX,valIDX,wr,options);
    net                              = runSpecificmodelsIIa(trainIDX,wr,options);
    [predict ytrue acc prob idxAll ] = runSpecificmodelsIIb(valIDX,net,options);

    return
    
    
%%                                                                                                     
%
function net = runSpecificmodelsIIa( trainIDX, wr, options )
    global datafeatures
    global dataclasses
    
    
    
    nElem = length(trainIDX);
    
    switch( options.method )

      case 'threshold'
        switch(options.submethod)
          case 'libsvm'
            commandtmp = genSVMstring(options,true);

            mySTARTTIME = tic;
            net   = svmtrain(dataclasses(trainIDX,end), ...
                             datafeatures(trainIDX,:),...
                             commandtmp);
            mySTARTTIME = tic;
            toc(mySTARTTIME);

          case 'svm'
            kkk
            quadprog_options = optimset('MaxIter',options.maxiter,'TolFun',options.epsilon,'TolX', options.epsilon,'Display','off');
            
            handle_kernel_fun = @(U,V) my_svm_kernelfunction(U, V, options);
            
            
            net = svmtrain( datafeatures(trainIDX,:) , dataclasses(trainIDX),'autoscale',false,...
                            'BoxConstraint', options.C, 'Kernel_Function', handle_kernel_fun,...
                            'QUADPROG_OPTS', quadprog_options ...
                            );

          otherwise
            kka
        end
      
      otherwise
        myerror(' **** runSpecificmodelsIIa method unknown **** ');
    end
    
    return
    
    
%%                                                                                                     
%
function [predict ytrue acc prob idxAll] = runSpecificmodelsIIb(valIDX,net,options)
    global datafeatures
    global dataclasses
    idxAll = [];
    
    ytrue = dataclasses(valIDX);
    
    acc = 0;
    nElem = length( valIDX );
    prob  = zeros(nElem,1);

    switch (options.method)
        
      case 'threshold'
        switch(options.submethod)
          case 'libsvm'
            commandsvm = '-b 1';
            
            [predict acc prob] = ...
                svmpredict( zeros(length(valIDX),1), datafeatures(valIDX,:), ...
                            net, commandsvm );
            
            
            
          case 'svm'
            % [predict acc prob] = svmpredict (dataclasses(valIDX), datafeatures(valIDX,:), net,'-b 1');
            % predict = svmclassify (net, datafeatures(valIDX,:));

            d = datafeatures(valIDX,:);
            
            K  = my_svm_kernelfunction(d, net.SupportVectors, ...
                                       options);
            P  = K.*repmat(net.Alpha, 1, length(valIDX))';
            fx = sum(P,2)+net.Bias;
            
            predict = logical(fx<0)*2+1;
            
            prob = 1./(1+exp(-fx));
            % prob = logsig(fx);

            % [predict, fx, prob, dataclasses(valIDX)]
            % pause
            % sum(fx>0)
            % pause

          otherwise
            kkka
        end
      
      otherwise
        myerror(' **** runSpecificmodelsIIb method unknown **** ');
        
    end

    

    return
    
%%
%
function roc_data = runSpecificModelsIII( trainIDX, testIDX, wr, best_options, roc_data )
    global dataclasses

    % oldGamma = best_options.gamma;
    % best_options.gamma = 2^-7;

    [predict, ytrue, prob, net] = runSpecificModelsII( trainIDX, trainIDX, wr, best_options);

    for i=1:length(wr)
        
        best_error     = inf;
        best_threshold = 0;

        for threshold = 0.50:0.01:1
            irej = calc_rej(threshold,prob);
            
            RR   = sum(irej)/length(trainIDX);
            idx  = ~irej;
            
            predictOutOfReject = predict(idx);
            
            ER = mean( ytrue(idx) ~= predictOutOfReject );
            
            % Error evaluation
            all_errors = wr(i)*RR+ER;

            if ( (all_errors < best_error) && (RR ~= 0) )
                best_error     = all_errors;
                best_threshold = threshold;
                % [i sum(irej) best_threshold all_errors length(trueClass) wr(i)*RR length(trainIDX)]
            end
        end
        best_options.best_threshold = best_threshold;

        if best_options.prune
            [predict1, ~, prob1, net, idxAll] = runSpecificModelsII( trainIDX, trainIDX, wr, best_options);
            misclassified = predict1 ~= dataclasses(trainIDX);

            irej = calc_rej(best_options.best_threshold,prob1);
            
            fprintf(1,'\n\t num train. instances: %.1f\n',length(trainIDX));
            fprintf(1,'\t rejected: %.1f\n',sum(irej));
            fprintf(1,'\t misclassified: %.1f\n',sum(misclassified));
            % remove rejected and misclassified instances
            irej = irej | misclassified;
            
            
            trainIDX_tmp = changeDataset(irej,trainIDX,best_options);
            
            % train model with the clean dataset
            best_options.method = 'standard';

            net = trainModels(trainIDX_tmp, wr, best_options);
            starttime = tic;
            pred = testModels( testIDX, net, best_options);
            testtime   = toc(starttime);

            best_options.method = 'threshold';

            if isempty(pred)
                perf = 0;
            else
                perf = mean( pred == dataclasses(testIDX) );
            end
            % [pred, classTest]
        end

        roc_data = [roc_data;  wr(i) perf obtainNumberSVs(net,best_options) best_threshold testtime];
    end
    
    return
    
function irej = calc_rej(threshold,prob)
    irej = max( prob, [], 2 ) < threshold;
    return

    
%%                                                                               
%
function net = trainModels( trainIDX, wr, options )
    global datafeatures
    global dataclasses
    
    switch( options.method )
      case {'standard','sca_flip','sca_del','ssca_del'}
        
        switch(options.submethod)
          case 'libsvm'
            commandtmp = genSVMstring(options);

            mySTARTTIME = tic;
            net   = svmtrain(dataclasses(trainIDX,end), ...
                             datafeatures(trainIDX,:),...
                             commandtmp);
            toc(mySTARTTIME);

          case 'svm'
            quadprog_options = optimset('MaxIter',options.maxiter,'TolFun',options.epsilon,'TolX', options.epsilon,'Display','off');
            
            handle_kernel_fun = @(U,V) my_svm_kernelfunction(U, V, options);
            net = svmtrain( datafeatures(trainIDX,:), dataclasses(trainIDX), 'autoscale', false, ...
                            'BoxConstraint', repmat(options.C,length(trainIDX),1), 'Kernel_Function', handle_kernel_fun,...
                            'options', quadprog_options );
          
          otherwise
            kka
        end
        
      case 'weights'

        switch(options.submethod)
          case 'libsvm'
            classes = unique(dataclasses(trainIDX));
            
            % oldGamma
            options.gamma = 2^-7;
            commandtmp = genSVMstring(options);

            k = 1;

            for i=1:2:options.nclasses
                
                if ~options.reduce_now % & options.prune
                    j = classes(classes ~= i);
                    
                    c1 = i;
                    c2 = j;
                    
                    w1 = wr*options.C;
                    w2 = (1-wr)*options.C;
                    
                    commandtmptmp = sprintf('%s -w%d %f -w%d %f',commandtmp,c1,w1,c2,w2);
                else
                    commandtmptmp = commandtmp;
                end
                
                % commandtmptmp
                % pause
                
                net{k}   = svmtrain( dataclasses(trainIDX), ...
                                     datafeatures(trainIDX,:), ...
                                     commandtmptmp );
                
                k = k + 1;
            end
            
            % net
            % options.nclasses
            % unique(dataclasses(trainIDX))
            
          case 'svm'
            quadprog_options = optimset('MaxIter',options.maxiter,'TolFun',options.epsilon,'TolX', options.epsilon,'Display','off');
            
            handle_kernel_fun = @(U,V) my_svm_kernelfunction(U, V, options);
            k = 1;
            for i=1:2:options.nclasses
                
                weights = options.C * ones(length(trainIDX),1);
                
                if ~options.reduce_now & options.prune
                    ind = dataclasses(trainIDX) == i;
                    weights( ind) = wr*weights(ind);
                    weights(~ind) = (1-wr)*weights(~ind);
                end
                
                net{k} = svmtrain( datafeatures(trainIDX,:) , dataclasses(trainIDX),'autoscale',false,...
                                   'BoxConstraint', weights, 'Kernel_Function', handle_kernel_fun,...
                                   'QUADPROG_OPTS', quadprog_options ...
                                   );
                k = k + 1;
            end
        end
      
      otherwise
        myerror('(trainmodels) method unknown');
    end

    
    return
    
%%                                                                               
%
function [predict, ytrue, idxAll, sign] = testModels(valIDX, net, options, points)
    global datafeatures
    global dataclasses
    
    ytrue  = [];
    idxAll = [];
    sign   = [];

    ytrue = dataclasses(valIDX);
    
    nElem    = length(valIDX);
    
    predictP = zeros( size(net,2), nElem );
    prob     = zeros( size(net,2), nElem );
    % init 
    
    switch( options.method )
      case 'weights'

        for i = 1:size(net,2)
            switch(options.submethod)
              case 'libsvm'
                
                predictP(i,:) = svmpredict(zeros(length(valIDX),1), datafeatures(valIDX,:), ...
                                           net{i}, '-b 0' )';
              case 'svm'
                K = my_svm_kernelfunction(datafeatures(valIDX,:), net{i}.SupportVectors,...
                                          options);
                P = K.*repmat(net{i}.Alpha, 1, length(valIDX))';
                sign = sum(P,2)+net{i}.Bias;
                
                predictP(i,:) = logical(sign<0)*2+1;
                
                %predictP(i,:) = svmclassify(net{i}, datafeatures(valIDX,:))';
            end
        end

        if size(net,2) > 1
            predict = testModelsAux(predictP);
        else
            predict = predictP';
        end

      case {'standard','sca_flip','sca_del','ssca_del'}
        if nargin < 4
            switch(options.submethod)
              case 'libsvm'
                [predict, ~, sign] = svmpredict(zeros(length(valIDX),1), datafeatures(valIDX,:), ...
                                     net, '-b 0' );

              case 'svm'
                d = datafeatures(valIDX,:);

                K = my_svm_kernelfunction(d, net.SupportVectors,...
                                          options);
                P = K.*repmat(net.Alpha, 1, length(valIDX))';
                sign = sum(P,2)+net.Bias;
                
                predict = logical(sign<0)+1;

                % mean(predict ~= dataclasses(valIDX))

                % predict = svmclassify (net, datafeatures(valIDX,:));
                
                % mean(predict ~= dataclasses(valIDX))
                % pause

              otherwise
                kk
            end
        else 
            % debug purposes..
            switch(options.submethod)
              case 'libsvm'
                predict = svmpredict(zeros(length(points),1),points,net,...
                                     '-b 0');
                % unique(predict)
                kkaka
                ;
              otherwise
                kkaka
                predict = svmclassify (net, points);
            end
        end

      otherwise 
        myerror ( '(testmodels) method unknown') 
    end
    
    return

%%                                                                         
%
function predict = testModelsAux(predictP)
    predict = (sum(predictP,1)/2)';
    
    return
    
%%                                                 
% 
function [merror, RR, ER]  = calcErrorReject(predict,ytrue,valIDX,wr,options)
%global datafeatures
%global dataclasses
    
    % performance assessment
    % reject rate
    irej    = ~mod(predict,2);
    
    nElem   = length(valIDX);
    classes = ytrue;
    
    RR   = sum(irej)/nElem;
    idx  = ~irej;
    
    % misclassification rate
    trueClass          = classes(idx);
    predictOutOfReject = predict(idx);
    error              = find( trueClass ~= predictOutOfReject );

    ER   = length(error) / nElem;
    
    merror = wr*RR+ER;
    return


%%                                                                                                              
%
function options = setParameters(i,combinations,options)
% fprintf(1,'C: %.4d | gamma: %.4d\n',combinations(1,i),combinations(2,i));
% special options
    switch ( options.method ) 
      
      case {'threshold','weights','standard','sca_flip','sca_del','ssca_del'}
        switch( options.kernel )
          case {'linear',0}
            options.C     = combinations(1,i);
            options.gamma = 1;
          otherwise
            options.C     = combinations(1,i);
            options.gamma = combinations(2,i);
        end
        
      otherwise
        myerror( '(setparameters) method unknown');
    end
    return
    