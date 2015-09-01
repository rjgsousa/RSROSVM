
%% data loading
%
function [trainSetFeatures,trainSetClass,IDS,unique_IDS,unique_cls] = loadDataSets( options, dataset )

    trainSetFeatures = [];
    trainSetClass    = [];
    IDS              = [];
    unique_IDS       = [];
    unique_cls       = [];
    
    switch(dataset)
        
      case 'syntheticI'
        [trainSetFeatures,trainSetClass] = syntheticI();
    
      otherwise
        [trainSetFeatures, trainSetClass, IDS,unique_IDS, unique_cls] = loadRealDataset(options.project_lib_path,dataset);
    end

    return
    

% ------------------------------------------------------------------------
%% pol
function [trainSetFeatures,trainSetClass] = syntheticI()
    data = genSyntheticI(10000);
    
    trainSetFeatures = data(:,1:end-1);
    trainSetClass    = data(:,end);
	
    return;

