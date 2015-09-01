

%% Loads real dataset
%
function [data,classes,IDS,unique_IDS,unique_cls] = loadRealDataset(project_lib_path,datasetname)
    IDS = [];
    unique_IDS = [];
    unique_cls = [];
    
    subproblem = [];

    datasetnameOrig = datasetname;
    datasetname = lower( datasetname );

    underscore  = strfind(datasetname,'_');
    if ~isempty ( underscore )
        subproblem  = upper( datasetname(underscore+1:end) );
        datasetname = datasetname(1:underscore-1);
    end
    
    datasetname
    subproblem

    switch ( datasetname )
      case 'ijcnn'
        data    = cell(3,1);
        classes = cell(3,1);
        [classes{1}, data{1}] = libsvmread('../libraries/ijcnn/ijcnn1.tr');
        [classes{2}, data{2}] = libsvmread('../libraries/ijcnn/ijcnn1.val');
        [classes{3}, data{3}] = libsvmread('../libraries/ijcnn/ijcnn1.t');

        % 1-2
        classes{1} = (classes{1}+1)/2+1;
        classes{2} = (classes{2}+1)/2+1;
        classes{3} = (classes{3}+1)/2+1;

        data = normalizeData( data ) ;
        
      case 'shuttle'
        data    = cell(3,1);
        classes = cell(3,1);
        [classes{1}, data{1}] = libsvmread('../libraries/shuttle/shuttle.scale.tr');
        [classes{2}, data{2}] = libsvmread('../libraries/shuttle/shuttle.scale.val');
        [classes{3}, data{3}] = libsvmread('../libraries/shuttle/shuttle.scale.t');
        
        classes{1} = parse_classes( classes{1} );
        classes{2} = parse_classes( classes{2} );
        classes{3} = parse_classes( classes{3} );
        
      case 'usps'
        data    = cell(3,1);
        classes = cell(3,1);

        [classes{1}, data{1}] = libsvmread('../libraries/usps/usps');
        [classes{3}, data{3}] = libsvmread('../libraries/usps/usps.t');
        % cls=1 is the symbol 0
        
        cls = classes{1}-1;
        ind = ( cls == 1 | cls == 2 | ...
                cls == 4 | cls == 5 | ...
                cls == 7 );
        cls( ind) = 1;
        cls(~ind) = 2;
        
        classes{1} = cls;
        
        cls = classes{3}-1;
        ind = ( cls == 1 | cls == 2 | ...
                cls == 4 | cls == 5 | ...
                cls == 7 );
        cls( ind) = 1;
        cls(~ind) = 2;
        
        classes{3} = cls;
        
      case 'pendigits'
        data    = cell(3,1);
        classes = cell(3,1);
        d = dlmread('../libraries/pendigits/pendigits.tra');
        cls = d(:,end);
        d(:,end) = [];
        data{1}    = d;
        ind = ( cls == 1 | cls == 2 | ...
                cls == 4 | cls == 5 | ...
                cls == 7 );
        cls( ind) = 1;
        cls(~ind) = 2;
        classes{1} = cls;

        d = dlmread('../libraries/pendigits/pendigits.tes');
        cls = d(:,end);
        d(:,end) = [];

        data{3}    = d;
        ind = ( cls == 1 | cls == 2 | ...
                cls == 4 | cls == 5 | ...
                cls == 7 );
        cls( ind) = 1;
        cls(~ind) = 2;
        classes{3} = cls;
      
        data = normalizeData( data );
        
      otherwise
        myerror('(load real datasets) unknown dataset');
    end

    return

function cls = parse_classes( cls )
    cls(cls>1) = 2;
    return


function [data] = normalizeData( data )
    minv = min(data{1},[],1);
    maxv = max(data{1},[],1);
    
    for n=1:size(data,1)
        if isempty(data{n})
            continue
        end
        dtmp = data{n};
        nelem = size(dtmp,1);
        
        dtmp = (dtmp - repmat(minv,nelem,1))./ ...
               (repmat(maxv,nelem,1)-repmat(minv,nelem,1));
    
        dtmp = dtmp*2-1;
        
        data{n} = dtmp;
    end
    
    return    
    
