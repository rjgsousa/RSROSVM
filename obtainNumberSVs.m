
% -------------
function totalSVs = obtainNumberSVs(model,options)
    switch (options.submethod)
      case 'libsvm'
        totalSVs = model.totalSV;
      otherwise
        totalSVs = length(model.Alpha);
    end
    return