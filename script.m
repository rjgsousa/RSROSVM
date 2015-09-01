
%%  script
function script(dataset)
degree = 3;
gamma  = 1;
randomset = 0;

% givenval = 0; % se tem dataset dividido em treino, validacao e teste
switch( dataset)
  case 'shuttle'
    gamma = 9/2;
    givenval = true;
    nround = 1; %train, val e test given

  case 'pendigits'
    gamma    = 16/2; % #features/2
    givenval = true; % train e test given
    nround   = 1; % train e test given

  case 'usps'
    gamma = 256/2;
    givenval = true;
    nround   = 1; % train e test given
    
  case 'ijcnn'
    gamma = 22/2; % #features/2
    givenval = true;
    nround = 5; %train, val e test given
    randomset = 23000;

  case 'waveformversion1'
    jjj
    givenval = false;
    
  case 'syntheticI'
    nround   = 20;
    givenval = false;
    gamma    = 1;
    degree   = 2;
    
  otherwise
    unknown_problem
end

prune  = true;
method = 'standard';
main(method,prune,dataset,givenval,degree,nround,gamma,randomset)

method = 'sca_flip';
main(method,prune,dataset,givenval,degree,nround,gamma,randomset)

method = 'sca_del';
main(method,prune,dataset,givenval,degree,nround,gamma,randomset)

method = 'ssca_del';
main(method,prune,dataset,givenval,degree,nround,gamma,randomset,0.0)
main(method,prune,dataset,givenval,degree,nround,gamma,randomset,0.3)
main(method,prune,dataset,givenval,degree,nround,gamma,randomset,0.9)
main(method,prune,dataset,givenval,degree,nround,gamma,randomset,1.3)

method = 'weights';
main(method,prune,dataset,givenval,degree,nround,gamma,randomset)

method = 'threshold';
main(method,prune,dataset,givenval,degree,nround,gamma,randomset)


return