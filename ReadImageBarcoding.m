clear all
warning off

load('data/INSECTS/data.mat')
load('data/INSECTS/splits.mat')

% `data.mat`
% * `embeddings_dna`: Vector Embeddings for DNA barcode data
% * `embeddings_img`: Vector Embeddings for IMAGE data
% * `labels`: Numeric labels for species
% * `species`: Species names
% * `G`: Genus labels of species
% * `nucleotides`: DNA barcode of the species
% * `bold_ids`: IDs of the sampels from BOLD system. You may use this ids to see the full details of the spekciemen in BOLD system.
% * `ids`: Image names of the samples in our dataset.
%
%
% `splits.mat`
% * `train_loc`: Indices of training data points for tuning
% * `trainval_loc`: Indices of training data points for final inference
% * `test_seen_loc`: Indices of test data from seen classes
% * `test_unseen_loc`: Indices of test data from unseen classes, classe non
% nel training set
% * `val_seen_loc`: Indices of validation data from seen classes
% * `val_unseen_loc`: Indices of validation data from unseen classes


for fold = 1
    test = [test_seen_loc test_unseen_loc];
    train = trainval_loc;
    idTR{fold}=train;
    idTE{fold}=test;
    labelTR{fold}=labels(train);
    labelTE{fold}=labels(test);

    %Genus
    labelTRgenus{fold}=G(labels(train))-1040;
    labelTEgenus{fold}=G(labels(test))-1040;

end

NumDescribed=length(test_seen_loc);%number of pattern from seen classes 

data=[embeddings_dna embeddings_img];%features