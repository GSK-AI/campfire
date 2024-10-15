import argparse
import pickle
import yaml

import numpy as np
import pandas as pd

import numpy.random as rn 
import copy


def set_split(row,held_out_compounds):
    if row["Metadata_JCP2022"] in held_out_compounds:
        return 'test_out_all_'
    else:
        return 'test_out_plate_'


def get_held_out_compounds(compound_list,control_list,N_held_out,):

    compound_list_wout_controls = compound_list[~np.in1d(compound_list,control_list)]
    N_not_control = len(compound_list_wout_controls)

    idxs = np.arange(0,N_not_control,1)

    held_out_idx = rn.choice(idxs,size=N_held_out,replace=False)

    held_out_compounds = compound_list_wout_controls[held_out_idx]

    assert len(held_out_compounds) == N_held_out

    return held_out_compounds

def split_target_plates(wells,control_list,N_held_out,hold_out_target_plates):

    Ncompounds = np.unique(wells["Metadata_JCP2022"].values).shape[0]
    assert Ncompounds == 302, "Unexpected number of compounds in TARGET2 plates"

    target_compound_ids = wells["Metadata_JCP2022"].values
    compound_list = np.unique(target_compound_ids)

    held_out_compounds = get_held_out_compounds(compound_list,control_list,N_held_out)

    rn.seed(10)

    wells['split'] = np.nan
    # For all compounds in the set of plates
    for compound in compound_list:
        #Get all wells with compound in them 
        wells_with_compound = wells.loc[wells["Metadata_JCP2022"]==compound]

        #of those wells, gather those in held out plates
        held_out_wells = wells_with_compound.loc[wells_with_compound['Metadata_Plate'].isin(hold_out_target_plates)]
        #and asisgn new column to determine their split via set_split (either test_out_all or test_out_plate)
        held_out_wells['split'] = held_out_wells.apply(set_split,axis=1)

        #Find wells _NOT_ in held out plates
        not_held_out_wells =  wells_with_compound.loc[~wells_with_compound['Metadata_Plate'].isin(hold_out_target_plates)]

        #If compound is in the list of compounds we hold out, assign to separate split
        if compound in held_out_compounds:
            not_held_out_wells['split'] = 'test_out_compound'

        #Otherwise, of the N plates that compound is present in, assign 70/10/20 to train/val/test

        #To keep our dataset balanced with respect to compound, we need to specify to only use 14 plates for training 
        #and we will randomly sample one well with this compound for a given plate. 
        #This means we don't pick up ~1000 training wells for DMSO 
        else:

            #get list of plates which aren't held out
            list_of_plates = np.unique(not_held_out_wells['Metadata_Plate'].values)
            Nplates = len(list_of_plates)

            #Number of plates we assign to each split
            Ntrain = 14
            Nval = 2
            Ntest = Nplates - Ntrain - Nval

            #Get indices correpsonding to plates in train/val/test
            idxs = np.arange(0,Nplates,1)

            train_idx = rn.choice(idxs,size=Ntrain,replace=False)

            idxs = idxs[~np.in1d(idxs,train_idx)]

            assert idxs.shape[0] == Nplates - Ntrain

            val_idx = rn.choice(idxs,size=Nval,replace=False)

            idxs = idxs[~np.in1d(idxs,val_idx)]

            test_idx = idxs 

            assert test_idx.shape[0] == Ntest

            #with indices, construct list of train/val/test plate strings
            train_well_plates = list_of_plates[train_idx]
            val_well_plates = list_of_plates[val_idx]
            test_well_plates = list_of_plates[test_idx]

            #Now for each plate, we sample 1 well at random (for many compounds, there is 1 well per plate anyway)
            not_held_out_wells['split'] = np.nan
            for plate in list_of_plates:

                random_well = not_held_out_wells.loc[not_held_out_wells['Metadata_Plate']==plate].sample(n=1)

                index = random_well.index[0]

                #find which split plate belongs to for this compound, and assign string
                if plate in train_well_plates:
                    split_str = 'train'
                elif plate in val_well_plates:
                    split_str = 'val'
                else:
                    split_str = 'test'

                not_held_out_wells.loc[index,'split'] = split_str

        #update wells dataframe 
        wells['split'].update(not_held_out_wells['split'])
        wells['split'].update(held_out_wells['split'])
    
    wells['split'].fillna('NaN',inplace=True)

    return wells,held_out_compounds


def split_compound_plates(wells,held_out_compounds,frac_compound_wells_in_train):

    Ncompounds = np.unique(wells["Metadata_JCP2022"].values).shape[0]
    assert Ncompounds == 58457, "Unexpected number of compounds in Compound plates"

    compound_plate_compound_ids = wells["Metadata_JCP2022"].values
    compound_list = np.unique(compound_plate_compound_ids)

    rn.seed(10)

    wells['split'] = np.nan
    # For all compounds in the set of plates
    for compound in compound_list:
        #Get all wells with compound in them 
        wells_with_compound = wells.loc[wells["Metadata_JCP2022"]==compound]

        #If compound is in the list of compounds we hold out, assign to separate split
        if compound in held_out_compounds:
            wells_with_compound['split'] = 'test_out_compound'

        #Otherwise, assign wells in plate at random to train/test splits
        else:

            #get list of plates which aren't held out
            list_of_plates = np.unique(wells_with_compound['Metadata_Plate'].values)
            Nplates = len(list_of_plates)

            #Number of plates we assign to each split
            Ntrain = int(Nplates*frac_compound_wells_in_train)

            #Now for each plate, we assign each well to train with given probability, otherwise leave out of train/test/val 
            wells_with_compound['split'] = np.nan
            for plate in list_of_plates:

                wells_with_compound.loc[wells_with_compound['Metadata_Plate']==plate,'split'] = rn.choice(['train',np.nan],p=[frac_compound_wells_in_train,1-frac_compound_wells_in_train],size=(wells_with_compound['Metadata_Plate']==plate).sum())


        #update wells dataframe 
        wells['split'].update(wells_with_compound['split'])
    
    wells['split'].fillna('NaN',inplace=True)

    return wells



def main(config) -> None:
    """
    Main Function:
    """
    seed = config["seed"]

    rn.seed(seed)


    # Plates: information on plate names, source of origin, and batch number 
    # Wells: information on plate, plate location, and compound key 
    # Compounds: map from compound key to inchikey etc 
    plates = pd.read_csv(config['plate_dir'])
    wells = pd.read_csv(config['well_dir'])
    compounds = pd.read_csv(config['compound_dir'])

    num_compounds_held_out = config['num_compounds_held_out']

    #We currently have data for source 3, so in this notebook we create controls for source 3 plates only 
    source_plates = plates.loc[plates["Metadata_Source"]=="source_3"]


    #splitting is applied differently to the quality control TARGET2 plates, and compound plates, so separate here 
    target_plates = source_plates.loc[source_plates['Metadata_PlateType']=='TARGET2']["Metadata_Plate"].values
    compound_plates = source_plates.loc[source_plates['Metadata_PlateType']=='COMPOUND']["Metadata_Plate"].values

    hold_out_target_plates = config['hold_out_target_plates']


    target_wells = wells.loc[wells["Metadata_Plate"].isin(target_plates)]

    control_list = config['control_list']
    frac_compound_wells_in_train = config['frac_compound_wells_in_train']

    target_wells, held_out_compounds = split_target_plates(target_wells,control_list,num_compounds_held_out,hold_out_target_plates)
    compound_plate_wells = split_compound_plates(wells,held_out_compounds,frac_compound_wells_in_train)

    return compound_list






if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, default="controls_config.yaml", help="Config file"
    )

    args = parser.parse_args()

    yaml_file = open(f"{args.config}")
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    main(config)


