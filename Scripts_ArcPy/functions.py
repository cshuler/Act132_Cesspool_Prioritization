import arcpy
arcpy.env.overwriteOutput = True 
from arcpy.sa import *
from arcpy import env

import matplotlib.pyplot as plt
import seaborn as sns

import os
import pandas as pd
import numpy as np
import scipy
from scipy import stats

from random import seed
from random import gauss
from random import randint
seed(5)

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_rows', 50)

tempspace = os.path.join(".", "tempspace")
intermidiate_DataFramesPath = os.path.join(".", 'Outputs/intermidiate_DataFrames')


def Format_OSDS_shp(fromshp, cleanitup=True):
    osds_temp   = os.path.join(tempspace, "OSDS_temp.shp")
    arcpy.CopyFeatures_management(fromshp, osds_temp)   # first make a copy

    # Create UID
    arcpy.management.AddField(osds_temp, "Uid", "TEXT")
    arcpy.CalculateField_management(osds_temp, "Uid", 
                                    'str(!X!)[0:10]+"_"+str(!Y!)[0:8]', "PYTHON3")
    # now convert to pandas to do stuff
    arcpy.TableToTable_conversion(osds_temp, tempspace, 'OSDS_all.csv') # Create a rational file format
    # Create a rational PanDataframe
    OSDS_All = pd.read_csv(os.path.join(tempspace, 'OSDS_all.csv'))
    
    if cleanitup:
    #Select only selected columns
        osds_cols = ['X', 'Y', 'Island', 'TMK', 'Uid']
        OSDS = OSDS_All[osds_cols].copy()
    else: 
        OSDS = OSDS_All
    
    return OSDS


def convert_OSDS_csv_to_shp(csvFilePath):
    # Convert CSV to shapefile of the OSDS points 
    arcpy.management.Delete(os.path.join(tempspace, "OSDS_cleaned.shp"))  # In case the shapefile already exists Arc will not overwright, have to delete
    spatialRef = arcpy.SpatialReference(4326)
    #csvFilePath = os.path.join(intermidiate_DataFramesPath, "OSDS.csv")
    shpFilePath = os.path.join(tempspace)
    arcpy.MakeXYEventLayer_management(csvFilePath, 'X', 'Y', 'OSDS_cleaned', spatial_reference=spatialRef)
    arcpy.FeatureClassToShapefile_conversion('OSDS_cleaned', shpFilePath)




def Calc_dist_to_variable(osds_path, shp_path, out_col):
    # receives the OSDS file and a line or point variable shp and cals the distance of each OSDS to a feature in the variable.shp
    
    OSDS = pd.read_csv(os.path.join(intermidiate_DataFramesPath, "OSDS.csv"), index_col=0)

    # Conduct the near analysis on the shp and the OSDS shapefile
    Stupid_tablePath = os.path.join(tempspace, 'stupidtable') # temporary container for stupid arc tables 
    arcpy.GenerateNearTable_analysis(osds_path, shp_path, Stupid_tablePath, method='GEODESIC')   # run the near analysis 
    arcpy.TableToTable_conversion(Stupid_tablePath, tempspace, 'near.csv')                       # Create a rational file format

    # Create a rational PanDataframe
    near_Coast_df = pd.read_csv(os.path.join(tempspace, 'near.csv'))
    # merge it on to the OSDS frame 
    tempframe = OSDS.join(near_Coast_df, how='outer').copy()
    tempframe.rename(columns={'NEAR_DIST': out_col}, inplace=True)

    # Create the frame
    Return_frame = tempframe[['Uid', out_col]].copy()

    return Return_frame


def Hist_and_stats_on_DF(DF, col): 
    # THis plots a histogram and gives basic stats for a numeric dataframe column
    
    DF[col].hist(bins=50)
    print("Average {} is {}".format(col, DF[col].mean()))
    print("Max {} is {}".format(col, DF[col].max()))
    print("Min {} is {}".format(col, DF[col].min()))
    
    
    
def Find_points_inside_polys(In_points, In_polygons, new_col_name):
    # Create a dataframe with those Uid's that geographically fall within a polygon area, and assign them a 1
    
    # First, make a layer from the feature class
    arcpy.MakeFeatureLayer_management(In_polygons, "caprock_lyr")
    arcpy.MakeFeatureLayer_management(In_points, "osds_lyr")
    # select the points inside the polygon by using WITHIN  
    arcpy.SelectLayerByLocation_management("osds_lyr", "WITHIN", "caprock_lyr")
    # copy the selected features to an output feature class
    arcpy.CopyFeatures_management("osds_lyr", os.path.join(tempspace, "test_caprock_OSDS.shp"))
    # read on OSDS data from shapefile 
    yes_caprock_points_path = os.path.join(tempspace, "test_caprock_OSDS.shp")
    arcpy.TableToTable_conversion(yes_caprock_points_path, tempspace, 'yes_caprock_all.csv') # Create a rational file format
    # Create a rational PanDataframe
    Caprock_All = pd.read_csv(os.path.join(tempspace, 'yes_caprock_all.csv'))
    Caprock_All[new_col_name] = True
    
    #Select only selected columns
    want_cols = ['Uid', new_col_name]
    Return_frame = Caprock_All[want_cols]
    
    return Return_frame


def print_stats_on_bool_layers(df):
    # print some stats on the bool layers
    
    OSDS = pd.read_csv(os.path.join(intermidiate_DataFramesPath, "OSDS.csv"), index_col=0)
    print('number of affected points = {}'.format(len(df)))
    print('number of total systems = {}'.format(len(OSDS)))
    print('percent of systems affected = {}'.format(len(df)/len(OSDS)))
    
    
    
def extract_values_from_rasters(In_raster, In_points, new_col_name, tempspace):
    # extract the values for a raster to intersecting point features
    
    arcpy.MakeFeatureLayer_management(In_points, "osds_lyr")
    ExtractValuesToPoints(In_points, In_raster, os.path.join(tempspace, "test_temp.shp"))

    # read on OSDS data from shapefile 
    extracted_points_path = os.path.join(tempspace, "test_temp.shp")
    arcpy.TableToTable_conversion(extracted_points_path, tempspace, 'extracted_all.csv') # Create a rational file format
    # Create a rational PanDataframe
    Extracted_All = pd.read_csv(os.path.join(tempspace, 'extracted_all.csv'))
    Extracted_All.dropna(axis = 0, how = 'all', inplace = True)

    #Select only selected columns
    want_cols = ['Uid', 'RASTERVALU']
    Extract_frame = Extracted_All[want_cols]
    Extract_frame.rename(columns={'RASTERVALU':new_col_name}, inplace=True)     # Rename to col that you want
    Extract_frame.loc[Extract_frame[new_col_name] < -100] = np.nan                 # Deal with the -9999 values set to nan
    Extract_frame = Extract_frame.dropna(axis = 0, how = 'all')                 # Drop any rows with no rainfall data
    
    return Extract_frame


def deal_with_no_FlikrCellsOSDS_pts(in_features, Flikr_cells):             

    arcpy.MakeFeatureLayer_management (in_features, "ESRI_is_lame")

    #Create stupid separate layer for just the outliers
    query = "Flikr_ID_s = ''"    # Where Flikr ID is null
    arcpy.SelectLayerByAttribute_management('ESRI_is_lame', "NEW_SELECTION", query)
    arcpy.CopyFeatures_management('ESRI_is_lame', os.path.join(tempspace, "ESRI_is_idiotic.shp"))

    # Do the spatial join to Flikr cells
    target_features = os.path.join(tempspace, "ESRI_is_idiotic.shp")
    join_features = Flikr_cells
    out_features =    os.path.join(tempspace, 'OutlierStarts_wFlikr_cells2.shp')
    arcpy.SpatialJoin_analysis(target_features, join_features, out_features, match_option="CLOSEST")
    
    ## Clean up the fields in the outlier shp to only include needed ones
    shp = os.path.join(tempspace, 'OutlierStarts_wFlikr_cells2.shp')
    fcList = [field.name for field in arcpy.ListFields(shp)]   # list fields
    fcList.remove('State_ID'); fcList.remove('Uid'); fcList.remove('FID'); fcList.remove('Shape') #pop off keepers  (Flokr_ID_1 is autogenerated when the 2nd merge is done)
    for field in fcList:
            arcpy.DeleteField_management(shp, fcList)  # delete extranious fields
            
    # read the paths data from shapefile into a pandas dataframe
    paths_path = os.path.join(tempspace, 'OutlierStarts_wFlikr_cells2.shp')
    columns_nams = [field.name for field in arcpy.ListFields(paths_path)]
    columns_nams.pop(1)  # remove stupid shape col
    temparr = arcpy.da.FeatureClassToNumPyArray(paths_path, columns_nams)
    Outlier_starts_DF =  pd.DataFrame(temparr)
    
    # read the paths data from shapefile into a pandas dataframe
    columns_nams = [field.name for field in arcpy.ListFields(in_features)]
    columns_nams.pop(1)  # remove stupid shape col
    temparr = arcpy.da.FeatureClassToNumPyArray(in_features, columns_nams)
    OSDS_flkrEndTmp_DF =  pd.DataFrame(temparr)
    ### This is to rectify the statewide fliker ID
    #OSDS_flkrEndTmp_DF = OSDS_flkrEndTmp_DF.rename(columns={ 'Flikr_ID_s':'State_ID'}) 

    # Do the merge addin on the outliers to the goodframe 
    OSDS_flkrEndTmp_DF_merge = OSDS_flkrEndTmp_DF.merge(Outlier_starts_DF, on='Uid', how='left')
    
    # Merge the flikr IDs with the replacements for the nanns
    OSDS_flkrEndTmp_DF_merge['Flikr_IDmerged2'] = np.where(OSDS_flkrEndTmp_DF_merge['Flikr_ID_s'] == " ", OSDS_flkrEndTmp_DF_merge['State_ID'], OSDS_flkrEndTmp_DF_merge['Flikr_ID_s'])
    del OSDS_flkrEndTmp_DF_merge['Flikr_ID']
    
    # rename columns 
    OSDS_flkrEndTmp_DF_merge = OSDS_flkrEndTmp_DF_merge.rename(columns={ 'Flikr_IDmerged2':'Flikr_ID'})   # 'Cess_ID':"Uid",? rename the cespool ID for some reason
    #Cut out extranious columns
    carelist = ["Uid", "Flikr_ID", "Flikr_X", "Flikr_Y"]
    OSDS_flkrEndTmp_DF_merge = OSDS_flkrEndTmp_DF_merge[carelist]
    
    Outframe = OSDS_flkrEndTmp_DF_merge.copy()
    
    return Outframe


def Census_data_joining(In_points, In_polygons, want_cols, Save_meta=False, Level=False):
    # Join on the census data (I think the census data shapefile must have a column that is in want cols
    
    # Do the spatial join
    arcpy.SpatialJoin_analysis(In_points, In_polygons, os.path.join(tempspace, "test_join_pt_2_poly.shp"))

    # read on OSDS data from shapefile 
    extracted_points_path = os.path.join(tempspace, "test_join_pt_2_poly.shp")
    arcpy.TableToTable_conversion(extracted_points_path, tempspace, 'extracted_all.csv') # Create a rational file format
    # Create a rational PanDataframe
    Extracted_All = pd.read_csv(os.path.join(tempspace, 'extracted_all.csv'))

    Extract_frame = Extracted_All[want_cols]

    if Save_meta: 
        # Create a block ID shapefile for playing with metadata on pivots
        extracted_blocks_path = os.path.join("..", "Projected_data/Census", '2010_Census_{}_Meta.shp'.format(Level))
        arcpy.TableToTable_conversion(extracted_blocks_path, tempspace, 'extracted_all.csv') # Create a rational file format
        # Create a rational PanDataframe
        Extracted_Blocks = pd.read_csv(os.path.join(tempspace, 'extracted_all.csv'))
        Extracted_Blocks.to_csv(os.path.join("Outputs/Census_aggregared_SHPs", 'Census_{}_metadata.csv'.format(Level)))
        
    return Extract_frame


def crossplotEm(x, y, row, col, num, xlabel, ylabel, title, c='k'):

    plt.subplot(row,col,num) 
    plt.scatter(x, y, alpha=0.7, c=c)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)

    # plop on a regression line
    mask = ~np.isnan(x) & ~np.isnan(y) # regression does not like NaN values, this line masks them out and regression is only performed on finite numbers
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[mask], y[mask])   # calculate regression stats for the ecoli and rainfall data
    rmse = np.sqrt(np.mean(np.abs(x-y)**2))
    r2 = r_value**2
    rX = np.linspace(min(x), max(x), 50)
    rY = slope*rX + intercept
    plt.plot(rX,rY, color='g', linestyle='--', alpha = .6, label = ("r$^2$ = " + "$%.2f$"%r2 +",\n Slope="+"$%.2f$"%slope +"\n RMSE="+"$%.2f$"%rmse) )
    plt.legend(loc=2) 
    plt.tight_layout()


# Functions to group based on tracks and blocks and block groups

def group_by_census_unit(MASTER_df, ID_column, omit_less_than=10):
    # first get a count of OSDS units in each unit    
    countPiv = pd.pivot_table(MASTER_df, index=ID_column, aggfunc = 'count')
    countPiv.reset_index(inplace=True)
    
    # now average ranks over the tracks
    Unit_Priority_Master = pd.pivot_table(MASTER_df, index=ID_column, aggfunc = 'mean')
    Unit_Priority_Master.reset_index(inplace=True)
    Unit_Priority_Master['OSDS_count'] = countPiv['Uid']   # Craete an OSDS count column (any column will do)
    # Unit_Priority_Master['Total_butt_units_count'] = Unit_Priority_Master['OSDS_count']*Unit_Priority_Master['PepPerHos']   # Craete an butt_units count column, meaning total number of people pooping in the block into OSDS
    
    # Pull out tracks with few OSDS or no population or are not in a track
    Unit_Priority_Master = Unit_Priority_Master[Unit_Priority_Master['OSDS_count'] > omit_less_than].copy()
    Unit_Priority_Master = Unit_Priority_Master[Unit_Priority_Master['Track_ID'] != 0].copy()
    # Unit_Priority_Master = Unit_Priority_Master[Unit_Priority_Master['Total_butt_units_count'] > 0]   # make sure that there are OSDS with population in the block

    # Create final scores and ranks, scaled by # of people on each cesspool
    Unit_Priority_Master['Final_Prioity_Score'] = Unit_Priority_Master['Weighted_Priority_mean']    
    Unit_Priority_Master['Final_Prioity_Rank']  = Unit_Priority_Master["Final_Prioity_Score"].rank(ascending=False)
    
    # Calculate the quantiles for the top 10% middle 40% and bottom 50%
    Score_for_HIGH = Unit_Priority_Master['Final_Prioity_Score'].quantile([0.5,0.75]).values[1]
    Score_for_MED = Unit_Priority_Master['Final_Prioity_Score'].quantile([0.5,0.75]).values[0]

    # Note if the census track/block has more than %50 if its OSDS units inside a 2017 priority zone
    Unit_Priority_Master["50pct_In_2017_CP_zone"] = Unit_Priority_Master['In_2017_CP_zone'].apply(lambda x: True if x>0.5 else False)

    # Calculate the catergorical Ranking Level, 
    Unit_Priority_Master["Final_Cat_Ranking"]=np.nan
    Unit_Priority_Master["Final_Cat_Ranking"] = np.where( (Unit_Priority_Master['Final_Prioity_Score'] >= Score_for_HIGH), "High", Unit_Priority_Master["Final_Cat_Ranking"])
    Unit_Priority_Master["Final_Cat_Ranking"] = np.where( (np.logical_and(Unit_Priority_Master['Final_Prioity_Score'] <= Score_for_HIGH, Unit_Priority_Master['Final_Prioity_Score'] >= Score_for_MED)), "Medium", Unit_Priority_Master["Final_Cat_Ranking"])
    Unit_Priority_Master["Final_Cat_Ranking"] = np.where( (Unit_Priority_Master['Final_Prioity_Score'] < Score_for_MED), "Low", Unit_Priority_Master["Final_Cat_Ranking"])
    
    Unit_Priority_Master["Fin_Rank"]=np.nan
    Unit_Priority_Master["Fin_Rank"] = np.where((Unit_Priority_Master['Final_Cat_Ranking'] == "High"), 1, Unit_Priority_Master["Fin_Rank"])
    Unit_Priority_Master["Fin_Rank"] = np.where((Unit_Priority_Master['Final_Cat_Ranking'] == "Medium"), 2, Unit_Priority_Master["Fin_Rank"])
    Unit_Priority_Master["Fin_Rank"] = np.where((Unit_Priority_Master['Final_Cat_Ranking'] == "Low"), 3, Unit_Priority_Master["Fin_Rank"])
    
# No Mo plus ranks
    # Add on a plus for those tracks that are inside of a 2017 zone 
#    Unit_Priority_Master["Final_Cat_Ranking"] = np.where( ((Unit_Priority_Master['Final_Prioity_Score'] >= Score_for_HIGH) & (Unit_Priority_Master['50pct_In_2017_CP_zone'] == True)), "High_+", Unit_Priority_Master["Final_Cat_Ranking"])
#    Unit_Priority_Master["Final_Cat_Ranking"] = np.where( ((Unit_Priority_Master['Final_Prioity_Score'] <= Score_for_MED) & (Unit_Priority_Master['50pct_In_2017_CP_zone'] == True)), "Low_+", Unit_Priority_Master["Final_Cat_Ranking"])
#    Unit_Priority_Master["Final_Cat_Ranking"] = np.where( ((np.logical_and(Unit_Priority_Master['Final_Prioity_Score'] <= Score_for_HIGH, Unit_Priority_Master['Final_Prioity_Score'] >= Score_for_MED)) & (Unit_Priority_Master['50pct_In_2017_CP_zone'] == True)), "Medium_+", Unit_Priority_Master["Final_Cat_Ranking"])
    
    

    
    return Unit_Priority_Master
    
    
def make_census_unit_SHP(Pivot_df, want_cols, OutShpPath, in_polygon_blocktrack):
    
    tmpShpPath = os.path.join(".", "tempspace", "TempShapeJunk.shp")
    
    # copy pristine dataset to something that can be joined on
    arcpy.CopyFeatures_management(in_polygon_blocktrack, tmpShpPath)
    
    joinfield =want_cols[0]
    arcsucksCSV = "temp_{}.csv".format(randint(1,100))  # This is needed because Arc will not update if a csv has been used before
    
    cleanframe = Pivot_df  #  [want_cols]    # I think this is redundant from the notebook, whatever....
    # Convert a pandas dataframe to an idiotic arc table view format (lame!)
    cleanframe.to_csv(os.path.join(".", "tempspace", arcsucksCSV))
    arcpy.TableToTable_conversion(os.path.join(".", "tempspace", arcsucksCSV), os.path.join(".", "tempspace"), "esrisucks")
    arcpy.management.MakeTableView(os.path.join(".", "tempspace", "esrisucks.dbf"), "esriislame")
    # Do the table joining
    arcpy.JoinField_management(tmpShpPath, joinfield, "esriislame", joinfield)
    
    # Remove the units with no priority score
    arcpy.MakeFeatureLayer_management(tmpShpPath, 'esriistheworst.lyr') 
    arcpy.SelectLayerByAttribute_management('esriistheworst.lyr', 'NEW_SELECTION', ' "Final_prio" > 1 ' )
    arcpy.CopyFeatures_management('esriistheworst.lyr', OutShpPath) 
    
    
    
# Get track names for census track plot
def get_track_names(path, unitlevel):
                    
    columns_nams = [field.name for field in arcpy.ListFields(path)]     # List of all col names
    columns_nams.pop(1)  # remove stupid shape col                           # THe "Shape" col will make numpy array to pandas puke
    temparr = arcpy.da.FeatureClassToNumPyArray(path, columns_nams)     # convert to numpy recarray
    Track_meta = pd.DataFrame(temparr)        
    Track_meta['Name_ID'] = Track_meta['Island']+": "+Track_meta['name']+": T#"+Track_meta[unitlevel].map(str)
    Track_meta = Track_meta[['Name_ID', unitlevel]] 
    
    return Track_meta

# Get track names for census track plot
def get_blkGrp_names(path, unitlevel):
                    
    columns_nams = [field.name for field in arcpy.ListFields(path)]     # List of all col names
    columns_nams.pop(1)  # remove stupid shape col                           # THe "Shape" col will make numpy array to pandas puke
    temparr = arcpy.da.FeatureClassToNumPyArray(path, columns_nams)     # convert to numpy recarray
    Track_meta = pd.DataFrame(temparr)        
    Track_meta['Name_ID'] = Track_meta['Island']+": "+Track_meta['name']+": BkGp#"+Track_meta[unitlevel].map(str)
    Track_meta = Track_meta[['Name_ID', unitlevel]] 
    
    return Track_meta


    
# For creating the soil suitibility rank based on its three thresholds. 
def read_3_thresholds(thresholds, idx_col):
    Ty = thresholds.loc[idx_col]['Type']
    T1 = thresholds.loc[idx_col]['T1']
    T2 = thresholds.loc[idx_col]['T2']
    T3 = thresholds.loc[idx_col]['T3']
    
    return Ty, T1, T2, T3

def cut_by_three_numeric_thresholds(OSDS_df, col_name, t_type, T1, T2, T3):
    # reminder P4 is good or more suitable for septics , 1 is bad or less suitable for septics 
    
    return_frame = OSDS_df[["Uid", col_name]].copy()
    Rcol = col_name
    
    if t_type == "ascending":
        return_frame.loc[return_frame[col_name] < T1,           '{}_Rank'.format(Rcol)] = "P1"
        return_frame.loc[return_frame[col_name].between(T1,T2), '{}_Rank'.format(Rcol)] = "P2"
        return_frame.loc[return_frame[col_name].between(T2,T3), '{}_Rank'.format(Rcol)] = "P3"
        return_frame.loc[return_frame[col_name] > T3,           '{}_Rank'.format(Rcol)] = "P4"

        return_frame = return_frame[["Uid", '{}_Rank'.format(Rcol)]]
        
    if t_type == "descending":
        return_frame.loc[return_frame[col_name] > T1,           '{}_Rank'.format(Rcol)] = "P1"
        return_frame.loc[return_frame[col_name].between(T2,T1), '{}_Rank'.format(Rcol)] = "P2"
        return_frame.loc[return_frame[col_name].between(T3,T2), '{}_Rank'.format(Rcol)] = "P3"
        return_frame.loc[return_frame[col_name] < T3,           '{}_Rank'.format(Rcol)] = "P4"

        return_frame = return_frame[["Uid", '{}_Rank'.format(Rcol)]]
    
    return return_frame



def print_costal_endpoint_flik_ID_analytics(OSDS_FLIK_ID, OSDS):
    # This is a one use function to clean up code in the notebook. 
    
    # Print analysics on how OSDS file from top cell, matches up with results of Bob model merge
    print("There are {} points with Flikr_IDs".format(len(OSDS_FLIK_ID)))
    print("There are {} OSDS shp points".format(len(OSDS)))
    A = OSDS['Uid'].unique();  B = OSDS_FLIK_ID['Uid'].unique()
    InAnotB = A[np.isin(A,B,invert=True)]; InBnotA = B[np.isin(B,A,invert=True)]
    print("There are {} OSDS units in OSDS, that do not have a flikr cell".format(len(InAnotB)))
    print("There are {} units with a Flikr cell, but are not in the OSDS file".format(len(InBnotA)))
    # Note there are 1651 OSDS on Molokai that are not considered 
    
    
# a 0 to 100 scaler function
def MY_minmaxscaler_dfCol(col, reverse=False):   
    # Reverse True indicates that bigger data values   (e.g dist to coast) get smaller ranks meaning less impact 
    # Reverse False indicates that smaller data values (e.g OSDS_density)  get smaller ranks meaning less impact  
    
    mx = col.max()
    mn = col.min()   
    if reverse==False:
        scaled_series = 100*(col-mn)/(mx-mn)       
    if reverse==True:
        scaled_series = 100*(col-mx)/(mn-mx)
    
    return scaled_series




# Plot a table figure heatmap for individual islands
def Plot_island_comparison_heatmap(Tracks_priority_frame_base, WantCols, Track_meta, Isla, Width, Height, Title): 

    Oahu_compare_priority_score_Track = Tracks_priority_frame_base[WantCols]
    Oahu_compare_priority_score_Track = Oahu_compare_priority_score_Track.sort_values("Final_Prioity_Score", ascending=False)
    Oahu_compare_priority_score_Track = Oahu_compare_priority_score_Track.merge(Track_meta, on='Track_ID', how='left')
    # Pull out th eisland to separate column
    Oahu_compare_priority_score_Track['island'] = Oahu_compare_priority_score_Track['Name_ID'].apply(lambda x: x.split(":")[0])
    # Cut to only oahu columns 
    Oahu_compare_priority_score_Track = Oahu_compare_priority_score_Track[Oahu_compare_priority_score_Track['island'] == Isla]
    # Add on the # of CPs to the index 
    Oahu_compare_priority_score_Track['Name_ID'] = Oahu_compare_priority_score_Track['Name_ID']+": CPs="+Oahu_compare_priority_score_Track['OSDS_count'].astype(str)
    Oahu_compare_priority_score_Track = Oahu_compare_priority_score_Track.set_index('Name_ID')                                  # Convert to pandas bliss
    del Oahu_compare_priority_score_Track['Track_ID'];  del Oahu_compare_priority_score_Track['OSDS_count']; del Oahu_compare_priority_score_Track['island']

    # Plot figure 
    fig, ax = plt.subplots(figsize=(Width, Height))
    plt.title(Title, fontsize=18, y=1.2)

    plt.tick_params(axis='x', which='major', labelsize=10, labelbottom = True, bottom=True, top = True, labeltop=True)
    plt.xticks(rotation=70)
    bar_label = "Priority score. Note total # of census units considered is {}".format(len(Tracks_priority_frame_base['OSDS_count']))
    g= sns.heatmap(Oahu_compare_priority_score_Track, annot=True,  cmap = 'YlOrBr', fmt=".1f", cbar_kws={'label': bar_label, "shrink": 0.5})
    plt.savefig(os.path.join(".", "Outputs/Figures", "Tracks_Master_scores_chart_{}.pdf".format(Isla)), bbox_inches='tight')
    plt.savefig(os.path.join(".", "Outputs/Figures", "Tracks_Master_scores_chart_{}.png".format(Isla)), bbox_inches='tight')
    return Oahu_compare_priority_score_Track



def Make_table_Track(Tracks_priority_frame_base, Isla):

    table_Track = Tracks_priority_frame_base
    table_Track = table_Track.sort_values("Final_Prioity_Score", ascending=False)
    trackpath = os.path.join("..", "Projected_data/Census/With_2017_priority", '2010_Census_Tracts_Meta_w2017.shp') 
    Track_meta = get_track_names(trackpath, "Track_ID")  # in the functions file
    table_Track = table_Track.merge(Track_meta, on='Track_ID', how='left')
    table_Track["Island"] = table_Track['Name_ID'].apply(lambda x: x.split(":")[0])
    table_Track["TrackName"] = table_Track['Name_ID'].apply(lambda x: x.split(":")[1])

    cutcols = ['Track_ID', 'TrackName', 'OSDS_count', "Fin_Rank"]  # 'Final_Prioity_Score', 'Final_Cat_Ranking'
    table_Track = table_Track[table_Track['Island']  == Isla][cutcols]

    return table_Track

def print_info_for_islands(Isla, table_track, Total_census_tracts):
    
    Num_tracts = len(table_track)
    Num_pools = table_track['OSDS_count'].sum()

    tot_high = len(table_track[table_track['Fin_Rank'] == 1])
    pct_high = int(round(tot_high/Num_tracts, 2)*100)
    cess_hi = table_track[table_track['Fin_Rank'] == 1]['OSDS_count'].sum()

    tot_med = len(table_track[table_track['Fin_Rank'] == 2])
    pct_med = int(round(tot_med/Num_tracts, 2)*100)
    cess_med = table_track[table_track['Fin_Rank'] == 2]['OSDS_count'].sum()

    tot_low = len(table_track[table_track['Fin_Rank'] == 3])
    pct_low = int(round(tot_low/Num_tracts, 2)*100)
    cess_low = table_track[table_track['Fin_Rank'] == 3]['OSDS_count'].sum()

    words = [Isla, Num_pools, Total_census_tracts, Num_tracts, pct_high, tot_high, pct_med, tot_med, pct_low, tot_low, Isla, cess_hi, cess_med, cess_low]

    text = "The island of {} contains {} cesspools and has a total of {} census tracts although only {} tracts contained more than 25 cesspools and were thus categorized with the HCPT. Of these tracts {}% or {} of them were categorized as high (1) priority (classified as the top 25% of census tracts statewide), {}% or {} of them were categorized as medium (2) priority and {}% or {} of them were categorized as low (3) priority. The total number of cesspools on {} catergorized as high (1) priority was {}, with {} and {} as medium (2) and low (3) priority, respectivly.".format(*words)

    print(text)
    