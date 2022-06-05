# Act132_Cesspool_Prioritization
 Codebase for the DoH sponsored cesspool prioritization update by SeaGrant and WRRC, 2021


Cesspools are a substandard sewage disposal method and widely recognized to harm human health and the environment. The state of Hawai‘i has an estimated 82,000 cesspools. To address pollution concerns, the Hawai‘i State Legislature mandated replacement of all cesspools by 2050. A major step in achieving this goal is to categorize cesspools based on potential or realized harm to humans and the environment. This report details a comprehensive tool designed for this purpose. After researching similar efforts, methods and datasets were chosen that met the needs of state government, cultural values, and environmental sensitivities. The Hawai‘i Cesspool Prioritization Tool (HCPT) was developed by integrating fifteen risk-factors that either control or relate to how cesspool impacts are distributed across communities and the environment. These factors were processed with a geospatial model to calculate a single prioritization score for every cesspool in Hawai‘i. Because sewage pollution impacts are cumulative, individual scores were consolidated by census boundary areas. Results from the HCPT prioritization were validated through comparison with a statewide assessment of nearshore wastewater impacts funded by Hawai‘i Act 132. Future data, organized within census area frameworks, can be layered onto the results to address equity and outreach challenges. 

The HCPT was designed to be as objective as possible with prioritization based solely on the relationships between datasets, thereby reducing human bias as much as possible. All data used in the HCPT is at the statewide scale, normalized, and based on regulatory rules or modeling outputs. The total number of cesspools in the state categorized as Priority Level 1 was 13,885, with 13,482 and 54,058 as Priority Level 2 and Priority Level 3, respectively. Approximately 35%, 7%, 21%, and 37% of cesspools in the Priority Level 1 group are located on O‘ahu, Maui, Kaua‘i, and Hawai‘i Island respectively. 


##### Figure Draft Results
<p align="center">
  <img width="450" height="450" src=Report_maps/Results_maps_numeric/FINAL_Priority_cats.jpg >
</p>



##### Figure: Plot showing how the threshold decay function converts data from each risk factor (in this case Distance to Coastline in meters) to a 0-100 score. Note the inset showing how the priority score equals 100 for all units within 50 ft (15.24m) from the coast (the state’s regulatory threshold), and how the score decays with greater distances from the coast. 

<p align="center">
  <img width="450" height="255" src=Report_maps/thresholddecay.jpg >
</p>
