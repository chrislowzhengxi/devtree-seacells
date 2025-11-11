library(igraph)
library(dplyr)
library(ggraph)
library(tidygraph)
library(RColorBrewer)
library(grid)

#####################################
### Section - 6, Development tree ###
#####################################

###################################
### Summary the edges and nodes ###
###################################


# loacal setting 
source("D:/projects/SHH/source/Qiu_TimeLapse/JAX_help_code_xy_localsetting.R")
source("D:/projects/SHH/source/Qiu_TimeLapse/JAX_code/JAX_color_code.R")
QiuFile_path #'D:/projects/SHH/Qiu_TimeLapse/other/'
tome_path = 'D:/projects/SHH/Qiu_TimeLapse/tome/mm/'
work_path = "D:/projects/SHH/result/Qiu_TimeLapse/tree/"


setwd(work_path)

#############################################################################
### comapre HOlly's 262 nodes vs 262 nodes in Fig 5g (copyed from the pdf) ##
#############################################################################

# refer to S6_development_tree_s3_Creat_graph.R
nodes_Holly =  read.table(file='nodes_filtered.txt', sep='\t', header=T)
dim(nodes_Holly) # [1] 262   6
head(nodes_Holly, 3)
  # celltype_new new_id           system meta_group celltype_num id
# 1       Oocyte      1 Pre_gastrulation     PGa_M1            2  1
# 2       1-cell      2 Pre_gastrulation     PGa_M2            3  2
# 3       2-cell      3 Pre_gastrulation     PGa_M3            3  3
nodes_Qiu = read.table('D:/projects/SHH/data/Qiu_TimeLapse/Fig5g_legend.txt', sep='\t', header=F)
dim(nodes_Qiu)  #[1] 262   2
head(nodes_Qiu, 3)
  # V1     V2
# 1  1 Oocyte
# 2  2 1-cell
# 3  3 2-cell

setdiff(nodes_Qiu$V2, nodes_Holly$celltype_new)
 # [1] "ExE ectoderm"                          
 # [2] "ExE visceral endoderm"                 
 # [3] "Blood prog."                           
 # [4] "ExE mesoderm"                          
 # [5] "Haematoendothelial prog."              
 # [6] "Paraxial mesoderm (Tbx6–)"             
 # [7] "Venous/capillary endothelial cells"    
 # [8] "Pituitary/pineal gland prog."          
 # [9] "Naive retinal prog. cells"             
# [10] "Retinal prog. cells"                   
# [11] "Amacrine/horizontal precursor cells"   
# [12] "Eye \037eld"                           
# [13] "Lung prog. cells"                      
# [14] "Midgut/hindgut epithelial cells"       
# [15] "Alveolar type 1 cells"                 
# [16] "Alveolar type 2 cells"                 
# [17] "Renal cortical stromal cells"          
# [18] "Nephron prog."                         
# [19] "Gastrointestinal SMCs"                 
# [20] "Gonad prog. cells"                     
# [21] "Limb mesenchyme prog."                 
# [22] "Airway SMCs"                           
# [23] "Renal medullary stromal cells"         
# [24] "Vascular SMCs"                         
# [25] "Vascular SMCs (Pparg+)"                
# [26] "Haematopoietic stem cells (Cd34+)"     
# [27] "Haematopoietic stem cells (Mpo+)"      
# [28] "Megakaryocyte-erythroid prog."         
# [29] "Monocytic MDSCs"                       
# [30] "PMN MDSCs"                             
# [31] "T-NK prog."                            
# [32] "B cell prog."                          
# [33] "Definitive early erythroblasts (Cd36–)"
# [34] "Definitive erythroblasts (Cd36+)"      
# [35] "GABAergic neurons (>E13.0)"            
# [36] "Glutamatergic neurons (>E13.0)"        
# [37] "Intermediate neuronal prog."           
# [38] "Neural prog. cells (Neurod1+)"         
# [39] "Neural prog. cells (Ror1+)"            
# [40] "NMPs and spinal cord prog."            
# [41] "Oligodendrocyte prog. cells"           
# [42] "Spinal cord dorsal prog. (>E13.0)"     
# [43] "Spinal cord ventral prog. (>E13.0)"    
# [44] "Cajal–Retzius cells"                   
# [45] "Mesodermal prog. (Tbx6+)"              
# [46] "Muscle prog. cells"                    
# [47] "Muscle prog. cells (Prdm1+)"           
# [48] "Adipocyte prog. cells"    

setdiff(nodes_Holly$celltype_new, nodes_Qiu$V2)
 # [1] "Extraembryonic ectoderm"                      
 # [2] "Extraembryonic visceral endoderm"             
 # [3] "Blood progenitors"                            
 # [4] "Extraembryonic mesoderm"                      
 # [5] "Hematoendothelial progenitors"                
 # [6] "Paraxial mesoderm (Tbx6-)"                    
 # [7] "Venous and capillary endothelial cells"       
 # [8] "Pituitary/Pineal gland progenitors"           
 # [9] "Naive retinal progenitor cells"               
# [10] "Retinal progenitor cells"                     
# [11] "Amacrine/Horizontal precursor cells"          
# [12] "Eye field"                                    
# [13] "Lung progenitor cells"                        
# [14] "Midgut/Hindgut epithelial cells"              
# [15] "Alveolar Type 1 cells"                        
# [16] "Alveolar Type 2 cells"                        
# [17] "Renal pericytes and mesangial cells"       #!!!!!!   
# [18] "Nephron progenitors"                          
# [19] "Gastrointestinal smooth muscle cells"         
# [20] "Gonad progenitor cells"                       
# [21] "Limb mesenchyme progenitors"                  
# [22] "Airway smooth muscle cells"                   
# [23] "Renal stromal cells"                          
# [24] "Vascular smooth muscle cells"                 
# [25] "Vascular smooth muscle cells (Pparg+)"        
# [26] "Hematopoietic stem cells (Cd34+)"             
# [27] "Hematopoietic stem cells (Mpo+)"              
# [28] "Megakaryocyte-erythroid progenitors"          
# [29] "Monocytic myeloid-derived suppressor cells"   
# [30] "PMN myeloid-derived suppressor cells"         
# [31] "T-NK progenitors"                             
# [32] "B cell progenitors"                           
# [33] "Definitive early erythroblasts (CD36-)"       
# [34] "Definitive erythroblasts (CD36+)"             
# [35] "GABAergic neurons (after E13.0)"              
# [36] "Glutamatergic neurons (after E13.0)"          
# [37] "Intermediate neuronal progenitors"            
# [38] "Neural progenitor cells (Neurod1+)"           
# [39] "Neural progenitor cells (Ror1+)"              
# [40] "NMPs and spinal cord progenitors"             
# [41] "Oligodendrocyte progenitor cells"             
# [42] "Spinal cord dorsal progenitors (after E13.0)" 
# [43] "Spinal cord ventral progenitors (after E13.0)"
# [44] "Cajal-Retzius cells"                          
# [45] "Mesodermal progenitors (Tbx6+)"               
# [46] "Muscle progenitor cells"                      
# [47] "Muscle progenitor cells (Prdm1+)"             
# [48] "Adipocyte progenitor cells"                   


library(dplyr)
library(stringdist) # for fuzzy matching

# Data frames for illustration purposes

# Function to standardize terms for better matching
standardize_terms <- function(term) {
  term <- tolower(term) # make all lowercase
  term <- gsub("prog\\.", "progenitor", term) # replace prog. with progenitor
  term <- gsub("haemato", "hemato", term) # unify UK/US spelling
  term <- gsub("ex[e|a]", "extraembryonic", term) # replace "ExE" with "extraembryonic"
  term <- gsub("[^a-z0-9 ]", "", term) # remove special characters
  term <- gsub("([^s])$", "\\1s", term) # add 's' to end if not present to standardize plural
  term <- trimws(term) # remove leading/trailing whitespace
  return(term)
}

# Apply standardization
nodes_Qiu$V2_standard <- sapply(nodes_Qiu$V2, standardize_terms)
nodes_Holly$celltype_new_standard <- sapply(nodes_Holly$celltype_new, standardize_terms)

# Fuzzy matching with string distance
matches <- data.frame(Qiu = character(), Holly = character(), Distance = numeric())

for (qiu_term in nodes_Qiu$V2_standard) {
  holly_term <- nodes_Holly$celltype_new_standard[
    which.min(stringdist::stringdist(qiu_term, nodes_Holly$celltype_new_standard, method = "jw"))
  ]
  distance <- min(stringdist::stringdist(qiu_term, nodes_Holly$celltype_new_standard, method = "jw"))
  matches <- rbind(matches, data.frame(Qiu = qiu_term, Holly = holly_term, Distance = distance))
}

# Merge back original terms for clarity
matches <- matches %>%
  left_join(nodes_Qiu, by = c("Qiu" = "V2_standard")) %>%
  left_join(nodes_Holly, by = c("Holly" = "celltype_new_standard"))

# Display the matches and distances
head(matches,3)
     # Qiu  Holly Distance V1     V2 celltype_new new_id           system
# 1 oocyte oocyte        0  1 Oocyte       Oocyte      1 Pre_gastrulation
# 2  1cell  1cell        0  2 1-cell       1-cell      2 Pre_gastrulation
# 3  2cell  2cell        0  3 2-cell       2-cell      3 Pre_gastrulation
  # meta_group celltype_num id
# 1     PGa_M1            2  1
# 2     PGa_M2            3  2
# 3     PGa_M3            3  3

tmp = matches[which(matches$Qiu != matches$Holly),]

write.csv(matches, file='node_mathes_all.csv')
write.csv(tmp, file='node_mismathes.csv')

## ensure there are no redundant nodes in Holly's version
grep('Airway', nodes_Holly$celltype_new, value=T)
# [1] "Airway club cells"          "Airway goblet cells"       
# [3] "Airway smooth muscle cells"


grep('Renal', nodes_Qiu$V2, value=T)
#[1] "Renal cortical stromal cells"  "Renal medullary stromal cells"
grep('Renal', nodes_Holly$celltype_new, value=T)
# [1] "Renal pericytes and mesangial cells" "Renal stromal cells"   

grep('Monocytes', nodes_Qiu$V2, value=T)
# 'Monocytes'
grep('derived suppressor cells', nodes_Holly$celltype_new, value=T)  
# [1] "Monocytic myeloid-derived suppressor cells"
# [2] "PMN myeloid-derived suppressor cells"    