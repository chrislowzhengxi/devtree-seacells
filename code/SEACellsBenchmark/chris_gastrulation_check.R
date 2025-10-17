

# # squeue -u xyang2
# # squeue -p bigmem --state=PD | wc -l
# # sinteractive -p bigmem  --mem=500G  --account=pi-xyang2 --time=2:00:00      

# library(ExperimentHub)
# library(MouseGastrulationData)

# ## for Christ, when running on compute nodes, set downloadRun = FALSE
# downloadRun = FALSE # TRUE

# # 1) Choose cache location
# localrun = FALSE # TRUE
# if (localrun) {
#   custom_cache <- ExperimentHub::getExperimentHubOption("CACHE") # F:/BioC_cache/ExperimentHub
# # [1] "C:\\Users\\Holly\\AppData\\Local/R/cache/R/ExperimentHub"
#   custom_cache <- normalizePath(custom_cache, winslash = "/", mustWork = FALSE) 
# } else {
#   custom_cache <- file.path("/scratch", "midway3", "xyang2", "BioC_cache", "ExperimentHub")  # Midway3
# }
   
# # 2) Point ExperimentHub to your cache & confirm
# dir.create(custom_cache, recursive = TRUE, showWarnings = FALSE)	
# ExperimentHub::setExperimentHubOption("CACHE", custom_cache)
# # refer to F:\projects\scRNA\data\Pijuan-Sala2019_gastrulation\download_2025.R
sce = readRDS("/project/imoskowitz/xyang2/SHH/MouseGastrulationData/mouse_gastrulation_sce.rds")
sce
# class: SingleCellExperiment 
# dim: 29452 139331 
# metadata(0):
# assays(1): counts
# rownames(29452): ENSMUSG00000051951 ENSMUSG00000089699 ...
  # ENSMUSG00000096730 ENSMUSG00000095742
# rowData names(2): ENSEMBL SYMBOL
# colnames(139331): cell_1 cell_2 ... cell_139330 cell_139331
# colData names(17): cell barcode ... colour sizeFactor
# reducedDimNames(2): pca.corrected umap
# mainExpName: NULL
# altExpNames(0):

meta = colData(sce)  # obs.
# rm(sce)
head(meta)
# DataFrame with 6 rows and 17 columns
              # cell        barcode    sample      pool       stage sequencing.batch
       # <character>    <character> <integer> <integer> <character>        <integer>
# cell_1      cell_1 AAAGGCCTCCACAA         1         1        E6.5                1
# cell_2      cell_2 AACAAACTCGCCTT         1         1        E6.5                1
# cell_3      cell_3 AACAATACCCGTAA         1         1        E6.5                1
# cell_4      cell_4 AACACTCTCATTCT         1         1        E6.5                1
# cell_5      cell_5 AACAGAGAATCAGC         1         1        E6.5                1
# cell_6      cell_6 AACATATGAATCGC         1         1        E6.5                1
           # theiler doub.density   doublet   cluster cluster.sub cluster.stage
       # <character>    <numeric> <logical> <integer>   <integer>     <integer>
# cell_1         TS9    0.0431142     FALSE         2           4             2
# cell_2         TS9    1.1297133     FALSE        12           1             1
# cell_3         TS9    0.0000000     FALSE        NA          NA            NA
# cell_4         TS9    0.0846885     FALSE        NA          NA            NA
# cell_5         TS9    0.1121205     FALSE         3           7             4
# cell_6         TS9    1.2547088     FALSE         1           1             3
       # cluster.theiler  stripped         celltype      colour sizeFactor
             # <integer> <logical>      <character> <character>  <numeric>
# cell_1               3     FALSE         Epiblast      635547   0.567139
# cell_2               1     FALSE Primitive Streak      DABE99   1.178520
# cell_3              NA      TRUE               NA          NA   0.605571
# cell_4              NA      TRUE               NA          NA   0.860882
# cell_5               6     FALSE     ExE ectoderm      989898   0.891500
# cell_6               7     FALSE         Epiblast      635547   1.226646
unique(meta$celltype)

# "/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/other/edges.txt
edges = read.table("/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/other/edges.txt", header=T, as.is=T, sep="\t")
dim(edges)  #[1] 507   6
head(edges)
           # system      x      y x_name          y_name	edge_type
# 1 Pre_gastrulation PGa_M1 PGa_M2 Oocyte          1-cell	Developmental progression
# 2 Pre_gastrulation PGa_M2 PGa_M3 1-cell          2-cell	Developmental progression
# 3 Pre_gastrulation PGa_M3 PGa_M4 2-cell          4-cell	Developmental progression
# 4 Pre_gastrulation PGa_M4 PGa_M5 4-cell          8-cell	Developmental progression
# 5 Pre_gastrulation PGa_M5 PGa_M6 8-cell          Morula	Developmental progression
# 6 Pre_gastrulation PGa_M6 PGa_M7 Morula Inner cell mass	Developmental progression

table(edges$system)
                 # Blood      Brain_spinal_cord            Endothelium 
                    # 33                    150                     14 
      # Epithelial_cells                    Eye           Gastrulation 
                    # 32                     21                     54 
    # Gastrulation_E8.5b                    Gut Lateral_plate_mesoderm 
                    # 49                     21                     44 
              # Mesoderm              Notochord               PNS_glia 
                    # 32                      1                      8 
           # PNS_neurons       Pre_gastrulation                  Renal 
                     # 9                     22                     17 
edges = subset(edges, system=='Gastrulation')
dim(edges)  # [1] 54  6

unique_nodes = unique(c(edges$x_name, edges$y_name))
length(unique_nodes)  #[1] 34

# setdiff(unique(meta$celltype), unique_nodes)
 # # [1] "Primitive Streak" => "Primitive streak"              NA                              
 # # [3] "ExE ectoderm"                   "Visceral endoderm"             
 # # [5] "ExE endoderm"                   "Rostral neurectoderm"          
 # # [7] "Blood progenitors 2"            "Mixed mesoderm"                
 # # [9] "ExE mesoderm"    =>  "Extraembryonic mesoderm"              "Pharyngeal mesoderm"           
# # [11] "Caudal epiblast"                "PGC"                           
# # [13] "Mesenchyme"                     "Haematoendothelial progenitors" =>"Hematoendothelial progenitors"
# # [15] "Blood progenitors 1"            "Paraxial mesoderm"             
# # [17] "Caudal neurectoderm"            "Somitic mesoderm"              
# # [19] "Caudal Mesoderm"                "Erythroid1"                    
# # [21] "Def. endoderm"  => "Definitive endoderm"                "Parietal endoderm"             
# # [23] "Anterior Primitive Streak"      "Forebrain/Midbrain/Hindbrain"  
# # [25] "Cardiomyocytes"                 "Erythroid2"                    
# # [27] "NMP"                            "Erythroid3"    
# setdiff(unique_nodes, unique(meta$celltype) )	
 # # [1] "Anterior primitive streak"        "Blood progenitors"               
 # # [3] "Lateral plate mesoderm"           "Paraxial mesoderm (Tbx6+)"       
 # # [5] "Extraembryonic visceral endoderm" "Extraembryonic mesoderm"  !!       
 # # [7] "Embryonic visceral endoderm"      "Definitive ectoderm"             
 # # [9] "Cardiopharyngeal mesoderm"        "CLE and NMPs"                    
# # [11] "Hindbrain"                        "Primitive streak"     !!           
# # [13] "Hematoendothelial progenitors" !!   "Definitive endoderm"  !!           
# # [15] "Amniotic mesoderm"                "Forebrain"                       
# # [17] "Primitive erythroid cells"        "Paraxial mesoderm (Tbx6-)"       
# # [19] "Midbrain"                         "Amniotic ectoderm"               
# # [21] "Gut mesenchyme"                   "Cardiogenic mesoderm"            
# # [23] "Primordial germ cells"            "Floor plate"          			 


nodes = read.table("/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/other/nodes.txt", header=T, as.is=T, sep="\t")
table(nodes$system)
                 # Blood      Brain_spinal_cord            Endothelium 
                    # 27                     56                     12 
      # Epithelial_cells                    Eye           Gastrulation 
                    # 26                     18                     36 
                   # Gut Lateral_plate_mesoderm               Mesoderm 
                    # 16                     28                     18 
             # Notochord               PNS_glia            PNS_neurons 
                     # 2                      6                      8 
      # Pre_gastrulation                  Renal 
                    # 16                     14 

nodes=subset(nodes, system=='Gastrulation')
setdiff(nodes$celltype_new, unique_nodes)
# [1] "Extraembryonic ectoderm" "Parietal endoderm"      

# /project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/other/pd_processed_PS_JaxE85.rds
pd = readRDS('/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/other/pd_processed_PS_JaxE85.rds')
dim(pd)  # [1] 262454     19
head(pd)
       # orig.ident nCount_RNA nFeature_RNA  day       group
# cell_1       cell       8962         2546 E6.5 Pijuan_Sala
# cell_2       cell      19573         3725 E6.5 Pijuan_Sala
# cell_5       cell      14190         3478 E6.5 Pijuan_Sala
# cell_6       cell      20343         3863 E6.5 Pijuan_Sala
# cell_8       cell      22546         4042 E6.5 Pijuan_Sala
# cell_9       cell      24506         4353 E6.5 Pijuan_Sala
                                        # cell_state
# cell_1                               E6.5:Epiblast
# cell_2 E6.5:Primitive streak and adjacent ectoderm
# cell_5                E6.5:Extraembryonic ectoderm
# cell_6                               E6.5:Epiblast
# cell_8                               E6.5:Epiblast
# cell_9                               E6.5:Epiblast
                                    # cell_type cell_id   UMAP_1    UMAP_2  UMAP_3d_1
# cell_1                               Epiblast  cell_1 5.523445  7.860553  1.6702673
# cell_2 Primitive streak and adjacent ectoderm  cell_2 2.816050  7.279688 -0.4638996
# cell_5                Extraembryonic ectoderm  cell_5 1.952989 15.887635 -3.8292493
# cell_6                               Epiblast  cell_6 5.727568  8.465960  1.7454512
# cell_8                               Epiblast  cell_8 5.401665  8.192011  1.5643262
# cell_9                               Epiblast  cell_9 5.601872  7.774686  1.7861273
       # UMAP_3d_2 UMAP_3d_3 Louvain_res_1 Louvain_res_2 Louvain_res_5     pre_celltype
# cell_1  8.798370 -1.890507     cluster_0     cluster_0     cluster_0         Epiblast
# cell_2  6.965925 -2.076103     cluster_0     cluster_5     cluster_1 Primitive Streak
# cell_5 11.217245  2.784134    cluster_20    cluster_13    cluster_24     ExE ectoderm
# cell_6  9.296762 -1.503458     cluster_0     cluster_0     cluster_0         Epiblast
# cell_8  9.209273 -1.887638     cluster_0     cluster_0     cluster_0         Epiblast
# cell_9  8.504643 -1.671524     cluster_0     cluster_0     cluster_0         Epiblast
               # celltype_update Anno
# cell_1                Epiblast <NA>
# cell_2        Primitive streak <NA>
# cell_5 Extraembryonic ectoderm <NA>
# cell_6                Epiblast <NA>
# cell_8                Epiblast <NA>
# cell_9                Epiblast <NA>

##############################
# Filter the cell metadata by the group column for "Pijuan_Sala," using the "celltype_update" column as the nodes.
table(pd$group)
        # Jax Pijuan_Sala 
     # 153597      108857 
pd = subset(pd, group=='Pijuan_Sala')
setdiff(pd$celltype_update, unique_nodes)
# [1] "Extraembryonic ectoderm" "Parietal endoderm"	 
setdiff( unique_nodes,  pd$celltype_update)  # character(0)

setdiff(pd$cell_id, meta$cell) # character(0)
length(setdiff(meta$cell, pd$cell_id)) # 30474  cells were filtered by Qiu
