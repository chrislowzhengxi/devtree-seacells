# ssh xyang2@midway3.rcc.uchicago.edu
# module load python/anaconda-2022.05
# source activate /project/xyang2/software-packages/env/velocity_2022.05_xy

# rcchelp balance  
# rcchelp usage
# squeue -u xyang2
# squeue -p bigmem --state=PD | wc -l
# squeue -p caslake --state=PD | wc -l

# sinteractive -p bigmem  --mem=500G  --account=pi-xyang2 --time=8:00:00 # -c 4  (used)
# sinteractive -p caslake  --cpus-per-task=1 --mem=180G --time=6:00:00  --account=pi-xyang2
# sinteractive -p gpu --account=pi-xyang2 --gres=gpu:1 --mem=180GB  --time=6:00:00 -c 4   
# cd /project/imoskowitz/xyang2/heart_dev/Atlas2_results
# R
## To quit your interactive job:
# exit or Ctrl-D
# scp -p -r D:\projects\SHH\source\Qiu_TimeLapse\S6_development_tree_s1.2_Early_stage_graph.R  xyang2@midway3.rcc.uchicago.edu:/project/xyang2/SHH/source_midway3/.
# scp -p -r xyang2@midway3.rcc.uchicago.edu:/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/results/* D:\projects\SHH\result\Qiu_TimeLapse\.
# scp -p -r D:\projects\SHH\source\Qiu_TimeLapse\JAX*.R xyang2@midway3.rcc.uchicago.edu:/project/xyang2/SHH/source_midway3/.

library(igraph)
library(dplyr)
library(ggraph)
library(tidygraph)
library(RColorBrewer)
library(grid)
library(scales)   # For adding color scales
library(viridis)

if (!requireNamespace("gridExtra", quietly = TRUE)) install.packages("gridExtra")
library(gridExtra)   # for grid.arrange()
library(grid)

#####################################
### Section - 6, Development tree ###
#####################################

###################################
### Summary the edges and nodes ###
###################################

# # midway3 setting 
# source("/project/xyang2/SHH/source_midway3/JAX_help_code_xy.R")      # CHRIS: Doesn't exist 
# source("/project/xyang2/SHH/source_midway3/JAX_color_code.R")
# QiuFile_path #'/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/other/'
# tome_path = '/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/tome/mm/'
# work_path = "/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/results/tree/"
 

# # loacal setting 
# source("D:/projects/SHH/source/Qiu_TimeLapse/JAX_help_code_xy_localsetting.R")
# source("D:/projects/SHH/source/Qiu_TimeLapse/JAX_code/JAX_color_code.R")
# QiuFile_path #'D:/projects/SHH/Qiu_TimeLapse/other/'
# tome_path = 'D:/projects/SHH/Qiu_TimeLapse/tome/mm/'
# work_path = "D:/projects/SHH/result/Qiu_TimeLapse/tree/"


# # # Chris 
# USER_ROOT <- "/project/imoskowitz/xyang2/chrislowzhengxi"
# # Chris settings
# # QiuFile_path = '/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/'
# QiuFile_path = "/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/other/"
# work_path = "/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/results/tree/"
# setwd(work_path)


# # ==== CHRIS PATHS: inputs vs outputs ====
# USER_ROOT <- "/project/imoskowitz/xyang2/chrislowzhengxi"

# # Read-only inputs from Qiu/Holly
# QiuFile_path <- "/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/other/"

# # Your output folder
# work_path <- file.path(USER_ROOT, "results/ucell/tree")
# dir.create(work_path, recursive = TRUE, showWarnings = FALSE)
# setwd(work_path)

# # Your scored edges CSV
# edges_csv <- file.path(USER_ROOT, "results/ucell/full_scored_edges_with_pregastrulation.csv")

# # Early object needed for the small sanity check later
# # Put a copy here once, then you are independent
# obj_path <- file.path(work_path, "obj_Early_PS.rds")
# if (!file.exists(obj_path)) {
#   message("Note: obj_Early_PS.rds not found in your output folder. ",
#           "Sanity-check code will be skipped.")
# }



# ==== PATHS: Read from Holly/Qiu, write to Chris ====
INPUT_QIU_OTHER   <- "/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/other"
INPUT_HOLLY_TREE  <- "/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/results/tree"
OUTPUT_CHRIS_TREE <- "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/tree"

# Make sure your output folder exists
dir.create(OUTPUT_CHRIS_TREE, recursive = TRUE, showWarnings = FALSE)

# Inputs
nodes_path   <- file.path(INPUT_QIU_OTHER, "nodes.txt")
edges_csv    <- file.path("/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell",
                          "full_scored_edges_with_pregastrulation.csv")

# Early object: prefer your copy if you place one in your output, else use Holly’s
obj_candidates <- c(
  file.path(OUTPUT_CHRIS_TREE, "obj_Early_PS.rds"),
  file.path(INPUT_HOLLY_TREE,  "obj_Early_PS.rds")
)
obj_path <- obj_candidates[file.exists(obj_candidates)][1]  # may be NA if neither exists


# Put these near the top, after you define QiuFile_path and work_path
holly_results <- "/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/results/tree/"

# obj_candidates <- c(
#   file.path(work_path, "obj_Early_PS.rds"),
#   file.path(holly_results, "obj_Early_PS.rds")
# )

# obj_path <- obj_candidates[file.exists(obj_candidates)][1]
# if (is.na(obj_path)) {
#   stop("obj_Early_PS.rds not found in work_path or Holly's results folder.")
# }

obj_early <- readRDS(obj_path)
df_cell_early <- obj_early@meta.data

# --- replacement helpers for missing JAX_help_code_xy.R ---
mergeRedundantCelltype <- function(nodes_df, edges_df,
                                   merge_by = "celltype_new",
                                   filtered_by = "Spatial continuity") {
  ed <- edges_df
  if (!is.null(filtered_by) && "edge_type" %in% names(ed)) {
    ed <- ed[ed$edge_type != filtered_by, , drop = FALSE]
  }

  # pick one row per cell type
  nodes_rep <- nodes_df %>%
    dplyr::group_by(.data[[merge_by]]) %>%
    dplyr::slice(1) %>%
    dplyr::ungroup()
  nodes_rep$new_id <- seq_len(nrow(nodes_rep))

  # map names to IDs
  id_map <- stats::setNames(nodes_rep$new_id, nodes_rep[[merge_by]])
  ed$x_id <- id_map[ed$x_name]
  ed$y_id <- id_map[ed$y_name]
  ed <- ed[!is.na(ed$x_id) & !is.na(ed$y_id), , drop = FALSE]

  list(nodes = nodes_rep, edges = ed)
}
# ------------------------------------------------------------


# nodes = read.table(paste0(QiuFile_path, "nodes.txt"), header=T, as.is=T, sep="\t")
nodes <- read.table(nodes_path, header = TRUE, sep = "\t", as.is = TRUE)
dim(nodes)  # [1] 283   4
head(nodes)
            # system meta_group celltype_new celltype_num
# 1 Pre_gastrulation     PGa_M1       Oocyte            2
# 2 Pre_gastrulation     PGa_M2       1-cell            3
# 3 Pre_gastrulation     PGa_M3       2-cell            3
# 4 Pre_gastrulation     PGa_M4       4-cell            3
# 5 Pre_gastrulation     PGa_M5       8-cell            3
# 6 Pre_gastrulation     PGa_M6       Morula            3
length(unique(nodes$system))  #[1] 14
length(unique(nodes$celltype_new))   #[1] 262
unique(nodes$system)
 # [1] "Pre_gastrulation"       "Gastrulation"           "Endothelium"
 # [4] "Epithelial_cells"       "Eye"                    "Gut"
 # [7] "PNS_glia"               "PNS_neurons"            "Renal"
# [10] "Lateral_plate_mesoderm" "Blood"                  "Brain_spinal_cord"
# [13] "Mesoderm"               "Notochord"

### now we merged edges which have been manually reviewed.

### edges_1 includes edges from pre-gastrulation and gastrulation stages
# edges_1 = read.table(paste0(QiuFile_path, "edges_1.txt"), header=F, as.is=T, sep="\t")
# ### edges_2 includes edges from organogenesis & fetal development
# edges_2 = read.table(paste0(QiuFile_path, "edges_2.txt"), header=F, as.is=T, sep="\t")
# ### edges_3 includes edges which are manually added to connect blood and PNS-neuron
# edges_3 = read.table(paste0(QiuFile_path, "edges_3.txt"), header=F, as.is=T, sep="\t")
# edges = rbind(edges_1, edges_2, edges_3)
# names(edges) = c("system", "x", "y", "x_name", "y_name", "edge_type")
# length((unique(c(edges$x, edges$y))))
# write.table(edges, paste0(work_path, "edges.txt"), row.names=F, sep="\t", quote=F)

# edges = read.table(paste0(QiuFile_path, "edges.txt"), header=T, as.is=T, sep="\t")
edges = read.csv("/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/full_scored_edges_with_pregastrulation.csv")
dim(edges)  #[1] 507   6
head(edges)
            # system      x      y x_name          y_name
# 1 Pre_gastrulation PGa_M1 PGa_M2 Oocyte          1-cell
# 2 Pre_gastrulation PGa_M2 PGa_M3 1-cell          2-cell
# 3 Pre_gastrulation PGa_M3 PGa_M4 2-cell          4-cell
# 4 Pre_gastrulation PGa_M4 PGa_M5 4-cell          8-cell
# 5 Pre_gastrulation PGa_M5 PGa_M6 8-cell          Morula
# 6 Pre_gastrulation PGa_M6 PGa_M7 Morula Inner cell mass
                  # edge_type
# 1 Developmental progression
# 2 Developmental progression
# 3 Developmental progression
# 4 Developmental progression
# 5 Developmental progression
# 6 Developmental progression
length((unique(c(edges$x, edges$y))))  # 283
length(unique(edges$system))  #[1] 15
table(edges$edge_type)    
     # Dataset equivalence Developmental progression        Spatial continuity
                   # 55                       307                       145
setdiff(edges$system, nodes$system)			# "Gastrulation_E8.5b" Is this the 
setdiff(nodes$system, edges$system)			# character(0)

#######################################################
##### maping celltype_new to cell_id ##################
#######################################################

df_cell_later = readRDS(file=paste0(QiuFile_path, 'df_cell_graph.rds'))
dim(df_cell_later)  # 11441407        5
colnames(df_cell_later)
# [1] "cell_id"         "celltype_update" "system"          "meta_group"
# [5] "celltype_new"
all(df_cell_later$system %in% nodes$system)  #TRUE
all(df_cell_later$celltype_new %in% nodes$celltype_new)  #TRUE
length(unique(df_cell_later$celltype_new))  # 231


# an object generated by following the code 'https://github.com/ChengxiangQiu/JAX_code/blob/main/Section_6_development_tree/step1_Early_stage_graph.R'
# tmp = readRDS( file= paste0(QiuFile_path, "obj_Early_PS.rds")) # 'No such file or directory'
obj_early = readRDS( file= paste0(work_path, "obj_Early_PS.rds"))
df_cell_early = obj_early@meta.data
dim(df_cell_early)  #[1] 111090      8
colnames(df_cell_early)
# [1] "orig.ident"   "nCount_RNA"   "nFeature_RNA" "day"          "group"
# [6] "cell_state"   "cell_type"    "cell_id"
head(df_cell_early,3)
                             # orig.ident nCount_RNA nFeature_RNA day
# E3.5_P8_Cell2_embryo1_single "E3.5"     "  823281" " 5049"      "E3.5"
# E3.5_P8_Cell3_embryo2_single "E3.5"     "  752208" " 5865"      "E3.5"
# E3.5_P8_Cell5_embryo3_single "E3.5"     "  696715" " 5390"      "E3.5"
                             # group      cell_state
# E3.5_P8_Cell2_embryo1_single "Mohammed" "E3.5:Inner cell mass"
# E3.5_P8_Cell3_embryo2_single "Mohammed" "E3.5:Inner cell mass"
# E3.5_P8_Cell5_embryo3_single "Mohammed" "E3.5:Inner cell mass"
                             # cell_type         cell_id
# E3.5_P8_Cell2_embryo1_single "Inner cell mass" "E3.5_P8_Cell2_embryo1_single"
# E3.5_P8_Cell3_embryo2_single "Inner cell mass" "E3.5_P8_Cell3_embryo2_single"
# E3.5_P8_Cell5_embryo3_single "Inner cell mass" "E3.5_P8_Cell5_embryo3_single"
#all(df_cell_early$system %in% nodes$system)  #TRUE
all(df_cell_early$cell_type %in% nodes$celltype_new)  #FALSE
length(unique(df_cell_early$cell_type))  # 47
setdiff(df_cell_early$cell_type, nodes$celltype_new)
# [1] "Primitive streak and adjacent ectoderm"
 # [2] "Mixed mesoderm"
 # [3] "Amniochorionic mesoderm"
 # [4] "Rostral neuroectoderm"
 # [5] "Caudal lateral epiblast"
 # [6] "Caudal neuroectoderm"
 # [7] "Paraxial mesoderm A"
 # [8] "Paraxial mesoderm B"
 # [9] "Paraxial mesoderm C"
# [10] "Amniochorionic mesoderm B"
# [11] "Forebrain/midbrain"
# [12] "Neuromesodermal progenitors"
# [13] "Amniochorionic mesoderm A"
# [14] "Fusing epithelium"
# [15] "Posterior floor plate"


#######################################################
##### filter edges as author did for Fig 5g  ###############
#######################################################
	   
### To better visualize the result, we took out the spatial continuity edges, and also collapse reundant nodes
edges_sub = edges[edges$edge_type != "Spatial continuity",]
length((unique(c(edges_sub$x, edges_sub$y))))  # 281

edges_sub = rbind(edges_sub, edges[edges$x %in% c("BS_M37", "BS_M39") | edges$y %in% c("BS_M37", "BS_M39"),])
length((unique(c(edges_sub$x, edges_sub$y))))  # 283

dim(edges_sub) # [1] 366   6
# tmp = read.table( file= paste0(QiuFile_path, "edges_sub.txt")) # 'No such file or directory'
# write.table(edges_sub, paste0(work_path, "edges_sub.txt"), row.names=F, sep="\t", quote=F)
write.table(edges_sub, file.path(OUTPUT_CHRIS_TREE, "edges_sub.txt"),
            row.names = FALSE, sep = "\t", quote = FALSE)

## removing redundant nodes ##
edges_sub$x_y = paste0(edges_sub$x, ":", edges_sub$y)
## find redundant edges from "Pre_gastrulation"(edges_x_1)  and "Gastrulation_E8.5b" (edges_x_2)
edges_x_1 = edges_sub[edges_sub$x_name == edges_sub$y_name & edges_sub$system == "Pre_gastrulation",]
edges_x_2 = edges_sub[edges_sub$x_name == edges_sub$y_name & edges_sub$system == "Gastrulation_E8.5b",]
## Identifying Edges to Keep by Matching Nodes by filtering edges that are related to the self-looping or redundant nodes identified in edges_x_1 and edges_x_2
edges_x_3 = edges_sub[edges_sub$x %in% as.vector(edges_x_1$y),]
edges_x_4 = edges_sub[edges_sub$x %in% as.vector(edges_x_2$y),]
## Renaming Nodes to Remove Redundancies, and replaces x with new_x values
edges_x_3_ = edges_x_3 %>% left_join(edges_x_1 %>% select(x,y) %>% rename(new_x = x, x=y), by = "x")
edges_x_3$x = as.vector(edges_x_3_$new_x)
edges_x_4_ = edges_x_4 %>% left_join(edges_x_2 %>% select(x,y) %>% rename(new_x = x, x=y), by = "x")
edges_x_4$x = as.vector(edges_x_4_$new_x)
## Removing Redundant Edges
edges_x_5 = edges_sub[!edges_sub$x_y %in% c(edges_x_1$x_y, edges_x_2$x_y, edges_x_3$x_y, edges_x_4$x_y),]
## Combining Filtered Edges
edges_x = rbind(edges_x_3, edges_x_4, edges_x_5)
print(edges_x[edges_x$x_name == edges_x$y_name,])
edges_x = edges_x[edges_x$x_name != edges_x$y_name,]
edges_x$x_y_name = paste0(edges_x$x_name, ":", edges_x$y_name)
x_table = table(edges_x$x_y_name)
tmp = edges_x[edges_x$x_y_name %in% names(x_table)[x_table != 1],]
print(tmp[order(tmp$x_name),])

redundant_edges = c("En_M5:En_M1", "Ga_M5:Ga_M6", "L_M7:L_M3", "En_M7:En_M5", "Ga_M23:En_M5", "BS_M20:BS_M2", "Ga_M17:En_M7")
edges_x = edges_x[!edges_x$x_y %in% redundant_edges,]
print(length(unique(c(edges_x$x, edges_x$y))))  # 262
print(length(unique(c(edges_x$x_name, edges_x$y_name))))  # 262

dim(edges_x)  # [1] 338   8  #!!!!!!!!!!!!!!!!
write.table(edges_x, file.path(OUTPUT_CHRIS_TREE, "nodes_sub.txt"), row.names=F, sep="\t", quote=F)

# dim(nodes_sub)  # 262   4    #!!!!!!!!!!!!!!!!
# nodes_sub = nodes[nodes$meta_group %in% c(edges_x$x, edges_x$y),]
# write.table(nodes_sub, paste0(work_path, "nodes_sub.txt"), row.names=F, sep="\t", quote=F)
nodes_sub <- nodes[nodes$meta_group %in% unique(c(edges_x$x, edges_x$y)), ]

# Sanity checks
stopifnot(nrow(nodes_sub) > 0)
message("nodes_sub rows: ", nrow(nodes_sub))
message("unique node names in edges_x: ", length(unique(c(edges_x$x_name, edges_x$y_name))))

# Save
write.table(nodes_sub, file.path(OUTPUT_CHRIS_TREE, "nodes_sub.txt"),
            row.names = FALSE, sep = "\t", quote = FALSE)



# tmp = read.table( file= paste0(QiuFile_path, "nodes_sub.txt")) # 'No such file or directory'

##############################################################
### plot graph as Fig5g shows ###
##############################################################

# source("/project/xyang2/SHH/source_midway3/JAX_help_code_xy.R")
# source("/project/xyang2/SHH/source_midway3/JAX_color_code.R")
# QiuFile_path #'/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/other/'
# tome_path = '/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/tome/mm/'
# work_path = "/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/results/tree/"
# setwd(work_path)


nodes = read.table(paste0(QiuFile_path, "nodes.txt"), header=T, as.is=T, sep="\t")
dim(nodes)  # [1] 283   4
head(nodes)
            # system meta_group celltype_new celltype_num
# 1 Pre_gastrulation     PGa_M1       Oocyte            2
# 2 Pre_gastrulation     PGa_M2       1-cell            3
# 3 Pre_gastrulation     PGa_M3       2-cell            3
# 4 Pre_gastrulation     PGa_M4       4-cell            3
# 5 Pre_gastrulation     PGa_M5       8-cell            3
# 6 Pre_gastrulation     PGa_M6       Morula            3


edges = read.table(paste0(QiuFile_path, "edges.txt"), header=T, as.is=T, sep="\t")
dim(edges)  #[1] 507   6
head(edges)
            # system      x      y x_name          y_name
# 1 Pre_gastrulation PGa_M1 PGa_M2 Oocyte          1-cell
# 2 Pre_gastrulation PGa_M2 PGa_M3 1-cell          2-cell
# 3 Pre_gastrulation PGa_M3 PGa_M4 2-cell          4-cell
# 4 Pre_gastrulation PGa_M4 PGa_M5 4-cell          8-cell
# 5 Pre_gastrulation PGa_M5 PGa_M6 8-cell          Morula
# 6 Pre_gastrulation PGa_M6 PGa_M7 Morula Inner cell mass
                  # edge_type
# 1 Developmental progression
# 2 Developmental progression
# 3 Developmental progression
# 4 Developmental progression
# 5 Developmental progression
# 6 Developmental progression

table(edges$edge_type)
# # Dataset equivalence Developmental progression Spatial continuity 
#          55                       307 			 145
       
                     
					  
# Create graph from edges , testing version ################################

# g <- graph_from_data_frame(d=edges[,c("x", "y")], vertices=nodes[,c("meta_group", "celltype_new", "celltype_num")], directed=TRUE)

# # Set node and edge attributes
# V(g)$label <- nodes$celltype_new
# V(g)$color <- as.factor(nodes$system)  # Assign colors based on system
# V(g)$shape <- "rectangle"

# # Plot the graph
# pdf(file='myTree.pdf', width=12, height=10)
# plot(g,
     # layout = layout_as_tree(g, root = which(V(g)$label == "Oocyte")),
     # vertex.size = 20,
     # vertex.label.cex = 0.8,
     # vertex.color = rainbow(length(unique(nodes$system)))[as.numeric(V(g)$color)],
     # edge.arrow.size = 0.5,
     # main = "Developmental Progression Tree")
# # IGRAPH 4efb32f DN-- 283 507 --
# dev.off()


# Map nodes to numbers
nodes$id <- 1:nrow(nodes)
node_name_map <- setNames(nodes$id, nodes$meta_group)

# Replace node names in edges with numbers
edges$x_number <- node_name_map[edges$x]
edges$y_number <- node_name_map[edges$y]

# Create a graph object
g <- graph_from_data_frame(d=edges[,c("x_number", "y_number")], 
							vertices = nodes[,c("id", "meta_group", "celltype_new", "system")],
							directed = TRUE) %>%
		as_tbl_graph()
names(vertex.attributes(g))
# [1] "name"         "meta_group"   "celltype_new" "system"      
# [5] "color" 
V(g)$id <- nodes$id

# # === Add SHH delta coloring ===
# # Assign SHH delta to edge weight and color by absolute delta
# E(g)$weight <- tbs_final$edges$abs_delta
# E(g)$color <- scales::col_numeric("RdBu", domain = NULL)(E(g)$weight)

E(g)$label <- edges$edge_type

#####################################################
# Plot the graph to a PDF
# the author did: 'For presentation purposes, 
# we removed most ‘spatial continuity’ edges and merged
# nodes with redundant labels derived from different datasets, resulting in a
# rooted graph comprising 262 cell-type nodes and 338 edges."
#
# following the same strategy, our code finds: 262 cell-type nodes and 355 edges
#####################################################

g_sub <- g %>% 
  activate(edges) %>%
  filter(label != 'Spatial continuity') %>%
  as_tbl_graph()  # Convert back to tbl_graph after filtering
g_sub
# A tbl_graph: 283 nodes and 362 edges
V(g_sub)$celltype_new %>% unique %>% length   #[1] 262
  
nodes_sub = subset(nodes, celltype_new %in% V(g_sub)$celltype_new)  
tbs_final = mergeRedundantCelltype(nodes_sub, edges, merge_by = "celltype_new", filtered_by='Spatial continuity')
dim(tbs_final$nodes) #[1] 262   6
dim(tbs_final$edges) #[1] 355  10
table(tbs_final$edges$edge_type)
      # Dataset equivalence Developmental progression 
                       # 53                       302
unique(tbs_final$edges$system)
 # [1] "Pre_gastrulation"       "Gastrulation"           "Gastrulation_E8.5b"     "Notochord"             
 # [5] "Blood"                  "Eye"                    "Renal"                  "Gut"                   
 # [9] "PNS_glia"               "PNS_neurons"            "Lateral_plate_mesoderm" "Endothelium"           
# [13] "Mesoderm"               "Epithelial_cells"       "Brain_spinal_cord" 
unique(tbs_final$nodes$system)
 # [1] "Pre_gastrulation"       "Gastrulation"           "Endothelium"            "Epithelial_cells"      
 # [5] "Eye"                    "Gut"                    "PNS_glia"               "PNS_neurons"           
 # [9] "Renal"                  "Lateral_plate_mesoderm" "Blood"                  "Brain_spinal_cord"     
# [13] "Mesoderm"               "Notochord" 

 write.table(tbs_final$nodes, file = file.path("/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/tree/nodes_filtered.txt"), row.names=FALSE, sep='\t') #!!!!!!!
 write.table(tbs_final$edges, file = file.path("/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/tree/edges_filtered.txt"), row.names=FALSE, sep='\t') #!!!!!!!


# Create a graph object
g <- graph_from_data_frame(d=tbs_final$edges[,c("x_id", "y_id")], 
							vertices = tbs_final$nodes[,c("new_id", "meta_group", "celltype_new", "system")],
							directed = TRUE) %>%
		as_tbl_graph()
names(vertex.attributes(g))
# [1] "name"         "meta_group"   "celltype_new" "system"      
# [5] "color" 
V(g)$id <- tbs_final$nodes$new_id

# Assign colors by system
n_colors <- length(unique(tbs_final$nodes$system))  # 14
palette <- colorRampPalette(brewer.pal(8, "Set1"))(n_colors)
V(g)$color <- palette[as.factor(tbs_final$nodes$system)]
levels(as.factor(tbs_final$nodes$system))
 # [1] "Blood"                  "Brain_spinal_cord"     
 # [3] "Endothelium"            "Epithelial_cells"      
 # [5] "Eye"                    "Gastrulation"          
 # [7] "Gut"                    "Lateral_plate_mesoderm"
 # [9] "Mesoderm"               "Notochord"             
# [11] "PNS_glia"               "PNS_neurons"           
# [13] "Pre_gastrulation"       "Renal"  

# Set edge labels and other properties
E(g)$label <- tbs_final$edges$edge_type
E(g)$arrow.size <- 0.5
E(g)$edge.width <- 0.5
E(g)$edge.color <- "grey"


# --- Try to attach SHH edge deltas (abs_delta, delta) robustly ---
attach_edge_deltas <- function(tbs_edges,
                               search_root = "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell") {
  # 1) First, check the already-loaded 'edges' object if it exists
  candidates <- list()
  if (exists("edges")) {
    candidates <- c(candidates, list(edges))
  }

  # 2) Search the results tree for plausible files
  found_files <- list.files(search_root,
                            pattern = "(edge|Edges).*\\.(csv|rds)$",
                            full.names = TRUE, recursive = TRUE)
  # Always include your known file last as a fallback (even if it lacks deltas)
  found_files <- unique(c(found_files,
                          "/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/full_scored_edges_with_pregastrulation.csv"))

  # Helper to read any file safely
  read_any <- function(f) {
    ext <- tolower(tools::file_ext(f))
    if (ext == "csv") {
      tryCatch(read.csv(f, check.names = FALSE), error = function(e) NULL)
    } else if (ext == "rds") {
      tryCatch(readRDS(f), error = function(e) NULL)
    } else NULL
  }

  for (f in found_files) {
    if (is.character(f)) {
      df <- read_any(f)
    } else {
      df <- f
      f <- "<in-memory 'edges'>"
    }
    if (is.null(df)) next

    # Needs x_name, y_name and the delta columns
    if (all(c("x_name","y_name") %in% names(df)) &&
        all(c("abs_delta","delta") %in% names(df))) {

      message("Merging SHH deltas from: ", f)
      # Build join key on names (stable after collapsing)
      tbs_edges$.key <- paste(tbs_edges$x_name, tbs_edges$y_name, sep = " :: ")
      df$.key        <- paste(df$x_name,        df$y_name,        sep = " :: ")

      keep_cols <- c(".key", "abs_delta", "delta")
      out <- merge(tbs_edges, df[, keep_cols, drop = FALSE], by = ".key", all.x = TRUE)
      out$.key <- NULL
      return(out)
    }
  }

  message("No file with both 'abs_delta' and 'delta' found. Proceeding without edge deltas.")
  tbs_edges
}

# Apply it
tbs_final$edges <- attach_edge_deltas(tbs_final$edges)

# Flag availability
have_edge_delta <- all(c("abs_delta","delta") %in% names(tbs_final$edges))


# === Add SHH delta edge aesthetics ===
stopifnot(length(E(g)) == nrow(tbs_final$edges))
stopifnot(all(c("abs_delta","delta") %in% names(tbs_final$edges)))

E(g)$abs_delta <- tbs_final$edges$abs_delta
E(g)$delta     <- tbs_final$edges$delta

names(edge.attributes(g))
#[1] "label"      "arrow.size" "edge.width" "edge.color"


pdf(file.path(OUTPUT_CHRIS_TREE, "myTree_filtered.pdf"), width = 20, height = 6)
ggraph(g, layout = "tree") + 
  geom_edge_link(arrow = arrow(length = unit(2, 'mm')), end_cap = circle(3, 'mm')) +
  geom_node_point(aes(color = as.factor(system)), size = 4) +
  geom_node_text(aes(label = id), vjust = 0.5, hjust = 0.5, size = 2) +
  scale_color_manual(values = palette, name = "System") +  # Add legend for system colors
  theme_void() +
  theme(legend.position = "bottom") +
  labs(color = "System")  
  
dev.off()

sub_sys = unique(tbs_final$nodes$system) %>%
		setdiff(., c("Pre_gastrulation" , "Gastrulation"))

pdf(file.path(OUTPUT_CHRIS_TREE, "myTree_filtered_perSystem.pdf"), width = 8, height = 6)
for( i in sub_sys){
# i="Lateral_plate_mesoderm"
	g_sub <- g %>%
		  activate(nodes) %>%
		  filter(system %in% c("Pre_gastrulation", "Gastrulation", i)) %>%
		  as_tbl_graph()	
	p = ggraph(g_sub, layout = "tree") + 
	  geom_edge_link(arrow = arrow(length = unit(2, 'mm')), end_cap = circle(3, 'mm')) +
	  geom_node_point(aes(color = as.factor(system)), size = 4) +
	  geom_node_text(aes(label = id), vjust = 0.5, hjust = 0.5, size = 2) +
	  scale_color_manual(values = palette, name = "System") +  # Add legend for system colors
	  #scale_color_manual(values = palette, name = "celltype_new") +  # Update legend to reflect celltype_new
	  theme_void() +
	  theme(legend.position = "top") +
	  labs(color = "System")  
	#print(p)  
	# Add combined labels as text at the bottom
	g_sub_sys =  g_sub %>%
		  activate(nodes) %>%
		  filter(system %in% c("Gastrulation", i)) %>%
		  as_tbl_graph()	
	combined_labels = paste(paste(V(g_sub_sys)$id, V(g_sub_sys)$celltype_new, sep = "_"), collapse = "\n")
    text_grob <- textGrob(combined_labels, x = 0, just = "left", gp = gpar(fontsize = 5))
   
  # Arrange plot and text side by side
    print(grid.arrange(p, text_grob, ncol = 2, widths = c(3, 1)) )
	
	Sys.sleep(3)
}	
dev.off()
# scp -p -r xyang2@midway3.rcc.uchicago.edu:/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/results/tree/*.pdf  D:\projects\SHH\result\Qiu_TimeLapse\tree\.

# # try more layout, can't make correct label yet !!!!
# layout <- layout_as_tree(g_sub, root = 1)  # Assuming node 1 is the root
# plot(g_sub,
#      layout = layout,
#      vertex.label = V(g_sub)$name,
#      vertex.size = 5,
#      vertex.label.cex = 0.5,
#      edge.arrow.size = 0.2,
#      edge.width = 0.2)

# ggraph(g_sub, 'partition', circular = TRUE) + 
#   geom_node_arc_bar(aes(fill = depth), size = 0.25) + 
#   coord_fixed()

# ggraph(g_sub, 'treemap') + #, weight = size
#   geom_edge_link() + 
#   geom_node_point(aes(colour = system)) +
#   coord_fixed()
 
# # library(sfnetworks)
# # ggraph(g_sub, 'sf') + 
# #   geom_edge_sf(aes(color = system)) + 
# #   geom_node_sf(size = 0.3)

# ======= BEGIN ADD (final DAG plot) =======
suppressPackageStartupMessages(library(ggraph))

pdf(file.path(OUTPUT_CHRIS_TREE, "myTree_filtered_perSystem.pdf"), width = 20, height = 10)
p <- ggraph(g, layout = "sugiyama") +
  geom_edge_link(arrow = arrow(length = unit(2, "mm")),
                 end_cap = circle(2, "mm"),
                 size = 0.3, colour = "grey60") +
  geom_node_point(aes(color = as.factor(system)), size = 1.8) +
  geom_node_text(aes(label = celltype_new), size = 1.6, vjust = -0.5, check_overlap = TRUE) +
  scale_color_manual(values = palette, name = "System") +
  theme_void() +
  theme(legend.position = "bottom", plot.margin = margin(10, 10, 10, 10))
print(p)
dev.off()

pdf(file.path(OUTPUT_CHRIS_TREE, "myTree_filtered_sugiyama_unlabeled.pdf"), width = 20, height = 10)
p2 <- ggraph(g, layout = "sugiyama") +
  geom_edge_link(arrow = arrow(length = unit(2, "mm")),
                 end_cap = circle(2, "mm"),
                 size = 0.3, colour = "grey60") +
  geom_node_point(aes(color = as.factor(system)), size = 2) +
  scale_color_manual(values = palette, name = "System") +
  theme_void() +
  theme(legend.position = "bottom", plot.margin = margin(10, 10, 10, 10))
print(p2)
dev.off()



pdf(file.path(OUTPUT_CHRIS_TREE, "myTree_filtered_sugiyama_SHHedges.pdf"), width = 20, height = 10)
p <- ggraph(g, layout = "sugiyama") +
  geom_edge_link(aes(edge_width = abs_delta, edge_colour = abs(delta)),
                 arrow = arrow(length = unit(2, "mm")),
                 end_cap = circle(2, "mm")) +
  scale_edge_colour_viridis(name = "|delta|", option = "C", direction = 1) +
  scale_edge_width(range = c(0.2, 2), guide = guide_legend(title = "abs_delta")) +
  geom_node_point(aes(color = as.factor(system)), size = 2) +
  theme_void() +
  theme(legend.position = "bottom")
print(p)
dev.off()
# ======= END ADD =======

  
#####################################################
## check if nodes match edge names #################
#####################################################

length(unique(nodes$meta_group))  # 283
length(unique(c(edges$x, edges$y)))  # 283
all(unique(c(edges$x, edges$y)) %in% unique(nodes$meta_group))  #TRUE


## check if I can track cell IDs for each nodes #################
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

#Qiu said: The data can be divided into three main parts:
## 1) Pre-gastrulation
# Please filter the cell metadata by the group column for "Cheng" and "Mohammed," using the "cell_type" column as the nodes. You can access the data here.
# https://shendure-web.gs.washington.edu/content/members/cxqiu/public/backup/jax/download/other/pd_processed_Early_PSE65.rds
# Please note that for nodes such as Oocyte, 1-cell, and Morula from the very early stages, the cell numbers are limited, and I did not analyze them in detail.
# If you need this information, you can find it in the paper.
# https://www.nature.com/articles/nature12364
x= which(nodes$system =='Pre_gastrulation')
y = unique(nodes[x, 'celltype_new']) 

obj_e = readRDS(paste0(work_path, "obj_Early_PS.rds"))
table(obj_e[[]]$group)
      # Cheng    Mohammed Pijuan_Sala
       # 1724         509      108857
# pd_early = pd_pre
pd_early <- df_cell_early

pd_pre = subset(pd_early, group %in% c('Cheng',    'Mohammed'))
dim(pd_pre)   #[1] 2233    8
colnames(pd_pre)
# [1] "orig.ident"   "nCount_RNA"   "nFeature_RNA" "day"          "group"
# [6] "cell_state"   "cell_type"    "cell_id"

all(y %in% pd_pre$cell_type)  #[1] FALSE
setdiff(y, pd_pre$cell_type)  # 
# [1] "Oocyte"           "1-cell"           "2-cell"           "4-cell"
# [5] "8-cell"           "Morula"           "Trophectoderm"    "Primitive streak"
setdiff(pd_pre$cell_type, y)  # "Primitive streak and adjacent ectoderm"


pd_pre_Qiu = readRDS(paste0(QiuFile_path, "pd_processed_Early_PSE65.rds")) 
dim(pd_pre_Qiu)  #[1] 5717   12
table(pd_pre_Qiu$group)
      # Cheng    Mohammed Pijuan_Sala
    #   1724         509        3484
pd_pre_Qiu = subset(pd_pre_Qiu, group %in% c('Cheng',    'Mohammed'))
colnames(pd_pre_Qiu)
# [1] "orig.ident"           "nCount_RNA"           "nFeature_RNA"
 # [4] "day"                  "group"                "cell_state"
 # [7] "cell_type"            "cell_id"              "integrated_snn_res.1"
# [10] "seurat_clusters"      "UMAP_1"               "UMAP_2"
all(y %in% pd_pre_Qiu$cell_type)  #[1] FALSE
all(pd_pre_Qiu$orig.ident %in% pd_pre$orig.ident) #TRUE
setdiff(y, pd_pre_Qiu$cell_type)  # 
# [1] "Oocyte"           "1-cell"           "2-cell"           "4-cell"
# [5] "8-cell"           "Morula"           "Trophectoderm"    "Primitive streak"
setdiff(pd_pre_Qiu$cell_type,  y)  # 
#[1] "Primitive streak and adjacent ectoderm"

   
# 2) Gastrulation
# Filter the cell metadata by the group column for "Pijuan_Sala," using the "celltype_update" column as the nodes. The data is available here.
# https://shendure-web.gs.washington.edu/content/members/cxqiu/public/backup/jax/download/other/pd_processed_PS_JaxE85.rds
x= which(nodes$system =='Gastrulation') 
y = unique(nodes[x, 'celltype_new']) 

pd_gas = subset(pd_early, group == 'Pijuan_Sala')
dim(pd_gas)   #[1] 108857    8
colnames(pd_gas)
# [1] "orig.ident"   "nCount_RNA"   "nFeature_RNA" "day"          "group"
# [6] "cell_state"   "cell_type"    "cell_id"

pd_gas_Qiu = readRDS(paste0(QiuFile_path, "pd_processed_PS_JaxE85.rds")) 
dim(pd_gas_Qiu)  #[1] 262454     19
table(pd_gas_Qiu$group)
      #      Jax Pijuan_Sala
     #    153597      108857
pd_gas_Qiu = subset(pd_gas_Qiu, group =='Pijuan_Sala')
colnames(pd_gas_Qiu)
# [1] "orig.ident"      "nCount_RNA"      "nFeature_RNA"    "day"
 # [5] "group"           "cell_state"      "cell_type"       "cell_id"
 # [9] "UMAP_1"          "UMAP_2"          "UMAP_3d_1"       "UMAP_3d_2"
# [13] "UMAP_3d_3"       "Louvain_res_1"   "Louvain_res_2"   "Louvain_res_5"
# [17] "pre_celltype"    "celltype_update" "Anno"
all(y %in% pd_gas_Qiu$celltype_update)  #[1] TRUE
 
all(pd_gas_Qiu$orig.ident %in% pd_gas$orig.ident)  # TRUE



# 3) Organogenesis
# For this section, use the "system" and "celltype_new" columns as the systems and nodes. You can find the data here.
# https://shendure-web.gs.washington.edu/content/members/cxqiu/public/backup/jax/download/other/df_cell_graph.rds
x= which(nodes$system %in% c('Pre_gastrulation','Gastrulation'))

# pd_all is the new E8-P0 data
pd_all = readRDS(paste0(QiuFile_path, "df_cell_graph.rds"))
rownames(pd_all) = as.vector(pd_all$cell_id)
dim(pd_all) # [1] 11441407        5
colnames(pd_all)  # [1] "cell_id"         "celltype_update" "system"          "meta_group"
# [5] "celltype_new"
y = unique(nodes[-x, 'celltype_new']) 
length(y)  # [1] 231
all(y %in% pd_all$celltype_new)  #TRUE 
y = unique(nodes[-x, 'meta_group']) 
length(y)  # [1] 231
all(y %in% pd_all$meta_group)  #TRUE 

### from which object to extract the cells
y = unique(nodes[-x, 'system']) 
length(y)  # [1] 12
subsys = list.files(path = '/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/results/', pattern='_adata_scale.obs.csv')  
subsys = lapply(subsys, function(x) unlist(strsplit(x,'_adata_'))[1]) %>%
			unlist
all(y %in% subsys)  #TRUE 



##############################################################
### making Histogram for accepted edges and rejected edges (NOT RUN YET!!!) ###
# NOT repeatable because the file "edges_MNNs.txt" is missing !!!!!!!
##############################################################

# source("/project/xyang2/SHH/source_midway3/JAX_help_code_xy.R")
# source("/project/xyang2/SHH/source_midway3/JAX_color_code.R")
# QiuFile_path #'/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/other/'
# tome_path = '/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/tome/mm/'
# work_path = "/project/imoskowitz/xyang2/SHH/Qiu_TimeLapse/results/tree/"
# setwd(work_path)

# tmp = read.table(paste0(QiuFile_path, "edges_MNNs.txt"), header=T, sep="\t", as.is=T) # 'No such file or directory'
#dat = read.table(paste0(work_path, "edges_MNNs.txt"), header=T, sep="\t", as.is=T) # 'No such file or directory'
# dat = read.table(paste0(work_path, "edges_sub.txt"), header=T, sep="\t", as.is=T)  
# colnames(dat)
# #[1] "system"    "x"         "y"         "x_name"    "y_name"    "edge_type"

# dat = dat[dat$MNN_pairs_normalized > 1,]

# dat_1 = dat[dat$Comments %in% c("Developmental progression", "Spatial continuity"),]
# dat_2 = dat[dat$Comments %in% c("x","X"),]

# dat_uniq = NULL
# x_uniq = NULL
# for(i in 1:nrow(dat_2)){
#     tmp = paste0(dat_2$x[i], ":", dat_2$y[i])
#     tmp_r = paste0(dat_2$y[i], ":", dat_2$x[i])
#     if(tmp %in% x_uniq | tmp_r %in% x_uniq){
#         next
#     } else {
#         dat_uniq = rbind(dat_uniq, dat_2[i,])
#         x_uniq = c(x_uniq, tmp)
#     }
# }

# dat_1$group = "Accepted"
# dat_uniq$group = "Rejected"
# df = rbind(dat_1, dat_uniq)
# df$log2_MNN_pairs_normalized = log2(df$MNN_pairs_normalized)

# ### Extended Data Fig. 11d

# p <- df %>%
#     ggplot( aes(x=log2_MNN_pairs_normalized, fill=group)) +
#     geom_histogram( color="#e9ecef", alpha=0.5, position = 'identity') +
#     scale_fill_manual(values=c("#f85633", "#0058d6")) +
#     theme_ipsum() +
#     labs(fill="")



