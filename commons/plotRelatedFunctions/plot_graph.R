# Title     : Functional connectivity graph Visualization based on three different connectivity measures
# Objective : Functional connectivity visualization
# Created by: mohsen
# Modified on: 8/7/19

source("/home/mohsen/projects/network/commons/tools/basicFunctions.R")

args = commandArgs(TRUE)
path = as.character(args[1])
subDirs = list.dirs(path, full.names = FALSE, recursive = FALSE)

for (p in subDirs){

  read_path = paste(path, p, "/", sep = "")
  write_path = paste(read_path, "graph_plots", "/", sep = "")

  if(!dir.exists(write_path)){
      dir.create(file.path(write_path))
  }
  setwd(file.path(write_path))

  graph_lists = list.files(path = read_path)[endsWith(list.files(path = read_path), ".gml")]
  graphDataList = lapply(graph_lists, function(x) read_graph(file = paste(read_path, x, sep = ''), format = "gml"))

  ## Writing Graph individualy
  for(g in 1:length(graph_lists)){
    svglite(file = paste('strength-',graph_lists[g], '.svg', sep=''))
    graphPlot(graphDataList[[g]], graph_lists[g],"strength")
    dev.off()
  }

  # cluster
  svglite(file = "out3.svg")

  ## Writing Graph individualy
  for(g in 1:length(graph_lists)){
    svglite(file = paste('cluster-',graph_lists[g], '.svg', sep=''))
    clusterGraph(graphDataList[[g]], graph_lists[g])
    dev.off()
  }

    print(paste("**** Generated Graph Visualization for method ", p, ".", sep=""))

}



