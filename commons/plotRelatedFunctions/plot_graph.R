# Title     : Functional connectivity graph Visualization based on three different connectivity measures
# Objective : Functional connectivity visualization
# Created by: mohsen
# Modified on: 8/7/19

rm(list = ls(all=TRUE))
source("/home/mohsen/projects/network/commons/tools/basicFunctions.R")

args = commandArgs(TRUE)
path = as.character(args[1])
requiredDires = c("Correlation", "Hawkes", "MutualInformation")
subDirs = list.dirs(path, full.names = FALSE, recursive = FALSE)
subDirs = intersect(subDirs, requiredDires)


for (p in subDirs){

  read_path = paste(path, p, "/", sep = "")
  write_path = paste(read_path, "graphPlots", "/", sep = "")
  strength_path = paste(write_path, "strength", "/", sep = "")
  hub_path = paste(write_path, "hub", "/", sep = "")
  cluster_path = paste(write_path, "cluster", "/", sep = "")


  if(!dir.exists(write_path)){
      dir.create(file.path(write_path))
  }
  if(!dir.exists(strength_path)){
      dir.create(file.path(strength_path))
  }
  if(!dir.exists(hub_path)){
      dir.create(file.path(hub_path))
  }
  if(!dir.exists(cluster_path)){
      dir.create(file.path(cluster_path))
  }
  setwd(file.path(write_path))

  all_files = list.files(path = read_path)
  epochs_ordered = unique(sapply(strsplit(all_files[endsWith(all_files, ".gml")], "__"), function(x) x[1]))
  graphDataList = lapply(epochs_ordered, applyFunToGraphList, path = read_path, all_files = all_files)

## Writing Graph individualy - strength
  for(g in 1:length(epochs_ordered)){
    svglite(file = paste(strength_path, epochs_ordered[g] , '.svg', sep=''))
    graphPlot(graphDataList[[g]], "", "strength")
    dev.off()
  }

## Writing Graph individualy - hub
  for(g in 1:length(epochs_ordered)){
    svglite(file = paste(hub_path, epochs_ordered[g] , '.svg', sep=''))
    graphPlot(graphDataList[[g]], "", "hub")
    dev.off()
  }

# cluster
  svglite(file = "out3.svg")

## Writing Graph individualy
  for(g in 1:length(epochs_ordered)){
    svglite(file = paste(cluster_path, epochs_ordered[g], '.svg', sep=''))
    clusterGraph(graphDataList[[g]], "")
    dev.off()
  }

    print(paste("**** Generated Graph Visualization for method ", p, ".", sep=""))

}

## Plotting Graph degree distribution
data_frame = read.csv(file = paste(path, "all_chain_network_info.csv", sep=""), header = TRUE)
deg_df = data.frame("degree" = data_frame$degrees, "neuron" = data_frame$neuron,
                    "method" = data_frame$method, "epoch" = data_frame$epoch,
                    "chain" = data_frame$chain) %>%
                    group_by(neuron, epoch) %>%
                    summarise(degree = mean(degree)) %>%
                    arrange(neuron, epoch)

epochsUnique = unique(deg_df$epoch)

write_path = paste(path, "degreeDistribution", "/", sep = "")

  if(!dir.exists(write_path)){
      dir.create(file.path(write_path))
  }
setwd(file.path(write_path))

for (i in epochsUnique){
  ggsave(filename = paste(i, ".svg",sep = ""), plot = plot_degree(deg_df, i),
  width = 10, height = 8, units = "in", dpi = "retina", device='svg')
}
