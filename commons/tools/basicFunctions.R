# Title     : Various User defined function, Mostly needed for graph visualization
# Objective : Visualization
# Created by: mohsen
# Created on: 8/7/19

library(igraph)
library(svglite)
library(ggplot2)
require(dplyr)

# Rescale between a and b
stan = function(x, a, b){
  (x - min(x))/(max(x) - min(x))*(b-a) + a
}

# A star layout graph visualization with high degree node (strength) in center
graphPlot = function(g, main, type){

  weights_list = sapply(c(1:length(g)), function(x) E(g[[x]])$weight)
  length_vec = sapply(weights_list, function(x) length(x))
  min_length = min(length_vec)
  min_index = which.min(length_vec)

  average_weight = function(len, wlistelem){
    wlistelem[1:len]
  }

  weights_vec = rowMeans(sapply(weights_list, average_weight, len=min_length))
  # select minimim weight length graph
  G = g[[min_length]]
  E(G)$weight = weights_vec
  # Configuring Vertex size based on degree

  if(type == "strength"){

    deg <- rowMeans(sapply(c(1:length(g)), function(x) graph.strength(g[[x]])))
  } else{
    deg <- rowMeans(sapply(c(1:length(g)), function(x) hub_score(g[[x]])$vector))
  }
  V(G)$size <- stan(deg, 10, 30)

  # Set edge width based on weight
  E(G)$width <- stan(weights_vec, 1, 5)

  #change arrow size and edge color
  E(G)$arrow.size <- .4

  #Color scaling function
  #Using jet colour map
  c_scale <- colorRamp(c("#000066", "blue", "cyan",
                         "yellow", "red", "#660000"))
  #Applying the color scale to edge weights.
  #rgb method is to convert colors to a character vector.
  E(G)$color = apply(c_scale(stan(weights_vec, 0, 1)), 1, function(x) rgb(x[1]/255,x[2]/255,x[3]/255) )

  # Configuring layout
  coords <- layout_as_star(G, V(G)[which.max(V(G)$size)])

  plot(G, layout = coords, main = main)

}

# A cluster graph visualization based on cluster_optimal function from igraph package
clusterGraph = function(g, main){

  weights_list = sapply(c(1:length(g)), function(x) E(g[[x]])$weight)
  length_vec = sapply(weights_list, function(x) length(x))
  min_length = min(length_vec)
  min_index = which.min(length_vec)

  average_weight = function(len, wlistelem){
    wlistelem[1:len]
  }

  weights_vec = rowMeans(sapply(weights_list, average_weight, len=min_length))
  # select minimim weight length graph
  G = g[[min_length]]
  E(G)$weight = weights_vec

  min_weight = min(weights_vec)

  if(min_weight < 0){
    E(G)$weight = stan(E(G)$weight, 0, 1)
  }
  cg = cluster_optimal(G)
  plot(cg, edge.arrow.mode=0, G, main = main)
}

# A function to load a list of graphs among all chains
applyFunToGraphList = function(path, all_files, epoch){
  graph_lists = all_files[startsWith(all_files, epoch) & endsWith(all_files, ".gml")]
  graphDataList = lapply(graph_lists, function(x) read_graph(file = paste(path, x, sep = ''), format = "gml"))
  graphDataList
}

# A function to degree distribution
plot_degree = function(data, x){

 p = ggplot(data = filter(data, epoch == x), mapping = aes(x=degree)) +
    stat_density(bw=0.7, alpha = 0.4, color="red", fill="red") + ylab("") + xlab("") +
    theme_classic() + theme(legend.position = "none", text = element_text(size=30),
                            axis.text.x=element_text(colour="black"), axis.text.y=element_text(colour="black"))

  return(p)
}