# Title     : Various User defined function, Mainly needed for graph visualization
# Objective : Visualization
# Created by: mohsen
# Created on: 8/7/19

library(igraph)
library(svglite)

# Rescale between a and b
stan = function(x, a, b){
  (x - min(x))/(max(x) - min(x))*(b-a) + a
}

# A star layout graph visualization with high degree node (strength) in center
graphPlot = function(g, main, type){

  # Configuring Vertex size based on degree

  if(type == "strength"){

    deg <- graph.strength(g)
  } else{
    deg <- hub_score(g)$vector
  }
  V(g)$size <- stan(deg, 10, 30)

  # Set edge width based on weight
  E(g)$width <- stan(E(g)$weight, 1, 5)

  #change arrow size and edge color
  E(g)$arrow.size <- .4

  #Color scaling function
  #Using jet colour map
  c_scale <- colorRamp(c("#000066", "blue", "cyan",
                         "yellow", "red", "#660000"))
  #Applying the color scale to edge weights.
  #rgb method is to convert colors to a character vector.
  E(g)$color = apply(c_scale(stan(E(g)$weight, 0, 1)), 1, function(x) rgb(x[1]/255,x[2]/255,x[3]/255) )

  # Configuring layout
  coords <- layout_as_star(g, V(g)[which.max(V(g)$size)])

  plot(g, layout = coords, main = main)

}

# A cluster graph visualization based on cluster_optimal function from igraph package
clusterGraph = function(g, main){
  min_weight = min(E(g)$weight)

  if(min_weight < 0){
    E(g)$weight = stan(E(g)$weight, 0, 1)
  }
  cg = cluster_optimal(g)
  plot(cg, edge.arrow.mode=0, g, main = main)
}


