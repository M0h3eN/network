rm(list=ls())

library(igraph)


stan = function(x, a, b){
  (x - min(x))/(max(x) - min(x))*(b-a) + a
}


netEncStim = read_graph(file = "/home/mohsen/Rdir/GraphGml/Enc-In-Stim___1.gml", format = "gml")
netEncNoStim = read_graph(file = "/home/mohsen/Rdir/GraphGml/Enc-In-NoStim___1.gml", format = "gml")

netMemStim = read_graph(file = "/home/mohsen/Rdir/GraphGml/Mem-In-Stim___1.gml", format = "gml")
netMemNoStim = read_graph(file = "/home/mohsen/Rdir/GraphGml/Mem-In-NoStim___1.gml", format = "gml")

netSaccStim = read_graph(file = "/home/mohsen/Rdir/GraphGml/Sac-In-Stim___1.gml", format = "gml")
netSacNoStim = read_graph(file = "/home/mohsen/Rdir/GraphGml/Sac-In-NoStim___1.gml", format = "gml")

graphDataList = list(netEncStim, netMemStim, netSaccStim, netEncNoStim, netMemNoStim, netSacNoStim)


graphPlot = function(g, main, type){

  # Configuring Vertex size based on degree
  
  if(type == "strength"){
  
    deg <- graph.strength(g)
  } else if(type =="hub"){
    deg <- hub_score(g)$vector  
  } else{
    deg <- sna::infocent(dat = as.matrix(as_adjacency_matrix(g)))  
  }
  V(g)$size <- stan(deg, 10, 30)
  
  # Set edge width based on weight
  E(g)$width <- stan(E(g)$weight, 1, 5)
  
  #change arrow size and edge color
  E(g)$arrow.size <- .4
  
  #Color scaling function
  c_scale <- colorRamp(c('gray80', 'cyan', 'yellow', 'red'))
  #Applying the color scale to edge weights.
  #rgb method is to convert colors to a character vector.
  E(g)$color = apply(c_scale(stan(E(g)$weight, 0, 1)), 1, function(x) rgb(x[1]/255,x[2]/255,x[3]/255) )

  # Configuring layout 
  coords <- layout_as_star(g, V(g)[which.max(V(g)$size)])
  
  plot(g, layout = coords, main = main) 

}

clusterGraph = function(g, main){
  cg = cluster_optimal(g)  
  plot(cg, edge.arrow.mode=0, g, main = main)
}

graphList = c("Encoding-Stim","Memory-Stim", "Saccade-Stim", 
              "Encoding-WithoutStim", "Memory-WithoutStim", "Saccade-WithoutStim")

# plot all graph
par(mfrow=c(2,3), mar=c(1,1,1,1))

# Strengh
for(g in 1:length(graphList)){
  graphPlot(graphDataList[[g]], graphList[g],"info")
}
dev.off()

par(mfrow=c(2,3), mar=c(1,1,1,1))
for(g in 1:length(graphList)){
  clusterGraph(graphDataList[[g]], graphList[g])
}
dev.off()



