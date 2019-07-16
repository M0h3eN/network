rm(list=ls())

library(igraph)
library(svglite)
library(rsvg)

stan = function(x, a, b){
  (x - min(x))/(max(x) - min(x))*(b-a) + a
}

args = commandArgs(TRUE)
path = as.character(args[1])
if(!dir.exists(args[2])){
    dir.create(file.path(args[2]))
}
setwd(file.path(args[2]))

netEncStim = read_graph(file = paste(path, 'Enc-In-Stim.gml', sep=''), format = "gml")
netEncNoStim = read_graph(file = paste(path, 'Enc-In-NoStim.gml', sep=''), format = "gml")

netMemStim = read_graph(file = paste(path, 'Mem-In-Stim.gml', sep=''), format = "gml")
netMemNoStim = read_graph(file = paste(path, 'Mem-In-NoStim.gml', sep=''), format = "gml")

netSaccStim = read_graph(file = paste(path, 'Sac-In-Stim.gml', sep=''), format = "gml")
netSacNoStim = read_graph(file = paste(path, 'Sac-In-NoStim.gml', sep=''), format = "gml")

graphDataList = list(netEncStim, netMemStim, netSaccStim, netEncNoStim, netMemNoStim, netSacNoStim)

# createMixCentrality = function(g){
#
# pr_cent <- proper_centralities(g)
# cen = calculate_centralities(g, include = pr_cent)
# nam = attributes(cen)$names
#
# filterList = function(cen, nam){
#   outList = list()
#   for(x in 1:length(nam)){
#     dat = cen[nam[x]][[1]]
#     if ((!as.logical(sum(is.nan(dat))) && !is.null(dat) && !(var(dat) <= 0.01))){
#       outList[[nam[x]]] = stan(dat, 0, 1)
#     } else next
#   }
#   outList
# }
#
# cenList = as.data.frame(filterList(cen, nam))
# as.numeric(rowMeans(cenList))
#
# }

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

  # Set names to Nx
  V(g)$label <- sapply(c(1:length(V(g)$label)), function(x) paste("N",x,sep = ''))

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

graphList = c("Visual-Stim","Memory-Stim", "Saccade-Stim",
              "Visual-WithoutStim", "Memory-WithoutStim", "Saccade-WithoutStim")


# Strengh
svglite(file = "out.svg")
par(mfrow=c(2,3), mar=c(1,1,1,1))
for(g in 1:length(graphList)){
  graphPlot(graphDataList[[g]], graphList[g],"strength")
}
rsvg_svg("out.svg", "strength.svg")
file.remove(paste(args[2],'/','out','.svg', sep=''))
dev.off()

## Writing Graph individualy
for(g in 1:length(graphList)){
  svglite(file = paste(g, '.svg', sep=''))
  graphPlot(graphDataList[[g]], graphList[g],"strength")
  rsvg_svg(paste(g, '.svg', sep=''), paste('strength-',graphList[g], '.svg', sep=''))
  file.remove(paste(args[2],'/',g,'.svg', sep=''))
}
dev.off()

# hub
svglite(file = "out1.svg")
par(mfrow=c(2,3), mar=c(1,1,1,1))

for(g in 1:length(graphList)){
  graphPlot(graphDataList[[g]], graphList[g],"hub")
}
rsvg_svg("out1.svg", "hub.svg")
file.remove(paste(args[2],'/','out1','.svg', sep=''))
dev.off()

## Writing Graph individualy
for(g in 1:length(graphList)){
  svglite(file = paste(g, '.svg', sep=''))
  graphPlot(graphDataList[[g]], graphList[g],"hub")
  rsvg_svg(paste(g, '.svg', sep=''), paste('hub-',graphList[g], '.svg', sep=''))
  file.remove(paste(args[2],'/',g,'.svg', sep=''))
}
dev.off()

# info
svglite(file = "out2.svg")

# plot all graph
par(mfrow=c(2,3), mar=c(1,1,1,1))

for(g in 1:length(graphList)){
  graphPlot(graphDataList[[g]], graphList[g],"info")
}
rsvg_svg("out2.svg", "info.svg")
file.remove(paste(args[2],'/','out2','.svg', sep=''))
dev.off()

## Writing Graph individualy
for(g in 1:length(graphList)){
  svglite(file = paste(g, '.svg', sep=''))
  graphPlot(graphDataList[[g]], graphList[g],"info")
  rsvg_svg(paste(g, '.svg', sep=''), paste('info-',graphList[g], '.svg', sep=''))
  file.remove(paste(args[2],'/',g,'.svg', sep=''))
}
dev.off()

# cluster
svglite(file = "out3.svg")

par(mfrow=c(2,3), mar=c(1,1,1,1))
for(g in 1:length(graphList)){
  clusterGraph(graphDataList[[g]], graphList[g])
}
rsvg_svg("out3.svg", "cluster.svg")
file.remove(paste(args[2],'/','out3','.svg', sep=''))
dev.off()

## Writing Graph individualy
for(g in 1:length(graphList)){
  svglite(file = paste(g, '.svg', sep=''))
  clusterGraph(graphDataList[[g]], graphList[g])
  rsvg_svg(paste(g, '.svg', sep=''), paste('cluster-',graphList[g], '.svg', sep=''))
  file.remove(paste(args[2],'/',g,'.svg', sep=''))
}
dev.off()





