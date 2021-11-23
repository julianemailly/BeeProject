#### Geometry Functions

# Convert a polar vector into a cartesian vector
PolarToCartesian = function(vector)
{
  # Dependencies :
  # NA
  
  cartesianVector = vector[1] * c(cos(vector[2]),sin(vector[2]));
  
  return(cartesianVector)
}

# Normalize a matrix by row
NormalizeMatrix = function(matrix)
{
  # matrix : any matrix object to normalize by row
  # Dependencies :
  # NA
  
  for(line in 1:nrow(matrix))
  {
    lineSum = sum(matrix[line,]);
    for(col in 1:ncol(matrix))
    {
      matrix[line,col]=matrix[line,col]/lineSum;
    }
  }
  return(matrix)
}

# Get the distance between two points
DistanceBetween = function(a,b)
{
  # a, b : cartesian coordinates (x,y) of two points in a 2 dimension space
  # Dependencies : NA

  distance = sqrt((b[1]-a[1])**2 + (b[2]-a[2])**2);

  return(distance)
}

# Get the length of a route
RouteLength = function(arrayGeometry,route)
{
  # Dependencies :
  # NA
  
  distanceMatrix = as.matrix(dist(arrayGeometry[,2:3],upper = T,diag=T));
  
  numberOfVisits = length(route);
  distance = 0;
  for(visit in 1:(numberOfVisits-1))
  {
    distance = distance + distanceMatrix[route[visit],route[visit+1]];
  }
  return(distance)
}



#### Spatial Array Generation & Manipulation

# Places the nest and flowers in a spatial plane according to the inputs given.
CreateEnvironment = function(arrayInfos,arrayNumber,reuseGeneratedArrays)
{
  # This code is used if you generate the environment procedurally
  if(arrayInfos$environmentType == "generate")
  {
    # Create a name for the folder of the array
    arrayInfos$arrayID = CreateArrayID(arrayInfos,arrayNumber);
    
    # Retrieve the list of arrays available in the Arrays folder.
    listOfKnownArrays = list.files(paste(getwd(),"/Arrays",sep=""));
    arrayFolder = paste(getwd(),"/Arrays/",arrayInfos$arrayID,sep="");
    
    # If reuseGeneratedArrays is TRUE, look for similar arrays
    if(reuseGeneratedArrays)
    {
      similarArray = which(CharacterMatch(listOfKnownArrays,arrayInfos$arrayID));
      
      if(length(similarArray)!=0)
      {
        arrayGeometry = read.csv(paste(arrayFolder,"/arrayGeometry.csv",sep=""));
        
        if(!silentSim) cat("A similar array has been found: ",listOfKnownArrays[similarArray],". Importing.\n",sep="")
        
        return(list(arrayGeometry=arrayGeometry,arrayInfos=arrayInfos,arrayFolder=arrayFolder))
      }
    }
    
    # If reuseGeneratedArrays is FALSE, or there was no similar arrays, generate a new one
    # Adjust arrayID to make sure there is no overlap with an existing array.
    if(!reuseGeneratedArrays) 
    {
      arrayTypeID = substr(arrayInfos$arrayID,1,nchar(arrayInfos$arrayID)-(nchar(arrayNumber)+1))
      arrayFileNumber = GetIDNumberForArray(listOfKnownArrays,arrayTypeID);
      arrayInfos$arrayID = CreateArrayID(arrayInfos,arrayFileNumber);
      arrayFolder = paste(getwd(),"/Arrays/",arrayInfos$arrayID,sep="");
    }
    if(!silentSim) cat("Generating new array. Array ID is: ",arrayInfos$arrayID,".\n",sep="")
    
    dir.create(arrayFolder);
    arrayGeometry = GetPatchyArray(arrayInfos$numberOfResources,
                                   arrayInfos$numberOfPatches,
                                   arrayInfos$patchinessIndex,
                                   arrayInfos$envSize,
                                   arrayInfos$flowerPerPatch);
    
    
    
    # Write the array and parameters.
    arrayInfosSaved = arrayInfos;
    arrayInfosSaved$flowerPerPatch = paste(arrayInfosSaved$flowerPerPatch,collapse="");
    write.csv(as.data.frame(arrayInfosSaved),paste(arrayFolder,"/arrayInfos.csv",sep=""),row.names = F);
    write.csv(arrayGeometry,paste(arrayFolder,"/arrayGeometry.csv",sep=""),row.names = F);
    
    return(list(arrayGeometry=arrayGeometry,arrayInfos=arrayInfos,arrayFolder=arrayFolder))
  }
  
  # This code is used if you want to use a specific array
  if(arrayInfos$environmentType != "generate")
  {
    if(!silentSim) cat("Loading known array : ",arrayInfos$environmentType,"\n",sep="");
    arrayFolder = paste(getwd(),"/Arrays/",arrayInfos$environmentType,sep="");
    arrayGeometry = LoadArrayGeometry(paste(getwd(),"/Arrays/",arrayInfos$environmentType,"/arrayGeometry.csv",sep=""));
    arrayInfos = read.csv(paste(getwd(),"/Arrays/",arrayInfos$environmentType,"/arrayInfos.csv",sep=""),stringsAsFactors = F)
    arrayInfos$arrayID = arrayInfos$environmentType;
    
    return(list(arrayGeometry=arrayGeometry,arrayInfos=arrayInfos,arrayFolder=arrayFolder))
  }
}

# Sets the first line (nest) at coordinates (0,0) and adjust all other coordinates.
NormalizeArrayGeometry = function(arrayGeometry)
{
  # arrayGeometry : spatial positions (ID,x,y) of the flowers/nest
  # Dependencies : NA

  ## Set the first line (nest) at coordinates (0,0)
  arrayGeometry$x = arrayGeometry$x - arrayGeometry$x[1];
  arrayGeometry$y = arrayGeometry$y - arrayGeometry$y[1];
  
  return(arrayGeometry)
}

# Load an array geometry file and normalize it.
LoadArrayGeometry = function(FileName)
{
  # FileName : character, path to array csv file.
  # Dependencies :
  # NormalizeArrayGeometry
  
  return(NormalizeArrayGeometry(read.csv(FileName)))
}

# Create an unique name for an array
CreateArrayID = function(arrayInfos,arrayNumber)
{
  flowerPerPatchText = paste(arrayInfos$flowerPerPatch,collapse = "");
  if(arrayInfos$numberOfPatches==1) flowerPerPatchText = "NULL";
  arrayTypeID = paste("Array-",
                      arrayInfos$numberOfResources,"-",
                      arrayInfos$numberOfPatches,"-",
                      arrayInfos$patchinessIndex,"-",
                      flowerPerPatchText,"-",
                      arrayInfos$envSize,sep="")
  arrayID = paste(arrayTypeID,
                  sprintf("%02d",arrayNumber),sep="_");
  return(arrayID)
}

# Scan among known arrays to find a array ID not taken to give to the new array
GetIDNumberForArray = function(listOfKnownArrays,arrayTypeID)
{
  listOfSimilarArrays = listOfKnownArrays[CharacterMatch(listOfKnownArrays,arrayTypeID)];
  availableNumbers = 1:(length(listOfSimilarArrays)+1)
  if(length(listOfSimilarArrays)>0)
  {
    for(i in 1:length(listOfSimilarArrays))
    {
      scan = listOfSimilarArrays[i];
      scanLength = nchar(scan);
      for(chrNumber in scanLength:1)
      {
        chr = substr(scan,chrNumber,chrNumber);
        if(chr=="_")
        {
          scannedArrayNumber = as.numeric(substr(scan,chrNumber+1,scanLength))
          availableNumbers = setdiff(availableNumbers, scannedArrayNumber)
        }
      }
    }
  }
  return(min(availableNumbers))
}

# Generate an environment of flowers procedurally.
GetPatchyArray = function(resources,numberOfPatches,patchiness,envSize,flowerPerPatch=NULL,silent=F)
{
  # Dependencies :
  # PolarToCartesian
  # DistanceBetween 
  
  # resources : 
  
  # patchiness : float between 0 and 1, depicting the degree of patchiness.
  # patchiness = 1 : very patchy.
  # patchiness = 0 : nearly uniform distribution of flowers.
  
  # This function is an algorithm that generates environments with food sites distributed in it.
  # The basis of this algorithm is to generate an environment that can be challenging for a species, given its perception range.
  # Most of the distances between the flowers and patches are depending on the perceptionRange and the envSize.
  # Choose them wisely to create a relevant environment for your species.
  
  perceptionRange = 10; # Used to be a parameter, but no longer used. As it is needed for this function, it is set as a constant here. Any positive value is fine.
  
  if(patchiness>1 || patchiness<0) stop("Unsupported Patchiness Index. Values can range between 0 and 1.");
  if(numberOfPatches > resources) stop("You must have at least one resources per patch.");
  if(any(c(resources,numberOfPatches) == 0)) stop("You must at least have one resource and one patch.");
  if(4*perceptionRange>=envSize) stop("The environment size must be at least superior to 4 times the perceptionRange.");
  
  
  # Interpatch parameters
  distMinToNest = 2*perceptionRange;
  distMaxToNest = envSize;
  distNestIncrement = envSize/20;
  distMinPatchPatch = 16*perceptionRange;
  
  # Initialize output matrices
  patchCenters = matrix(0,nrow = numberOfPatches,ncol = 2);
  arrayGeometry = data.frame(ID = c(0:resources),x = 0,y = 0,patch = 1);
  
  # We create a first patch center
  patch1dist = runif(1,distMinToNest,distMaxToNest);
  patch1azi = runif(1,0,2*pi);
  patchCenters[1,] = PolarToCartesian(c(patch1dist,patch1azi));
  patchOk = 1;
  
  while(patchOk < numberOfPatches)
  {
    patchToDo = patchOk + 1;
    
    # Creating a patch center
    distanceNestPatches = runif(1,distMinToNest,distMaxToNest);
    azimuthNestPatches = runif(1,0,2*pi);
    polarVector = c(distanceNestPatches,azimuthNestPatches);
    patchCenters[patchToDo,] = PolarToCartesian(polarVector);
    
    distPatchPatch = rep(0,patchOk);
    
    # Checking distance with other patches
    for(patch in 1:patchOk)
    {
      distPatchPatch[patch] = DistanceBetween(patchCenters[patch,],patchCenters[patchToDo,]);
    }
    # If the new patch center does not respect the rule, remove it and start again
    if(any(distPatchPatch < distMinPatchPatch)) next else patchOk = patchOk + 1;
  }
  
  # We now have patch centers that follow our rules (distances between patches and to the nest)
  # Now, we want to "populate" these patches. First, we divide the resources between them
  
  # We retrieve the remaining number of flowers to put in the array. 
  # We set a poisson distribution to distribute them, with a lambda arbitrarily set to the mean number of flowers per patch.
  if(is.null(flowerPerPatch))
  {
    population = resources - numberOfPatches;
    lambdaPop = population/numberOfPatches;
    
    patchPop = rep(0,numberOfPatches);
    # We sample the distribution of the flowers among the patches until we get a distribution fitting the pop size.
    while(sum(patchPop) != population)
    {
      patchPop = sample(c(1:population),numberOfPatches,replace=T,prob = dpois(1:population,lambdaPop));
    }
  } else {
    if(length(flowerPerPatch)!=numberOfPatches) stop("Vector length of flowerPerPatch is different than the numberOfPatches.")
    patchPop = flowerPerPatch-1;
  }
  
  
  # We now know how many flowers to put at each patch;
  i = 1;
  for(patch in 1:numberOfPatches)
  {
    # Intrapatch parameters
    distMinToPatch = 2*perceptionRange; # High Patchiness = low min distance to patch center (= Closer flowers in the patch)
    distMaxToPatch = envSize - (envSize-4*perceptionRange)*patchiness; # High Patchiness = low max distance to patch center (= Closer flowers in the patch)
    
    patchCenter = patchCenters[patch,];
    if(patch==1) i = i + 1;
    if(patch>1) i = i + 1 + patchPop[patch-1];
    arrayGeometry[i,2:3] = patchCenter;
    arrayGeometry[i,4] = patch+1;
    
    if(patchPop[patch] > 0)
    {
      flowerOk = 0;
      failedPositioning = 0;
      while(flowerOk < patchPop[patch])
      {
        
        distancePatchFlower = runif(1,distMinToPatch,distMaxToPatch);
        azimuthPatchFlower = runif(1,0,2*pi);
        polarVector = c(distancePatchFlower,azimuthPatchFlower);
        
        cartesianVector = PolarToCartesian(polarVector);
        arrayGeometry[i+flowerOk+1,2:3] = patchCenter + cartesianVector;
        arrayGeometry[i+flowerOk+1,4] = patch+1;
        
        distToFlowers = c()
        for(flower in 1:(i+flowerOk))
        {
          distToFlowers[flower] = DistanceBetween(as.numeric(arrayGeometry[i+1+flowerOk,2:3]),as.numeric(arrayGeometry[flower,2:3]))
        }
        
        if(any(distToFlowers < distMinToPatch)) 
        {
          failedPositioning = failedPositioning + 1;
          if(failedPositioning > 200) 
          {
            cat("GetPatchyArray failed to position a flower in the last 200 iterations. Increasing range of possible positions by ",distNestIncrement,"m.\n",sep="");
            distMaxToPatch = distMaxToPatch + distNestIncrement;
            failedPositioning = 0;
          }
          next
        } else {
          flowerOk = flowerOk + 1;
        }
      }
    }
  }
  return(arrayGeometry)
}



#### Construction of probability matrix and optimal route assessment

# Gets the probability of a vector happening given the function [probability = 1/x^distFactor]
DistanceFunction = function(x,distFactor)
{
  return(y = 1/(x^distFactor))
}

# Generates the probability matrix for a given array
GetDistFctProbability = function(arrayGeometry,distFactor,beeData)
{
  
  distances = dist(arrayGeometry[,2:3],upper=T);
  probabilities = as.matrix(DistanceFunction(distances,distFactor));
  
  initialMatrix = NormalizeMatrix(probabilities);
  
  initialMatrixList = list();
  
  for(bee in 1:length(beeData[,1])) initialMatrixList[[bee]] = initialMatrix;
  
  return(initialMatrixList)
}

# Attempts to retrieve an existing optimal route file for the array tested. Otherwise, calls OptimalRouteAssessment.
SimDetectionOptimalRoute = function(array,arrayGeometry,beeData,probabilityMatrix,outputFolder,silent=F,optimalRouteQuality)
{
  # Dependencies :
  # OptimalRouteAssessment
  cat("Checking on arrayFolder : ",outputFolder,"\n",sep="")
  knownFiles = list.files(path=outputFolder); numberOfFiles = length(knownFiles);
  
  if(numberOfFiles>0 & any(knownFiles == "optimalRoute.csv"))
  {
    cat("There is a optimalRoute.csv file available for this array.\n")
    
    dataFile = read.csv2(paste(outputFolder,"/optimalRoute.csv", sep=""),header = T);
    optimalRoute = as.numeric(dataFile);
    
    return(optimalRoute)
  }
  
  # If not found in the memory, assess the optimal route quality
  optimalRoute = OptimalRouteAssessment(array,beeData,probabilityMatrix,arrayGeometry,silent=F,optimalRouteQuality);
  
  # Register it in a file
  write.csv(optimalRoute,paste(outputFolder,"/optimalRoute.csv",sep=""),row.names = F);
  return(optimalRoute)
}

# Makes an assessment of the optimal route quality of an array by releasing 100 bees for 30 bout each. Retrieves the best route quality.
OptimalRouteAssessment = function(array,beeData,initialProbabilityMatrix,arrayGeometry,silent=F,optimalRouteQuality)
{
  # Dependencies :
  # LoadArrayGeometry
  # GetWinProbabilities
  # InitializeBeeData
  # RebootBeeData
  # CompetitiveRoute
  
  if(!silent) cat("No data on optimal route found. Assessing optimal route with simulations.\n");
  
  numberOfSimulations = 100;
  numberOfBouts = 30;
  numberOfBees = 1;
  
  listOfVisitationSequences = list();
  matrixOfBeeData = matrix(0,ncol=5,nrow=numberOfSimulations*numberOfBouts*numberOfBees);
  
  matrixOfBeeData[,1]=rep(1:numberOfSimulations,each=numberOfBouts*numberOfBees);
  matrixOfBeeData[,2]=rep(1:numberOfBouts,times=numberOfSimulations,each=numberOfBees);
  matrixOfBeeData[,3]=rep(1:numberOfBees,times=numberOfSimulations*numberOfBouts);
  
  bestQualityOfSim = matrix(0,nrow=1,ncol=numberOfSimulations);
  bestRouteSim = list();
  
  i=1;
  for(sim in 1:numberOfSimulations)
  {

    probabilityArray = initialProbabilityMatrix;
    
    
    for(bout in 1:numberOfBouts)
    {
      
      beeData = RebootBeeData(beeData);
      
      
      currentBout = CompetitiveRoute(bout,arrayGeometry,probabilityArray,beeData[1,],optimalRouteQuality,silentSim=T,useQLearning=FALSE);
      
      ## Update variables
      probabilityArray = currentBout$Learning;
      beeData = currentBout$beeData;
      optimalRouteQuality = currentBout$optimalRouteQuality
      
      matrixOfBeeData[i,4] = beeData$numberOfResources[1];
      matrixOfBeeData[i,5] = currentBout$quality[1];
      
      
      listOfVisitationSequences[[(sim-1)*numberOfBouts+bout]] = currentBout$Sequences;
      
      i = i + numberOfBees;
      
    }
    qualitiesOfSim = matrixOfBeeData[which(matrixOfBeeData[,1]==sim),5];
    bestQualityOfSim[,sim] = max(qualitiesOfSim);
    boutOfBestQuality = which(qualitiesOfSim == bestQualityOfSim[,sim]);
    bestRouteSim[[sim]] = cbind(listOfVisitationSequences[[(sim-1)*numberOfBouts+boutOfBestQuality[1]]],1);
  }
  
  # Assess the max quality attained in each sim
  
  simOptRouteQuality = max(bestQualityOfSim);
  
  simReachOpti = which(bestQualityOfSim==simOptRouteQuality);
  allBestRoutes = unique(bestRouteSim[simReachOpti]);
  
  simRouteQualityCount = count(as.numeric(bestQualityOfSim));
  
  proportionOfOptRoute = simRouteQualityCount[which(simRouteQualityCount$x==simOptRouteQuality),2]/numberOfSimulations*100;
  
  if(!silent) cat("Out of ",numberOfSimulations," simulations in ",array,", ",proportionOfOptRoute,"% reached the maximum quality of ",simOptRouteQuality,". Setting this value as optimal route quality.\n",sep="")
  if(!silent) cat("A total of ",length(allBestRoutes)," routes had this quality. They were the following :\n",sep="");
  if(!silent) print(allBestRoutes);
  return(simOptRouteQuality)
}


###For 2 individuals
# Attempts to retrieve an existing optimal route (2ind) file for the array tested. Otherwise, calls OptimalRouteAssessment2Ind.
SimDetectionOptimalRoute2Ind = function(array,arrayGeometry,beeData,probabilityMatrix,outputFolder,silent=F,optimalRouteQuality)
{
  # Dependencies :
  # OptimalRouteAssessment
  cat("Checking on arrayFolder : ",outputFolder,"\n",sep="")
  knownFiles = list.files(path=outputFolder); numberOfFiles = length(knownFiles);
  
  if(numberOfFiles>0 & any(knownFiles == "optimalRoute2Ind.csv"))
  {
    cat("There is a optimalRoute2Ind.csv file available for this array.\n")
    
    dataFile = read.csv2(paste(outputFolder,"/optimalRoute2Ind.csv", sep=""),header = T);
    optimalRoute = as.numeric(dataFile);
    
    return(optimalRoute)
  }
  
  # If not found in the memory, assess the optimal route quality
  optimalRoute = OptimalRouteAssessment2Ind(array,beeData,probabilityMatrix,arrayGeometry,silent=F,optimalRouteQuality);
  
  # Register it in a file
  write.csv(optimalRoute,paste(outputFolder,"/optimalRoute2Ind.csv",sep=""),row.names = F);
  return(optimalRoute)
}

# Makes an assessment of the optimal route quality of an array by releasing 100 bees for 30 bout each. Retrieves the best route quality.
OptimalRouteAssessment2Ind = function(array,beeData,initialProbabilityMatrix,arrayGeometry,silent=F,optimalRouteQuality)
{
  # Dependencies :
  # LoadArrayGeometry
  # GetWinProbabilities
  # InitializeBeeData
  # RebootBeeData
  # CompetitiveRoute
  
  if(!silent) cat("No data on optimal route for 2 individuals found. Assessing optimal route with simulations.\n");
  
  numberOfSimulations = 100; 
  numberOfBouts = 30;
  numberOfBees = 2;
  
  listOfVisitationSequences = list();
  listOfResourcesTaken = list();
  matrixOfBeeData = matrix(0,ncol=5,nrow=numberOfSimulations*numberOfBouts*numberOfBees);
  
  matrixOfBeeData[,1]=rep(1:numberOfSimulations,each=numberOfBouts*numberOfBees);
  matrixOfBeeData[,2]=rep(1:numberOfBouts,times=numberOfSimulations,each=numberOfBees);
  matrixOfBeeData[,3]=rep(1:numberOfBees,times=numberOfSimulations*numberOfBouts);
  
  bestQualityOfSim = matrix(0,nrow=1,ncol=numberOfSimulations);
  bestRouteSim = list();
  
  i=1;
  j=1;
  for(sim in 1:numberOfSimulations)
  {
    probabilityArray = initialProbabilityMatrix;
    
    
    for(bout in 1:numberOfBouts)
    {
      beeData = RebootBeeData(beeData);
      
      
      currentBout = CompetitiveRoute(bout,arrayGeometry,probabilityArray,beeData,optimalRouteQuality,silentSim=T,useQLearning=FALSE);
      
      ## Update variables
      probabilityArray = currentBout$Learning;
      beeData = currentBout$beeData;
      optimalRouteQuality = currentBout$optimalRouteQuality
      
      matrixOfBeeData[i:(i+numberOfBees-1),4] = beeData$numberOfResources;
      matrixOfBeeData[i:(i+numberOfBees-1),5] = currentBout$quality[1]+currentBout$quality[2]; #there are gonna be duplications of the same number but doesn't matter 
      
      
      listOfVisitationSequences[[(sim-1)*numberOfBouts+bout]] = currentBout$Sequences;
      
      i = i + numberOfBees;
      listOfResourcesTaken[[j]] = currentBout$listOfBoutResources;
      j = j + 1;
      
    }
    qualitiesOfSim = matrixOfBeeData[which(matrixOfBeeData[,1]==sim),5];
    bestQualityOfSim[,sim] = max(qualitiesOfSim);
    boutOfBestQuality = ceiling((which(qualitiesOfSim == bestQualityOfSim[,sim]))/numberOfBees); #need to ccount for the repetitions due to the nb of bees
    bestRouteSim[[sim]] = cbind(listOfVisitationSequences[[(sim-1)*numberOfBouts+boutOfBestQuality[1]]],1);
  }
  
  # Assess the max quality attained in each sim
  
  simOptRouteQuality = max(bestQualityOfSim);
  
  simReachOpti = which(bestQualityOfSim==simOptRouteQuality);
  allBestRoutes = unique(bestRouteSim[simReachOpti]);
  
  simRouteQualityCount = count(as.numeric(bestQualityOfSim));
  
  proportionOfOptRoute = simRouteQualityCount[which(simRouteQualityCount$x==simOptRouteQuality),2]/numberOfSimulations*100;
  
  if(!silent) cat("Out of ",numberOfSimulations," simulations in ",array,", ",proportionOfOptRoute,"% reached the maximum quality of ",simOptRouteQuality,". Setting this value as optimal route quality.\n",sep="")
  if(!silent) cat("A total of ",length(allBestRoutes)," routes had this quality. They were the following :\n",sep="");
  if(!silent) print(allBestRoutes);
  return(simOptRouteQuality)
}









#### Management of data

# Sets up a data frame including all important informations on the foraging bees.
InitializeBeeData = function(numberOfBees,paramTracking,paramIndiv)
{
  # Dependencies :
  # NA
  
  beeData = data.frame(ID=seq(1:numberOfBees),
                       paramTracking,
                       paramIndiv);
  
  return(beeData)
}

# Sets up a data frame of parameters of bees.
BuildBeeInfos = function(numberOfBees,paramIndiv,arrayID)
{
  beeInfos = data.frame(ID = seq(1:numberOfBees),
                        paramIndiv,
                        arrayID);
  
  # Remove the bestRouteQuality;
  beeInfos = subset(beeInfos,select = -c(bestRouteQuality));
  
  return(beeInfos)
}

# Resets the parameters of beeData between bouts.
RebootBeeData = function(beeData)
{
  # Dependencies :
  # NA
  
  beeData$numberOfResources=0;
  beeData$boutFinished=F;
  beeData$distanceTravelled=0;
  return(beeData);
}



#### Learning

# Route quality evaluation function  
GetQuality = function(numberOfResources,routeLength)
{
  return((numberOfResources^2)/routeLength)
}

# Computes the route quality of a given visitation sequence
GetRouteQuality = function(visitationSequence,arrayGeometry,resourcesForaged)
{
  # Dependencies :
  # RouteLength
  # GetQuality
  
  routeLength = 0;
  numberOfResources = sum(resourcesForaged);
  
  routeLength = RouteLength(arrayGeometry,visitationSequence);
  if(routeLength != 0) routeQuality = GetQuality(numberOfResources,routeLength);
  if(routeLength == 0) routeQuality = 0;
  return(routeQuality);
}

# Change the probability matrix of a individual depending on its last performed bout.
ApplyLearning = function(route,probabilityMatrix,flowerOutcomeMatrix,beeData,minProbVisit,routeQuality)
{
  # Dependencies :
  # NA
  
  # Vector of the route used by the bee
  # probabilityMatrix : matrix of array size * array size, with probabilities to do each vector linking two flowers
  # flowerOutcomeMatrix : matrix of array size * array size, with values of -1 (negative outcome), 0 (no visit) or 1 (positive outcome).
  # learningFactor : value added to a probability with positive outcome.
  # abandonFactor : value substracted to a probability with negative outcome.
  # minProbVisit : Since no prob in the matrix can be negative, we set all negative values (after applying abandonFactor) to this value.
  # routeCompare : boolean. TRUE : Each new route is compared to the best known. If new is of lesser quality, no learning. FALSE : No route compare. All vectors used are subject to learning if they reward the agent.
  # bestRouteQuality : numeric value of best route quality experienced so far. Used in routeCompare situations.
  # routeQuality : numeric value of last route quality experienced. used in routeCompare situations.
  # numberOfResources : numeric containing the number of resources gathered by the bee during the bout
  
  if(any(dim(probabilityMatrix) != dim(flowerOutcomeMatrix))) stop("Probability and FlowerOutcome matrices of different lengths!");
  
  # Apply learning and abandon factor to flower outcomes;
  
  if(!beeData$routeCompare)
  {
    # No route comparison, probe all values in the flowerOutcomeMatrix to input the changes in probabilities.
    
    newProbMatrix = matrix(1,nrow = length(flowerOutcomeMatrix[,1]),ncol = length(flowerOutcomeMatrix[1,]));
    
    newProbMatrix[which(flowerOutcomeMatrix==1)] = beeData$learningFactor;
    newProbMatrix[which(flowerOutcomeMatrix==-1)] = beeData$abandonFactor;
    
    # Add both matrices
    probabilityMatrix = probabilityMatrix * newProbMatrix;
  }
  
  if(beeData$routeCompare) 
  {
    newProbMatrix = matrix(1,nrow = length(flowerOutcomeMatrix[,1]),ncol = length(flowerOutcomeMatrix[1,]));
    
    # Check if new route is better or equal
    if(routeQuality >= beeData$bestRouteQuality & beeData$numberOfResources==beeData$cropCapacity & beeData$bestRouteQuality>0) ## /!\ Assuming crop capacity of 5.
    {
      # Add all vectors of the route as a positive outcome
      for(visit in 1:(length(route)-1))
      {
        from = route[visit];
        to = route[visit+1];
        newProbMatrix[from,to] = beeData$learningFactor;
      }
      
    }
    # Indifferent to the route comparison, apply the abandon.
    newProbMatrix[which(flowerOutcomeMatrix==-1)] = newProbMatrix[which(flowerOutcomeMatrix==-1)] * beeData$abandonFactor;
    
    probabilityMatrix = probabilityMatrix * newProbMatrix;
  }
  
  
  # Change all negative values to the minimum prob of visit
  probabilityMatrix[which(probabilityMatrix<0)] = minProbVisit;
  
  # Set all vectors back to the nest at 0;
  if(!beeData$allowNestReturn) probabilityMatrix[,1] = 0;
  
  # Normalize the matrix
  probabilityMatrix = NormalizeMatrix(probabilityMatrix);
  
  return(probabilityMatrix);
}



#### Other

# Function that looks into a character object for a specific match with another set of characters.
CharacterMatch = function(chrObject,chrSearched)
{
  # chrObject : a character object we want to scan (Chr or vector of Chr)
  # chrSearched : the character searched
  
  # The function returns a vector with TRUE if chrSearched is found in a chrObject. Otherwise, it returns FALSE.
  
  # Dependencies : 
  # NA
  
  if(length(chrObject)==0)
  {
    return(FALSE)
  }
  
  outputBool = logical(length(chrObject));
  i = 0;
  for(chrItem in chrObject)
  {
    i = i + 1;
    objectLength = nchar(chrItem);
    matchLength = nchar(chrSearched);
    numberOfScans = objectLength - matchLength + 1;
    
    matchPosition = c();
    isMatched = F;
    
    for(scan in 1:numberOfScans)
    {
      objScan = substr(chrItem,start = scan,stop = scan+matchLength-1);
      
      if(objScan == chrSearched)
      {
        outputBool[i] = TRUE;
        isMatched=T;
        break;
      }
    }
    if(!isMatched) outputBool[i] = FALSE;
  }
  return(outputBool)
}

# Function that also looks for a subset of characters "chrSearched" in a character object "chrObject", and returns the position of this "chrSearched" subset in the object.
FindCharacterPosition = function(chrObject,chrSearched,firstOnly=T)
{
  # chrObject : a character object we want to scan
  # chrSearched : the character searched
  # firstOnly : If T, stops at the first matching character found.
  
  # The function returns the positions (if chrSearched is of size > 1, the position of the first character of the match) of the match. If firstOnly is F, all potential match positions are returned in a vector.
  # If no match is found, return NULL.
  
  if(length(chrObject)==0)
  {
    cat("No character object to scan. Returning NULL")
    return(NULL)
  }
  if(nchar(chrObject) < nchar(chrSearched))
  {
    cat("Length of desired match superior to object size. Returning NULL")
    return(NULL)
  }
  
  objectLength = nchar(chrObject);
  matchLength = nchar(chrSearched);
  numberOfScans = objectLength - matchLength + 1;
  
  matchPosition = c();
  
  for(scan in 1:numberOfScans)
  {
    
    objScan = substr(chrObject,start = scan,stop = scan+matchLength-1);
    
    if(objScan == chrSearched)
    {
      matchPosition = c(matchPosition, scan);
      if(firstOnly) break;
    }
  }
  return(matchPosition)
}

# Create a video output for a given simulation. Requires FFMPEG (Also, untested on other OS than Windows 10, might bug)
CreateVideoOutput = function(arrayGeometry,matrixOfVisitationSequences,listOfResourcesTaken,beeData,framerate,simulationsToPrint,outputFolderTracking)
{
  # beeData : 
  
  numberOfBees = nrow(beeData);
  numberOfBouts = max(matrixOfVisitationSequences[,2])
  
  # Retrieve the limits of the 2D space
  xMin = min(arrayGeometry$x); xMax = max(arrayGeometry$x); xRange = xMax-xMin;
  yMin = min(arrayGeometry$y); yMax = max(arrayGeometry$y); yRange = yMax-yMin;
  
  # Setup an array of different colors we can use for identification of the individuals
  correctColors = c("burlywood","darkolivegreen3","coral2","aquamarine","chartreuse1","deepskyblue2","cyan1","darkorchid4","coral","blue","firebrick3","cornsilk","darkorange","deeppink","mediumpurple","paleturquoise");
  indColors = correctColors[1:numberOfBees];
  
  # Define position for ind pin (small point to indicate the flower the bee is on at each step)
  indPins = c(-xRange*0.05,yRange*0.05);
  
  # Create the png files for all sims
  for(sim in simulationsToPrint)
  {
    # Retrieve the sim movements
    simSpatialMovements = subset(matrixOfVisitationSequences,matrixOfVisitationSequences[,1]==sim);
    
    cat("Printing Simulation : ",sim,".\n",sep="");
    
    # Create the output directory for each simulation
    outputDirectory = paste(outputFolderTracking,"/Simulation",sim,sep="");
    dir.create(outputDirectory,showWarnings = F);
    
    i = 0;
    for(bout in 1:numberOfBouts)
    {
      # Retrieve the bout movements
      boutSpatialMovements = subset(simSpatialMovements,simSpatialMovements[,2]==bout);
      sequences = boutSpatialMovements[,-c(1:3)];
      indSequence = list();
      numberOfMoves = 0;
      
      # Get which bee gets resources on each flower
      whoFeedsBout = listOfResourcesTaken[[(sim-1)*numberOfBouts+bout]];
      whoFeedsBout[[length(whoFeedsBout)+1]] = matrix(0,nrow=1,ncol=numberOfBees);
      
      # Extract sequences and max number of flower visits
      for(ind in 1:numberOfBees)
      {
        if(numberOfBees == 1) indSequence[[ind]] = sequences[which(sequences!=0)];
        if(numberOfBees > 1) indSequence[[ind]] = sequences[ind,which(sequences[ind,]!=0)];
        numberOfMoves = max(numberOfMoves,length(indSequence[[ind]]));
      }
      
      # Initialize the resource availability
      resourceAvailable = rep(1,nrow(arrayGeometry)-1);
      
      
      # Print the graph
      plot(arrayGeometry[,2:3],type="n",cex=2,asp=1,
           xlim=c(xMin-xRange/10,xMax+xRange/10),
           ylim=c(yMin-yRange/10,yMax+yRange/10),
           xlab="X (m)",ylab="Y (m)",
           main=paste("Bout :",bout));
      # Setup the nest
      points(x=arrayGeometry[1,2],y=arrayGeometry[1,3],pch=23,cex=2);
      # Setup the flowers
      points(x=arrayGeometry[-1,2],y=arrayGeometry[-1,3],pch=22,cex=2);
      
      # Loop of movement drawing
      for(move in 1:(numberOfMoves-1))
      {
        i = i + 1;
        
        from = move;
        to = move + 1;
        
        for(ind in 1:numberOfBees)
        {
          if(to<=length(indSequence[[ind]])) 
          {
            # Draw the next movement
            fromPos = indSequence[[ind]][from];
            toPos = indSequence[[ind]][to];
            lines(arrayGeometry[c(fromPos,toPos),2],
                  arrayGeometry[c(fromPos,toPos),3],
                  col=indColors[ind],
                  lwd=2);
            
            
            
            # Check if the individual found a flower
            if(all(toPos!=1 & resourceAvailable[toPos-1]==1)) 
            {
              if(whoFeedsBout[[move]][ind]==1)
              {
                points(x=arrayGeometry[toPos,2],y=arrayGeometry[toPos,3],pch=22,cex=2,bg=indColors[ind]);
                resourceAvailable[toPos-1]=0;
              }
            }
          }
        }
        
        # Print the png file
        dev.copy(png,paste(outputDirectory,sprintf("/plot%03d.png",i),sep=""));
        allPos=NULL;
        for(ind in 1:numberOfBees)
        {
          # Place a pin at individual positions
          toPos = indSequence[[ind]][to];
          if(is.na(toPos)) toPos = 1;
          
          points(x=arrayGeometry[toPos,2:3]+indPins,pch=21,bg="grey",cex=3.5,col="red");
          text(x=arrayGeometry[toPos,2:3]+indPins,labels=as.character(ind),col=indColors[ind],cex=1.5);
          
          allPos = c(allPos,toPos);
        }
        dev.off()
      }
    }
    
    # Merging PNG Files into video
    quality = 15;
    
    knownFiles = list.files(path=outputDirectory); numberOfFiles = length(knownFiles); 
    
    system(paste("ffmpeg -r ",framerate," -f image2 -s 800x800 -i ", outputDirectory,"/plot%03d.png -vcodec libx264 -crf ", quality, " -pix_fmt yuv420p ", outputDirectory,"/Sim",sim,"plot.mp4",sep=""),
           show.output.on.console = F);
    
    # Remove PNG files
    for(file in knownFiles)
    {
      file.remove(paste(outputDirectory,file,sep="/"))
    }
  }
}

# Extra functions for the Q-Learning algorithm:

SoftMax=function(valuesVector,betaQL){
  return(exp(betaQL*valuesVector)/sum(exp(betaQL*valuesVector)))
}

ApplyOnlineQLearning=function(QTable,state,action,reward,alphaPos,alphaNeg,gammaQL){
  #bee with Q Table (QTable), is in state (state), does action (action), gets a reward (reward)
  #apply Q Learning algorithm: QTable[state,action]=QTable[state,action]+alpha*(reward+gamma*max(Q(nextState,b)/b in actions)-Q[state,action])
  #alpha: learning rate, gamma: temporal discount factor
  #here, action also corresponds to the next state since completely deterministic environment so max(Q(nextState,b)/b in actions)=max(Q(action,b)/b in actions)
  #two RL systems: one that reinforces positively some values (use alphaPos), one that reinforces negatively some values (use alphaNeg)
  delta=reward+gammaQL*max(QTable[action,])-QTable[state,action];
  if (delta>=0) {
    QTable[state,action]=QTable[state,action]+alphaPos*delta;
  }else{
    QTable[state,action]=QTable[state,action]+alphaNeg*delta;
  }
  return(QTable)
}

InitializeQTableList=function(initializeQTable,arrayGeometry,distFactor,beeData){
  if(initializeQTable=="distance"){
    return(GetDistFctProbability(arrayGeometry,distFactor,beeData))
  }else if (initializeQTable=="zero"){
    nStates=nrow(arrayGeometry); #TO CHECK
    nBees=length(beeData[,1]);
    initialMatrix = matrix(0,nrow=nStates,ncol=nStates);
    initialMatrixList = list();
    for (bee in (1:nBees)){
      initialMatrixList[[bee]] = initialMatrix
    };
    return(initialMatrixList)
  } else if (initializeQTable=="noisydist"){
    Q=GetDistFctProbability(arrayGeometry,distFactor,beeData);
    n=nrow(Q[[1]]);
    for (ind in (1:length(Q)) ){
      Q[[ind]]=Q[[ind]]+0.5*rnorm(n*n)
    }
    return(Q)
  }
}