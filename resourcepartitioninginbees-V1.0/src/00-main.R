# Vector-based foraging model. 
# Thibaud Dubois et al., 2020 April

## see HOWTO.txt for usage

# 1 - Libraries & Sourcing ----------------------------------------------------------------

# Packages to load beforehand.
library(plyr) # count() 
## library(MASS) # kde2d()

# Clean environment
rm(list=ls());

# Source function files
source("01-Functions.R");

# Source parameters
source("02-Parameters.R");


# Main Function  -----------------------------------------------------------

CompetitiveRoute = function(bout,arrayGeometry,learningArray,beeData,optimalRouteQuality,silentSim,useQLearning)
{
  # beeRoute will contain the routes of all bees.
  # resourcesOnFlowers will contain the number of resources available on each flower.
  # flowerOutcome will attribute for each ind a value at each vector depending on their positive (1), neutral (0) or negative (-1) experience.
  # probabilityArray becomes learningArray. if !useQLearning, learningArray=probabilityArray, else learningArray=QTableList
  
  betaQL=beeData$beta[1];
  alphaPos=beeData$alphaPos[1];
  alphaNeg=beeData$alphaNeg[1];
  gammaQL=beeData$gamma[1];
  
  numberOfBees = nrow(beeData);
  beeRoute = matrix(1,nrow=numberOfBees,ncol=1);
  resourcesOnFlowers = matrix(c(0,rep(1,times=length(arrayGeometry[,1])-1)),
                              ncol=length(arrayGeometry[,1]),nrow=1);
  flowerOutcome = matrix(0,ncol=length(arrayGeometry[,1]),nrow=length(arrayGeometry[,1]));
  indFlowerOutcome = list(); for(bee in 1:numberOfBees) indFlowerOutcome[[bee]] = flowerOutcome;
  listOfBoutResources = list();
  allVisitsIndex = 0;
  if(leaveAfterMaxFail) arrayOfVectorUse = array(0,dim=c(length(arrayGeometry[,1]),length(arrayGeometry[,1]),numberOfBees));
  
  # differentExperienceSimulation check : if bee's startingBoutForNaive is not reached for said ind, finish its bout instantly.
  for(ind in 1:numberOfBees)
  {
    if(beeData$differentExperienceSimulation[ind])
    {
      if(bout < beeData$startingBoutForNaive[ind])
      {
        beeData$boutFinished[ind] = TRUE;
      }
    }
  }
  
  
  #### Foraging Loop : While at least one individual hasn't finished foraging
  while(any(!beeData$boutFinished)) 
  {
    
    allVisitsIndex = allVisitsIndex + 1;
    
    # We check which bee is still foraging
    stillForaging = which(beeData$boutFinished==F);
    
    
    # We add another slot for the visitation sequence
    beeRoute = cbind(beeRoute,0);
    visitNumber = dim(beeRoute)[2];
    
    
    # Sampling next destination for each individual still foraging
    for(ind in stillForaging) 
    {
      # Retrieve the bee's current position
      indPos = beeRoute[ind,visitNumber-1];
      previousPos = beeRoute[ind,visitNumber-2];
      
      
      # Mark all destinations which must be omitted
      omittedDestinations = c(indPos); # At least the self-loop is forbidden.
      if(!beeData$allowNestReturn[ind]) omittedDestinations = c(omittedDestinations,1); # If condition, omit the nest.
      if(beeData$forbidReverseVector[ind]) omittedDestinations = c(omittedDestinations,previousPos) # If condition, omit the previous visit.
      if(beeData$leaveAfterMaxFail[ind]) omittedDestinations = c(omittedDestinations,which(arrayOfVectorUse[indPos,,ind]==beeData$maxFail[ind]));
      omittedDestinations = unique(omittedDestinations); # Make sure each flower is mentioned only once.
      
      
      # Retrieve all potential destinations
      potentialDestinations = (arrayGeometry$ID+1)[-omittedDestinations];
      # Choose a destination : If no positive probabilities, go back to the nest (not triggering end of bout)
      if (!useQLearning) {probabilities=learningArray[[ind]][indPos,-omittedDestinations]
      } else {
        if (dynamicBeta) {
          probabilities=SoftMax(learningArray[[ind]][indPos,-omittedDestinations],betaQLVector[bout])
        }else{
          probabilities=SoftMax(learningArray[[ind]][indPos,-omittedDestinations],betaQL) 
        }
      };
      if(any(probabilities!=0))
      {
        
        if(length(potentialDestinations)>1) 
        {
          beeRoute[ind,visitNumber] = sample(potentialDestinations,size=1,
                                             prob=probabilities);
        } else {
          beeRoute[ind,visitNumber] = potentialDestinations;
        }
        
      } else {
        if(indPos==1) 
        { 
          beeData$boutFinished[ind] = TRUE;
          stillForaging = setdiff(stillForaging,ind);
          next
        } else {
          beeRoute[ind,visitNumber] = 1;
        }
      };
      # Update distance travelled
      beeData$distanceTravelled[ind] = beeData$distanceTravelled[ind] + DistanceBetween(as.numeric(arrayGeometry[beeRoute[ind,visitNumber-1],2:3]),
                                                                                        as.numeric(arrayGeometry[beeRoute[ind,visitNumber],2:3]));
      
      # Check if the bee chose the nest
      if(beeData$allowNestReturn[ind])
      {
        if(beeRoute[ind,visitNumber]==1 && beeData$distanceTravelled[ind]>0) beeData$boutFinished[ind]=T;
      }
    }
    
    
    # Checking if some individuals reached the same flower. If so, it triggers a competition interaction.
    flowerInCompetition = beeRoute[duplicated(beeRoute[,visitNumber]),visitNumber];
    
    
    # We isolate the flowers on which there is a competition, exclude potential detections of nest.
    flowerInCompetition = unique(flowerInCompetition); 
    flowerInCompetition = setdiff(flowerInCompetition,c(0,1));
    
    
    # Competition check
    # We create a matrix saying if a bee feeds (1) or not (0). Default to 1.
    whoFeeds = matrix(1,ncol=numberOfBees,nrow=1);
    for(flower in flowerInCompetition) 
    {
      # Which individuals are in competition
      individualsInCompetition = which(beeRoute[,visitNumber]==flower);
      
      # Which wins and which loses
      interactionWinner = sample(individualsInCompetition,size=1,prob=beeData$winProbabilities[individualsInCompetition]);
      interactionLoser = setdiff(individualsInCompetition,interactionWinner);
      
      # Set the loser's feeding potential to 0.
      whoFeeds[interactionLoser] = 0;
      
      # Updating the negative outcome for the loser.
      for(indLost in interactionLoser)
      {
        if(indFlowerOutcome[[indLost]][beeRoute[indLost,visitNumber-1],flower]==0)
        {
          indFlowerOutcome[[indLost]][beeRoute[indLost,visitNumber-1],flower] = -1;
          if(onlineReinforcement) 
          {
            if (!useQLearning){
              learningArray[[indLost]][beeRoute[indLost,visitNumber-1],flower] = 
                learningArray[[indLost]][beeRoute[indLost,visitNumber-1],flower] * beeData$abandonFactor[indLost];
              learningArray[[indLost]] = NormalizeMatrix(learningArray[[indLost]])
            }else{
              reward=0;
              if (costOfFlying){reward=reward-(distanceMatrix[beeRoute[indLost,visitNumber-1]][flower])/max(distanceMatrix)};
              learningArray[[indLost]]=ApplyOnlineQLearning(learningArray[[indLost]],(beeRoute[indLost,visitNumber-1]),flower,reward,alphaPos,alphaNeg,gammaQL); #TBD reward
            }
            
          }
          
        }
      }
    }
    whoFeeds[which(beeData$boutFinished==T)] = 0;
    
    
    # Feeding Loop
    for(ind in which(whoFeeds==1)) 
    {
      if(resourcesOnFlowers[beeRoute[ind,visitNumber]]==1)
      {
        # If there was a resource available, the individual feeds. The flower is emptied.
        resourcesOnFlowers[beeRoute[ind,visitNumber]]=0;
        beeData$numberOfResources[ind]=beeData$numberOfResources[ind]+1;
      } else {
        
        # If there wasn't any food on the flower, the individual does not feed.
        whoFeeds[ind] = 0;
      }
    }
    
    
    # Update whoFeeds output
    listOfBoutResources[[allVisitsIndex]] = whoFeeds;
    
    # Increases the counter if no resource is found on the flower
    for(ind in which(whoFeeds==0))
    {
      if(leaveAfterMaxFail) arrayOfVectorUse[beeRoute[ind,visitNumber-1],beeRoute[ind,visitNumber],ind] = arrayOfVectorUse[beeRoute[ind,visitNumber-1],beeRoute[ind,visitNumber],ind] + 1;
    }
    
    
    # Check on passive punitive reaction (if flower was empty on first visit)
    for(ind in stillForaging)
    {
      flowerVisited = beeRoute[ind,visitNumber];
      
      # If this is their first visit on this flower
      if(flowerVisited!=1)
      {
        if(indFlowerOutcome[[ind]][beeRoute[ind,visitNumber-1],flowerVisited]==0) 
        {
          # If they fed on this flower
          if(whoFeeds[ind]==1)
          {
            if(!beeData$routeCompare[ind]) 
            {
              indFlowerOutcome[[ind]][beeRoute[ind,visitNumber-1],flowerVisited] = 1;
              if(onlineReinforcement) 
              {
                if (!useQLearning){
                  learningArray[[ind]][beeRoute[ind,visitNumber-1],flowerVisited] = 
                    learningArray[[ind]][beeRoute[ind,visitNumber-1],flowerVisited] * beeData$learningFactor[ind];
                  learningArray[[ind]] = NormalizeMatrix(learningArray[[ind]])
                }else{
                  reward=1;
                  if (costOfFlying){reward=reward-(distanceMatrix[beeRoute[ind,visitNumber-1]][flowerVisited])/max(distanceMatrix)};
                  learningArray[[ind]]=ApplyOnlineQLearning(learningArray[[ind]],beeRoute[ind,visitNumber-1],flowerVisited,1,alphaPos,alphaNeg,gammaQL);#reward TBD
                }
              }
            }
          } else {
            indFlowerOutcome[[ind]][beeRoute[ind,visitNumber-1],flowerVisited] = -1;
            if(onlineReinforcement) 
            {
              if (!useQLearning){
                learningArray[[ind]][beeRoute[ind,visitNumber-1],flowerVisited] = 
                  learningArray[[ind]][beeRoute[ind,visitNumber-1],flowerVisited] * beeData$abandonFactor[ind];
                learningArray[[ind]] = NormalizeMatrix(learningArray[[ind]])
              }else{
                reward=0;
                if (costOfFlying){reward=reward-(distanceMatrix[beeRoute[ind,visitNumber-1]][flowerVisited])/max(distanceMatrix)};
                learningArray[[ind]]=ApplyOnlineQLearning(learningArray[[ind]],beeRoute[ind,visitNumber-1],flowerVisited,0,alphaPos,alphaNeg,gammaQL);#reward TBD
              }
            }
            
          }
        }
      }
    }
    
    
    # Check end of foraging bout
    beeData$boutFinished[which(beeData$numberOfResources==beeData$maxCrop)]=T;
    
    
    # Fail safe limit of distance travelled
    for(ind in 1:numberOfBees)
    {
      if(beeData$distanceTravelled[ind]>=beeData$maxDistance[ind]) beeData$boutFinished[ind] = T;
    }
    
  }
  
  #### Learning Process
  
  # Initialize the output vector
  routeQualities = c();
  
  # Post-bout Learning phase loop for each bee
  for(ind in 1:numberOfBees)
  {
    if(beeData$differentExperienceSimulation[ind] & bout<beeData$startingBoutForNaive[ind])
    {
      routeQualities[ind] = 0;
      break
    }
    
    
    # Get the route quality, then round it to 8 decimals (to match with the sinked value in .txt file)
    route = beeRoute[ind,which(beeRoute[ind,]!=0)];
    
    if(route[length(route)]!=1) route = c(route,1);
    routeQuality = GetRouteQuality(route,arrayGeometry,beeData$numberOfResources[ind]);
    routeQuality = round(routeQuality,8);
    
    # Check if the new route quality is higher than the optimal route found initially. If so, replace old optimal route.
    # Optimal route =/= Best route known. Optimal route is the TSP solution of the array.
    # This ensures that if the initial optimal route assessment failed (as it is assessed via simulations), any new more optimized route replaces the old one.
    if(!is.null(optimalRouteQuality))
    {
      if(routeQuality > optimalRouteQuality)
      {
        {
          if(!silentSim)
          {
            cat("The following route ended with superior quality than optimal : ",paste(route,collapse="-"),"\n",sep="");
          }
          optimalRouteQuality = routeQuality;
          write.csv(optimalRouteQuality,paste(arrayFolder,"/optimalRoute.csv",sep=""),row.names = F);
        }
      }
    }
    
    # Apply the learning (positive & negative) to the probability matrix.
    if(!onlineReinforcement) learningArray[[ind]] = ApplyLearning(route,
                                                                  learningArray[[ind]],
                                                                  indFlowerOutcome[[ind]],
                                                                  beeData[ind,],
                                                                  minProbVisit,
                                                                  routeQuality = routeQuality);
    
    
    # Register the route quality of the individual
    routeQualities[ind] = routeQuality;
    
    if(beeData$routeCompare[ind] && 
       beeData$bestRouteQuality[ind]<routeQuality && 
       beeData$numberOfResources[ind]==beeData$maxCrop[ind]) 
      beeData$bestRouteQuality[ind] = routeQuality;
  }
  
  return(list(Sequences = beeRoute,
              Learning = learningArray,
              beeData = beeData,
              quality = routeQualities,
              optimalRouteQuality = optimalRouteQuality,
              listOfBoutResources = listOfBoutResources))
}



#########################
## MAIN
#########################

# Get starting time of simulation to get computation time.
simStart = Sys.time();

# Create Output directory in the current working directory.
dir.create("Output",showWarnings = F)
dir.create("Arrays",showWarnings = F)

# If environmentType is not a "generate", there is no need for multiple arrays.
if(environmentType!="generate") numberOfArrays = 1;



numberOfParameterSets = 0;

# Successive loops for each parameter. Thus, all parameter combinations are tested.

for (alphaPos in alphaPosList)
{
  
  for (alphaNeg in alphaNegList)
  { 
    
    for (betaQL in betaList)
    {
      for (gammaQL in gammaList)
      {
        
        for(algorithmRC in param.useRouteCompare)
        {
          for(learningValue in param.learningFactor)
          {
            for(abandonValue in param.abandonFactor)
            {
              # 4.1 - Initializing ------------------------------------------------------------
              
              numberOfParameterSets = numberOfParameterSets + 1;
              
              # Create test name according to parameter values
              if(algorithmRC) textRC = "RouteCompare" else textRC = "NoRouteCompare";
              timnw = as.character(Sys.time());timstmp = paste(substr(timnw,1,4),substr(timnw,6,7),substr(timnw,9,10),substr(timnw,12,13),substr(timnw,15,16),substr(timnw,18,19),sep="")
              testName = paste(environmentType,"-",testNameGeneral,"-ParamSet",numberOfParameterSets,"-",timstmp,sep="")
              if(!silentSim) cat("Starting simulation for test : ",testName,"\n",sep="");
              # Create the output folders
              outputFolder = paste(getwd(),"/Output/",testName,sep="");
              dir.create(outputFolder,showWarnings = F);
              
              
              
              # Complete list of individual parameters. Can be a single value or a vector if you want to attribute different values to each forager.
              # Cannot take lists to test different combinations of parameters.
              paramIndiv = list(maxDistance = maximumBoutDistance,
                                #maxCrop = (nrow(arrayGeometry)-1)/numberOfBees,
                                maxCrop = 5,
                                learningFactor = learningValue,
                                abandonFactor = abandonValue,
                                maxFails = numberOfMaxFails,
                                routeCompare = algorithmRC,
                                bestRouteQuality = 0,
                                leaveAfterMaxFail = leaveAfterMaxFail,
                                allowNestReturn = allowNestReturn,
                                forbidReverseVector = forbidReverseVector,
                                differentExperienceSimulation = differentExperienceSimulation,
                                startingBoutForNaive = startingBoutForNaive,
                                winProbabilities = rep(1/numberOfBees,times=numberOfBees),
                                useQLearning=useQLearning,
                                initializeQTable=initializeQTable,
                                alphaPos=alphaPos,
                                alphaNeg=alphaNeg,
                                beta=betaQL,
                                gamma=gammaQL)
              ;
              
              # 4.2 - Simulation --------------------------------------------------------------
              
              # Parameters tracked during the simulation for each bee
              paramTracking = list(numberOfResources = rep(0,numberOfBees),
                                   probabilityWin = rep(0,numberOfBees),
                                   boutFinished = rep(F,numberOfBees),
                                   distanceTravelled = rep(0,numberOfBees));
              
              # Initialize the parameter dataframe for each bee
              beeData = InitializeBeeData(numberOfBees,paramTracking,paramIndiv);
              
              
              
              
              for(arrayNumber in 1:numberOfArrays)
              {
                # Initialize the informations of the arrays
                arrayInfos = list(environmentType = environmentType,
                                  numberOfResources = numberOfResources,
                                  numberOfPatches = numberOfPatches,
                                  patchinessIndex = patchinessIndex,
                                  envSize = envSize,
                                  flowerPerPatch = flowerPerPatch)
                
                # Generate array: See 3.1 in 02-Parameters for details.
                env = CreateEnvironment(arrayInfos,arrayNumber,reuseGeneratedArrays)
                
                # Redistribute the outputs of env in variables.
                arrayGeometry = env$arrayGeometry;
                arrayInfos = env$arrayInfos;
                arrayFolder = env$arrayFolder;
                
                print('initializing probability matrices')
                # Initialize the probability matrix
                initialProbabilityArray = GetDistFctProbability(arrayGeometry,distFactor,beeData);
                
                # Apply modifiers to matrices depending on options
                for(ind in 1:numberOfBees)
                {
                  if(!beeData$allowNestReturn[ind]) initialProbabilityArray[[ind]][,1] = 0;
                }
                
                # Distribute probability matrix to all bees
                initialProbabilityMatrices = list();
                for(bee in 1:numberOfBees) initialProbabilityMatrices[[bee]] = NormalizeMatrix(initialProbabilityArray[[bee]]);
                
                # Save a copy of the initial probability matrix
                write.csv(initialProbabilityMatrices[[1]],paste(arrayFolder,"/probabilityMatrix.csv",sep=""),row.names = F);
                
                print('initializing Q Tables')
                #Initialize Q Tables
                initialQTableList=InitializeQTableList(initializeQTable,arrayGeometry,distFactor,beeData);
                #Save a copy of the initial QTable
                write.csv(initialQTableList[[1]],paste(arrayFolder,"/QTable.csv",sep=""),row.names = F);
                
                # Get maximum route quality of the array (simulating 300 1Ind for 30 each bouts to try and find the optimal route).
                {
                  optimalRouteQuality = NULL;
                  optimalRouteQuality = SimDetectionOptimalRoute(arrayInfos$arrayID,arrayGeometry,beeData,initialProbabilityMatrices,arrayFolder,silentSim,0);
                  cat("Optimal Route Quality : ",optimalRouteQuality,"\n",sep="")
                }
                
                # Get maximum route quality for 2 ind of the array (simulating 300 1Ind for 30 each bouts to try and find the optimal route).
                {
                  optimalRouteQuality2Ind = NULL;
                  optimalRouteQuality2Ind = SimDetectionOptimalRoute2Ind(arrayInfos$arrayID,arrayGeometry,beeData,initialProbabilityMatrices,arrayFolder,silentSim,0);
                  cat("Optimal Route Quality for 2 Individuals : ",optimalRouteQuality2Ind,"\n",sep="")
                }
                
                # Initialize distance matrix
                distanceMatrix = as.matrix(dist(arrayGeometry[,2:3],upper = T,diag=T)); 
                
                outputFolderSim = paste(outputFolder,"/Array",sprintf("%02d",arrayNumber),sep="");
                dir.create(outputFolderSim,showWarnings = F);
                # Create a DF of information to be retrieve in further analyses (and remember what parameters were used).
                beeInfos = BuildBeeInfos(numberOfBees,paramIndiv,arrayInfos$arrayID);
                write.csv(beeInfos,paste(outputFolderSim,"/beeInfos.csv",sep=""),row.names = F);
                
                savedArrayInfos = arrayInfos;
                arrayInfosSaved = arrayInfos;
                arrayInfosSaved$flowerPerPatch = paste(arrayInfosSaved$flowerPerPatch,collapse="");
                write.csv(as.data.frame(arrayInfosSaved),paste(outputFolderSim,"/arrayInfos.csv",sep=""),row.names = F);
                write.csv(arrayGeometry,paste(outputFolderSim,"/arrayGeometry.csv",sep=""),row.names = F);
                
                # Initialize the output structures
                listOfVisitationSequences = list();
                listOfResourcesTaken = list();
                arrayOfMatrixDistance = data.frame(sim = rep(1:numberOfSimulations,each=numberOfBouts*numberOfBees),
                                                   bout = rep(1:numberOfBouts,times = numberOfSimulations, each=numberOfBees),
                                                   bee = rep(1:numberOfBees,times=numberOfSimulations*numberOfBouts),
                                                   distance = numeric(numberOfSimulations*numberOfBouts*numberOfBees));
                matrixOfBeeData = matrix(0,ncol=5,nrow=numberOfSimulations*numberOfBouts*numberOfBees);
                matrixOfBeeData[,1] = rep(1:numberOfSimulations,each=numberOfBouts*numberOfBees);
                matrixOfBeeData[,2] = rep(1:numberOfBouts,times=numberOfSimulations,each=numberOfBees);
                matrixOfBeeData[,3] = rep(1:numberOfBees,times=numberOfSimulations*numberOfBouts);
                
                i = 1;
                j = 1;
                # Sim loop
                for(sim in 1:numberOfSimulations)
                {
                  # Initialize simulation objects
                  beeSequences = list();
                  if(!useQLearning) {learningArray=initialProbabilityMatrices} else {learningArray=initialQTableList};
                  # Bout loop
                  for(bout in 1:numberOfBouts)
                  {
                    beeData = RebootBeeData(beeData);
                    currentBout = CompetitiveRoute(bout,arrayGeometry,learningArray,beeData,optimalRouteQuality,silentSim,useQLearning);
                    # For Sensitivity Analysis, compare learningArray and currentBout$Learning
                    for(bee in 1:numberOfBees)
                    {
                      previousMatrix = learningArray[[bee]];
                      nextMatrix = currentBout$Learning[[bee]];
                      
                      arrayOfMatrixDistance$distance[i+(bee-1)] = sum(abs(previousMatrix - nextMatrix));
                    }
                    
                    ## Update variables
                    learningArray = currentBout$Learning;
                    beeData = currentBout$beeData;
                    optimalRouteQuality = currentBout$optimalRouteQuality;
                    
                    matrixOfBeeData[i:(i+numberOfBees-1),4] = beeData$numberOfResources;
                    matrixOfBeeData[i:(i+numberOfBees-1),5] = currentBout$quality;
                    
                    listOfVisitationSequences[[(sim-1)*numberOfBouts+bout]] = currentBout$Sequences;
                    
                    i = i + numberOfBees;
                    
                    listOfResourcesTaken[[j]] = currentBout$listOfBoutResources;
                    j = j + 1;
                  }
                }
                
                # 4.3 - Formatting Raw Data ------------------------------------------
                maxLength = 0;
                for(visitSeq in 1:length(listOfVisitationSequences))
                {
                  maxLength = max(maxLength,dim(listOfVisitationSequences[[visitSeq]])[2]);
                }
                i=0;
                matrixOfVisitationSequences = matrix(0,nrow=numberOfBouts*numberOfSimulations*numberOfBees,ncol=maxLength+2);
                for(bout in 1:length(listOfVisitationSequences))
                {
                  for(bee in 1:numberOfBees)
                  {
                    i = i + 1;
                    seqBee = listOfVisitationSequences[[bout]][bee,];
                    if(seqBee[length(seqBee)]!=1) seqBee = c(seqBee[which(seqBee!=0)],1);
                    lengthSeq = length(seqBee);
                    seqBee = c(seqBee,rep(0,times=(maxLength+2)-lengthSeq));
                    matrixOfVisitationSequences[i,]=seqBee;
                  }
                }
                
                matrixOfVisitationSequences = cbind(rep(1:numberOfSimulations,each=numberOfBouts*numberOfBees),
                                                    rep(1:numberOfBouts,times=numberOfSimulations,each=numberOfBees),
                                                    rep(1:numberOfBees,times=numberOfSimulations*numberOfBouts),
                                                    matrixOfVisitationSequences);
                
                write.csv(matrixOfVisitationSequences,paste(outputFolderSim,"/matrixOfVisitationSequences.csv",sep=""),row.names=F);
                
                #### Transform the Route Quality & Foraged Resources into arrays (temporary, should use arrays from the beginning)
                arrayOfRouteQuality = array(data=0,dim=c(numberOfSimulations,numberOfBouts,numberOfBees));
                arrayOfRouteQualityAbs = array(data=0,dim=c(numberOfSimulations,numberOfBouts,numberOfBees));
                arrayOfForagedResources = array(data=0,dim=c(numberOfSimulations,numberOfBouts,numberOfBees));
                vectorOfRawRouteQuality = c();
                vectorOfRouteQuality = c();
                for(ind in 1:numberOfBees)
                {
                  matrixOfRouteQualityAbs = matrix(subset(matrixOfBeeData[,5],matrixOfBeeData[,3]==ind),nrow=numberOfSimulations,ncol=numberOfBouts,byrow=T);
                  
                  if(!is.null(optimalRouteQuality)) matrixOfRouteQuality = matrixOfRouteQualityAbs/optimalRouteQuality;
                  
                  arrayOfRouteQuality[,,ind] = matrixOfRouteQuality;
                  arrayOfRouteQualityAbs[,,ind] = matrixOfRouteQualityAbs;
                  vectorOfRawRouteQuality = c(vectorOfRawRouteQuality,as.vector(t(arrayOfRouteQualityAbs[,,ind])))
                  vectorOfRouteQuality = c(vectorOfRouteQuality,as.vector(t(arrayOfRouteQuality[,,ind])))
                  
                  matrixOfForagedResources = matrix(subset(matrixOfBeeData[,4],matrixOfBeeData[,3]==ind),nrow=numberOfSimulations,ncol=numberOfBouts,byrow=T);
                  arrayOfForagedResources[,,ind] = matrixOfForagedResources;
                }
                
                
                
                routeQualityAbsDF = data.frame(Bee = rep(1:numberOfBees,each = numberOfSimulations*numberOfBouts),
                                               Simulation = rep(1:numberOfSimulations,times = numberOfBees,each = numberOfBouts),
                                               Bout = rep(1:numberOfBouts, times = numberOfSimulations*numberOfBees),
                                               rawQuality = vectorOfRawRouteQuality,
                                               relativeQuality = vectorOfRouteQuality);
                write.csv(routeQualityAbsDF,paste(outputFolderSim,"/routeQualityDF.csv",sep=""),row.names=F);
                
                
                # arrayOfMatrixDistance
                write.csv(arrayOfMatrixDistance,paste(outputFolderSim,"/arrayOfMatrixDistance.csv",sep=""),row.names = F);
                
                
                # 4.4 - Videos ------------------------------------------------------------------
                if(videoOutput)
                {
                  # Create a directory for video outputs
                  outputFolderTracking = paste(outputFolderSim,"/Tracking",sep="");
                  dir.create(outputFolderTracking,showWarnings = F);
                  
                  CreateVideoOutput(arrayGeometry,matrixOfVisitationSequences,listOfResourcesTaken,beeData,framerate,simulationsToPrint,outputFolderTracking)
                }
              }# arrayNumber
            }# param.abandonFactor
          }# param.learningFactor
        }# param.useRouteCompare
      } #gamma
    } #beta
  } #alphaneg
} #alphapos

simEnd = Sys.time();
simDuration = difftime(simEnd,simStart,units="secs");
cat("Simulation completed in :",as.numeric(simDuration),"seconds \n")

