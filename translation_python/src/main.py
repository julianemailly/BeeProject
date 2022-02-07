

"""

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

  
  for (alphaNeg in alphaNegList)
   
    
    for (betaQL in betaList)
    
      for (gammaQL in gammaList)
      
        
        for(algorithmRC in param.useRouteCompare)
        
          for(learningValue in param.learningFactor)
          
            for(abandonValue in param.abandonFactor)
            
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
                
                  if(!beeData$allowNestReturn[ind]) initialProbabilityArray[[ind]][,1] = 0;
                
                
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
                
                  optimalRouteQuality = NULL;
                  optimalRouteQuality = SimDetectionOptimalRoute(arrayInfos$arrayID,arrayGeometry,beeData,initialProbabilityMatrices,arrayFolder,silentSim,0);
                  cat("Optimal Route Quality : ",optimalRouteQuality,"\n",sep="")
                
                
                # Get maximum route quality for 2 ind of the array (simulating 300 1Ind for 30 each bouts to try and find the optimal route).
                
                  optimalRouteQuality2Ind = NULL;
                  optimalRouteQuality2Ind = SimDetectionOptimalRoute2Ind(arrayInfos$arrayID,arrayGeometry,beeData,initialProbabilityMatrices,arrayFolder,silentSim,0);
                  cat("Optimal Route Quality for 2 Individuals : ",optimalRouteQuality2Ind,"\n",sep="")
                
                
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
                
                  # Initialize simulation objects
                  beeSequences = list();
                  if(!useQLearning) learningArray=initialProbabilityMatrices else learningArray=initialQTableList;
                  # Bout loop
                  for(bout in 1:numberOfBouts)
                  
                    beeData = RebootBeeData(beeData);
                    currentBout = CompetitiveRoute(bout,arrayGeometry,learningArray,beeData,optimalRouteQuality,silentSim,useQLearning);
                    # For Sensitivity Analysis, compare learningArray and currentBout$Learning
                    for(bee in 1:numberOfBees)
                    
                      previousMatrix = learningArray[[bee]];
                      nextMatrix = currentBout$Learning[[bee]];
                      
                      arrayOfMatrixDistance$distance[i+(bee-1)] = sum(abs(previousMatrix - nextMatrix));
                    
                    
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
                  
                
                
                # 4.3 - Formatting Raw Data ------------------------------------------
                maxLength = 0;
                for(visitSeq in 1:length(listOfVisitationSequences))
                
                  maxLength = max(maxLength,dim(listOfVisitationSequences[[visitSeq]])[2]);
                
                i=0;
                matrixOfVisitationSequences = matrix(0,nrow=numberOfBouts*numberOfSimulations*numberOfBees,ncol=maxLength+2);
                for(bout in 1:length(listOfVisitationSequences))
                
                  for(bee in 1:numberOfBees)
                  
                    i = i + 1;
                    seqBee = listOfVisitationSequences[[bout]][bee,];
                    if(seqBee[length(seqBee)]!=1) seqBee = c(seqBee[which(seqBee!=0)],1);
                    lengthSeq = length(seqBee);
                    seqBee = c(seqBee,rep(0,times=(maxLength+2)-lengthSeq));
                    matrixOfVisitationSequences[i,]=seqBee;
                  
                
                
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
                
                  matrixOfRouteQualityAbs = matrix(subset(matrixOfBeeData[,5],matrixOfBeeData[,3]==ind),nrow=numberOfSimulations,ncol=numberOfBouts,byrow=T);
                  
                  if(!is.null(optimalRouteQuality)) matrixOfRouteQuality = matrixOfRouteQualityAbs/optimalRouteQuality;
                  
                  arrayOfRouteQuality[,,ind] = matrixOfRouteQuality;
                  arrayOfRouteQualityAbs[,,ind] = matrixOfRouteQualityAbs;
                  vectorOfRawRouteQuality = c(vectorOfRawRouteQuality,as.vector(t(arrayOfRouteQualityAbs[,,ind])))
                  vectorOfRouteQuality = c(vectorOfRouteQuality,as.vector(t(arrayOfRouteQuality[,,ind])))
                  
                  matrixOfForagedResources = matrix(subset(matrixOfBeeData[,4],matrixOfBeeData[,3]==ind),nrow=numberOfSimulations,ncol=numberOfBouts,byrow=T);
                  arrayOfForagedResources[,,ind] = matrixOfForagedResources;
                
                
                
                
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
                
                  # Create a directory for video outputs
                  outputFolderTracking = paste(outputFolderSim,"/Tracking",sep="");
                  dir.create(outputFolderTracking,showWarnings = F);
                  
                  CreateVideoOutput(arrayGeometry,matrixOfVisitationSequences,listOfResourcesTaken,beeData,framerate,simulationsToPrint,outputFolderTracking)
                
              # arrayNumber
            # param.abandonFactor
          # param.learningFactor
        # param.useRouteCompare
       #gamma
     #beta
   #alphaneg
 #alphapos

simEnd = Sys.time();
simDuration = difftime(simEnd,simStart,units="secs");
cat("Simulation completed in :",as.numeric(simDuration),"seconds \n")

"""