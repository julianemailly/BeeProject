# Parameters --------------------------------------------------------------

# Array details.
testNameGeneral = "Dubois"; # An identification name for the simulation.
environmentType = "plos"; # Either (i) input an array name (refer to folder names in the "Arrays" folder - Note that this folder is automatically created when you first generate an array), or (ii) input "generate" to generate procedural arrays. For the latter, provide details in 3.1.1

## 3.1.1 - Details for procedural arrays (if  using "generate" as environmentType). If not used, skip to 3.2
numberOfResources = 5; # Number of flowers (all patches combined)
numberOfPatches = 1; # Number of patches in which the flowers are distributed
patchinessIndex = 0; # Patchiness of the array. Takes values between 0 (homogeneous) and 1 (very heterogeneous).
envSize = 500; # Size of the environment in meters. 
flowerPerPatch = NULL; # Number of flowers per patch. If only one patch, set to NULL. Takes one value per patch, sum need to be equal to numberOfResources.
numberOfArrays = 10; # Number of different arrays created using the values above. Only used if environmentType == "generate".
reuseGeneratedArrays = TRUE; # If TRUE and there already are generated arrays with the same parameters, they will be used instead of generating new ones.

# Simulation parameters
numberOfBees = 1; # Number of bees moving simultaneously in the environment.
numberOfSimulations = 10; # Number of simulations for each set of parameter.
numberOfBouts = 20; # Number of bouts in each simulation.
distFactor = 2; # This variable contains the power at which the distance is taken in the [probability = 1/d^distFactor] function to estimate movement probabilities.

# Forager parameters. 
# Each item of the list objects will replicate the simulations for its values. The model will run for all combinations of parameters. May quickly increase computation time.
param.useRouteCompare = list(FALSE); # TRUE : Use the route-based learning algorithm (Reynolds et al., 2013). FALSE : Use our vector-based algorithm.
# /!\ Warning : The route-based algorithm is not identical to that of Reynolds et al., 2013. Mainly, it uses the route with revisits when assessing route quality.
param.learningFactor = list(1.5); # Strength of the learning process. Translates as a multiplication of any vector probability P by this factor (Should be 1 <= learningFactor)
param.abandonFactor = list(1.); # Strength of the abandon process. Translates as a multiplication of any vector probability P by this factor (Should be 0 <= abandonFactor <= 1)
maximumBoutDistance = 3000; # Maximum distance the bee can travel before being exhausted. After reaching this threshold, the bee goes back to its nest no matter what.

# Output management & technical parameters.
videoOutput = FALSE; # If TRUE, outputs videos showing the bee movements during simulation. /!\ Requires FFMPEG installed on computer /!\ You can find FFMPEG at the following address : https://ffmpeg.org/download.html. Also, not tested on other OS than Windows 10, may fail to work on Mac.
framerate = 8; # The number of frames per seconds on the video output
simulationsToPrint = 1 # Select the number of videos to print. Must not be above numberOfSimulations.
silentSim = FALSE; # If FALSE, provides comments in the Console on the progression of the simulation.

# Advanced parameters
minProbVisit = 0.0001; # (Deprecated as the learning is now multiplicative) Keep a minimal probability of vector use if its probability gets negative.
leaveAfterMaxFail = FALSE; # If a vector use leads to no reward twice in a bout, forbid its use until the end of the bout.
numberOfMaxFails = 2; # If leaveAfterMaxFail is TRUE, set the number of fails needed for the bee to ignore a vector.
allowNestReturn = TRUE; # If TRUE, probabilities to go to the nest from any position are not set to 0. If the nest is reached, the bout ends.
forbidReverseVector = TRUE; # If TRUE, foragers ignore reverse vectors. Example : if it just did 2->3, the prob of doing 3->2 is 0 until its next movement.
differentExperienceSimulation = FALSE; # If TRUE & numberOfBees = 2 : Setup the simulation so that Bee 1 starts foraging for n bouts before Bee 2 starts foraging.
startingBoutForNaive = c(1); # Bout at which each bee starts foraging. Should be < to numberOfBouts. Expects as many values as the numberOfBees if differentExperienceSimulation is TRUE.
onlineReinforcement = TRUE; # If TRUE, probability changes after a good/bad experience is immediate and not at the end of the bout.

# Parameters for the Q-learning/Rescorla-Wagner model
useQLearning=FALSE; #TRUE: if you want to use this Q learning model, FALSE: if you want to use T. Dubois' model
initializeQTable="zero"; #'zero' if yo want the Q table to be initialized as a null matrix, 'distance' if you want it to be initialized as the 1/d^distFactor matrix, 'noisydist if you want to add noise to the distance ditribution
alphaPosList=list(0.4); #positive reinforcement learning rate: 0<=alphaPos<=1
alphaNegList=list(0.4); #negative reinforcement learning rate: 0<=alphaNeg<=1
betaList=list(15); #exploration-exploitation parameter: 0<=beta
gammaList=list(0); #temporal discounting factor: 0<=gamma<=1. Here, set to 0 for simplicity
if (useQLearning) {onlineReinforcement=TRUE};


dynamicBeta=FALSE;
if (dynamicBeta) {
  startingBeta=1;
  finalBeta=10;
  boutsToIncreaseBeta=20;
  betaQLVector=rep(finalBeta,numberOfBouts);
  for (bout in 1:boutsToIncreaseBeta) {betaQLVector[bout]=startingBeta+(bout-1)*(finalBeta-startingBeta+1)/boutsToIncreaseBeta}
};

costOfFlying= FALSE
