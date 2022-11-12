import numpy as np
from numba import njit
import json
from .ActivationFunction import *

(
    LAYER_INPUT,
    # LAYER_INPUT_BACKFED,
    # LAYER_INPUT_NOISY,

    LAYER_HIDDEN,
    # LAYER_HIDDEN_PROBABLISTIC,
    # LAYER_HIDDEN_SPIKING,

    LAYER_OUTPUT,
    # LAYER_OUTPUT_INPUT,

    # LAYER_RECURRENT,
    # LAYER_MEMORY,
    # LAYER_DIFFERENT_MEMORY,

    # LAYER_KERNEL
) = range(3)

class ConvolutionalNeuralNetwork:
    __nbInputNeuron: int                                                    # Nombre de neurones d'entrée

    __nbHiddenLayer: int                                                    # Nombre de couche caché
    __hiddenLayerNbNeuron: np.ndarray                                       # Liste du nombre de neurone de chaque couche
    __hiddenLayerActivationFunction: np.ndarray                             # Liste des fonction d'activation de chaque couche (l'index correspond a la position de la couche)
    __hiddenLayerTypeList: np.ndarray                                       # Liste des type de neurones de chaque couche
    __hiddenLayerWeightList: np.ndarray                                     # Liste des poids entre chaque neurone [ input->layer1[ hiddenNEuron1[inputNeuron1, inputNeuron2, ...], hiddenNeuron[inputNeuron1, inputNeuron2, ...]], layer1->layer2[[]]]
    __hiddenLayerBiais: np.ndarray                                          # Liste des taux d'activation de chaque neurone de chaque couche
    __hiddenLayerAlpha: np.ndarray                                          # Liste de l'alpha de chaque neurone de chaque couche
    __hiddenLayerValueList: np.ndarray                                      # Liste des valeurs des couches cachées
    __hiddenLayerCoutList: np.ndarray                                       # Liste des taux d'erreur des couches cachées
    __hiddenLayerDeltaList: np.ndarray                                      # Liste des importances d'erreur des couches cachées

    __nbOutputNeuron: int                                                   # Nombre de neurone de sortie
    __outputActivationFunction: int                                         # Fonction d'activation de la sortie
    __outputLayerType: int                                                  # Liste de neurone de sortie
    __outputLayerWeightList: np.ndarray                                     # Liste des poids d'entrées
    __outputLayerBiais: np.ndarray                                          # Liste des taux d'activation de chaque neurone
    __outputLayerAlpha: np.ndarray                                          # Liste de l'alpha de chaque neurone
    __outputLayerValue: np.ndarray                                          # Liste des valeurs des neurones
    __outputLayerCout: np.ndarray                                           # Liste des taux d'erreur de chaque neurones
    __outputLayerDelta: np.ndarray                                          # Liste des importances d'erreur de chaque neurone

    __nbMaxNeuronPerLayer: int                                              # Nombre de neurone maximal par couche

    def __init__(self, nbInputNeuron: int, nbOutputNeuron: int, outputNeuronType: int, outputActivationFunction: int, nbMaxNeuronPerLayer: int = 90) -> None:
        """
            Réseau de neurone par convolution
            Params:
                nbInputNeuron               :   Nombre de neurone d'entrée
                nbOutputNeuron              :   Nombre de neurone de sortie
                outputNeuronType            :   Type de neurone de sortie (LAYER_####)
                outputActivationFunction    :   Fonction d'activation de la couche de sortie (FUNCTION_####)
                nbMaxNeuronPerLayer         :   Nombre maximal de neurone par couche
        """
        self.__nbInputNeuron = nbInputNeuron                                                                        # Nombre de neurone d'entrée
        
        self.__nbHiddenLayer = 0                                                                                    # Aucune couche caché
        self.__hiddenLayerNbNeuron = np.zeros((0), dtype=np.int)                                                    # Nombre de neurone de chaque couche
        self.__hiddenLayerActivationFunction = np.zeros((0), dtype=np.int)                                          # [ nCouche ]
        self.__hiddenLayerTypeList = np.zeros((0), dtype=np.int)                                                    # [ nCouche ]
        self.__hiddenLayerWeightList = np.zeros((0, nbMaxNeuronPerLayer, nbMaxNeuronPerLayer), dtype=np.float64)    # [ nCouche * [ nNeuron * [ nInput ] ] ]
        self.__hiddenLayerBiais = np.zeros((0, nbMaxNeuronPerLayer), dtype=np.float64)                              # [ nCouche * [ nNeuron ] ]
        self.__hiddenLayerAlpha = np.ones((0, nbMaxNeuronPerLayer), dtype=np.float64)                               # [ nCouche * [ nNeuron ] ]
        self.__hiddenLayerValueList = np.zeros((0, nbMaxNeuronPerLayer), dtype=np.float64)                          # [ nCouche * [ nNeuron ] ]
        self.__hiddenLayerCoutList = np.zeros((0, nbMaxNeuronPerLayer), dtype=np.float64)                           # [ nCouche * [ nNeuron ] ]
        self.__hiddenLayerDeltaList = np.zeros((0, nbMaxNeuronPerLayer), dtype=np.float64)                          # [ nCouche * [ nNeuron ] ]

        self.__nbOutputNeuron = nbOutputNeuron                                                                      # int
        self.__outputLayerType = outputNeuronType                                                                   # int
        self.__outputActivationFunction = outputActivationFunction                                                  # int
        self.__outputLayerWeightList = np.zeros((nbMaxNeuronPerLayer, nbMaxNeuronPerLayer))                         # Liste des poids de chaque neurone d'entrée de chaque neurone de la couche de sortie
        self.__outputLayerAlpha = np.ones(nbMaxNeuronPerLayer)                                                      # Alpha de chaque neurone de sortie
        self.__outputLayerBiais = np.random.random(nbMaxNeuronPerLayer)                                             # Taux d'activation des neurones de sortie
        self.__outputLayerValue = np.zeros(nbMaxNeuronPerLayer)                                                     # Liste des valeur de chaque neurone de sortie
        self.__outputLayerCout = np.zeros(nbMaxNeuronPerLayer)                                                      # Liste des erreur de chaque neurone de la liste de sortie
        self.__outputLayerDelta = np.zeros(nbMaxNeuronPerLayer)                                                     # Liste des erreur de chaque neurone de la liste de sortie

        self.__nbMaxNeuronPerLayer = nbMaxNeuronPerLayer                                                            # Nombre max de neurone par couche

        self.__LinkOutputLayerWeight()                                                                              # On link la couche de sortie avec la couche d'entrée
        return

    def AddLayer(self, nbNeuron: int, neuronType: int, activationFunction: int) -> None:
        """
        Ajout d'une couche au réseau
        Params:
            nbNeuron            :   Nombre de neurone de la couche
            neuronType          :   Type de neurone de la couche (LAYER_####)
            activationFunction  :   Fonction d'activation de la couche (FUNCTION_####)
        """
        self.__hiddenLayerTypeList = np.concatenate((self.__hiddenLayerTypeList, np.array([neuronType])), axis=0)                               # Ajout du type de la couche
        self.__hiddenLayerNbNeuron = np.concatenate((self.__hiddenLayerNbNeuron, np.array([nbNeuron])), axis=0)                                 # Ajout du nombre de neurone de la couche
        self.__hiddenLayerActivationFunction = np.concatenate((self.__hiddenLayerActivationFunction, np.array([activationFunction])), axis=0)   # Ajout de la fonction d'activation de la couche
        
        self.__hiddenLayerWeightList.resize((self.__nbHiddenLayer+1, self.__nbMaxNeuronPerLayer, self.__nbMaxNeuronPerLayer), refcheck=False)   # On ajoute une couche en plus
        newWeightList = 2*np.random.random((self.__nbMaxNeuronPerLayer, self.__nbMaxNeuronPerLayer))-1                                          # Generation de tout les poids de la nouvelle couche
        if self.__nbHiddenLayer == 0:                                                                                                           # Si c'est la premiere couche
            for xNeuron in range(nbNeuron):                                                                                                         # Pour chaque neurone de la couche
                for inNeuron in range(self.__nbInputNeuron, self.__nbMaxNeuronPerLayer):                                                                # Pour chaque poids d'entrée inutile
                    newWeightList[xNeuron][inNeuron] = 0                                                                                                    # Remise a 0 du poids inutile
            for xNeuron in range(nbNeuron, self.__nbMaxNeuronPerLayer):                                                                             # Pour chaque neurone inutile de la couche
                newWeightList[xNeuron].fill(0)                                                                                                          # Remise a 0 de tout les poids inutiles
        else:                                                                                                                                   # Sinon ce n'est pas la premiere couche
            for xNeuron in range(nbNeuron):                                                                                                         # Pour chaque neurone de la couche
                for inNeuron in range(self.__hiddenLayerNbNeuron[self.__nbHiddenLayer-1], self.__nbMaxNeuronPerLayer):                                  # Pour chaque poids d'entrée inutile
                    newWeightList[xNeuron][inNeuron] = 0                                                                                                    # Remise a 0 du poids inutile
            for xNeuron in range(nbNeuron, self.__nbMaxNeuronPerLayer):                                                                             # Pour chaque neurone inutile de la couche
                newWeightList[xNeuron].fill(0)                                                                                                          # Remise a 0 de tout les poids inutiles
        self.__hiddenLayerWeightList[self.__nbHiddenLayer] += newWeightList                                                                     #On ajoute la liste des poids

        self.__hiddenLayerBiais.resize((self.__nbHiddenLayer+1, self.__nbMaxNeuronPerLayer), refcheck=False)                                    # On augmente le nombre de couche de biais
        for xNeuron in range(nbNeuron):                                                                                                         # Pour chaque neurone
            self.__hiddenLayerBiais[self.__nbHiddenLayer][xNeuron] = np.random.random((1))                                                          # Poids aleatoire

        self.__hiddenLayerAlpha.resize((self.__nbHiddenLayer+1, self.__nbMaxNeuronPerLayer), refcheck=False)                                    # On augmente le nombre de couche d'alpha
        for xNeuron in range(nbNeuron):                                                                                                         # Pour chaque neurone
            self.__hiddenLayerAlpha[self.__nbHiddenLayer][xNeuron] = 1                                                                              # Tout les alpha a 1
   
        self.__hiddenLayerValueList.resize((self.__nbHiddenLayer+1, self.__nbMaxNeuronPerLayer), refcheck=False)                                # On augmente le nombre de couche de valeur
        self.__hiddenLayerCoutList.resize((self.__nbHiddenLayer+1, self.__nbMaxNeuronPerLayer), refcheck=False)                                 # On augmente le nombre de couche de taux d'erreur
        self.__hiddenLayerDeltaList.resize((self.__nbHiddenLayer+1, self.__nbMaxNeuronPerLayer), refcheck=False)                                # On augmente le nombre de couche d'importance d'erreur
        
        self.__nbHiddenLayer += 1                                                                                                               # +1 hidden layer

        self.__LinkOutputLayerWeight()                                                                                                          # Regénération des liens avec la derniere couche
        return

    def SaveNetwork(self, filename: str) -> None:
        """
            Sauvegarde du reseau de neurone dans un fichier
            Params:
                filename : Nom du fichier de destination
        """
        output = {}                                                                                             # JSON du réseau
        
        output["nbInput"] = int(self.__nbInputNeuron)                                                           # Nombre d'entrée
        output["nbHiddenLayer"] = int(self.__nbHiddenLayer)                                                     # Nombre de couche
        output["nbOutput"] = int(self.__nbOutputNeuron)                                                         # Nombre de sortie
        output["nbMaxNeuronPerLayer"] = int(self.__nbMaxNeuronPerLayer)                                         # Nombre maximum de neurone par couche

        output["hiddenLayers"] = {}                                                                             # Liste des couches cachée
        output["hiddenLayers"]["nbNeuron"] = []                                                                 # Nombre de neurone par couche
        output["hiddenLayers"]["activationFunction"] = []                                                       # Fonction d'activation de chaque couche
        output["hiddenLayers"]["type"] = []                                                                     # Type de neurone de chaque couche
        output["hiddenLayers"]["weightList"] = []                                                               # Poids de chaque neurone de chaque couche
        output["hiddenLayers"]["biais"] = []                                                                    # Taux d'activation de chaque couche
        output["hiddenLayers"]["alpha"] = []                                                                    # Alpha pour chaque neurone de chaque couche
        for xLayer in range(self.__nbHiddenLayer):                                                              # Pour chaque neurone de la liste
            output["hiddenLayers"]["nbNeuron"].append(int(self.__hiddenLayerNbNeuron[xLayer]))                      # Nombre de neurone de la couche
            output["hiddenLayers"]["activationFunction"].append(int(self.__hiddenLayerActivationFunction[xLayer]))  # Fonction d'activation de la couche
            output["hiddenLayers"]["type"].append(int(self.__hiddenLayerTypeList[xLayer]))                          # Type de neurone de la couche actuel

            output["hiddenLayers"]["weightList"].append([])                                                         # Preparation de la liste des poids de chaques neurones
            output["hiddenLayers"]["biais"].append([])                                                              # Preparation de la liste des biais de chaque neurone
            output["hiddenLayers"]["alpha"].append([])                                                              # Preparation de la liste des alpha de chaque neuronee
            for xNeuron in range(self.__nbMaxNeuronPerLayer):                                                       # Pour chaque neurone
                output["hiddenLayers"]["weightList"][xLayer].append([])                                                 # Preparation de la liste pour chaque entrée
                for inNeuron in range(self.__nbMaxNeuronPerLayer):                                                      # Pour chaque neurone d'entrée
                    output["hiddenLayers"]["weightList"][xLayer][xNeuron].append(float(self.__hiddenLayerWeightList[xLayer][xNeuron][inNeuron])) # Ajout du poids
                output["hiddenLayers"]["biais"][xLayer].append(float(self.__hiddenLayerBiais[xLayer][xNeuron]))         # Taux d'activation du neurone
                output["hiddenLayers"]["alpha"][xLayer].append(float(self.__hiddenLayerAlpha[xLayer][xNeuron]))         # Alpha du neurone

        output["outputLayer"] = {}                                                                              # Couche de sortie
        output["outputLayer"]["activationFunction"] = self.__outputActivationFunction                           # Fonction d'activation de la couche de sortie
        output["outputLayer"]["type"] = self.__outputLayerType                                                  # Type de neurone
        output["outputLayer"]["weightList"] = []                                                                # Liste des poids
        output["outputLayer"]["biais"] = []                                                                     # Taux d'activation de chaque neurone
        output["outputLayer"]["alpha"] = []                                                                     # Alpha pour chaque neurone de chaque neurone
        for xNeuron in range(self.__nbMaxNeuronPerLayer):                                                       # Pour chaque neurone
            output["outputLayer"]["weightList"].append([])                                                          # Preparation de la liste pour chaque entrée
            for inNeuron in range(self.__nbMaxNeuronPerLayer):                                                      # Pour chaque neurone d'entrée
                output["outputLayer"]["weightList"][xNeuron].append(float(self.__outputLayerWeightList[xNeuron][inNeuron]))# Ajout du poids
            output["outputLayer"]["biais"].append(float(self.__outputLayerBiais[xNeuron]))                          # Taux d'activation du neurone
            output["outputLayer"]["alpha"].append(float(self.__outputLayerAlpha[xNeuron]))                          # Alpha du neurone

        outputFile = open(filename, "w")
        json.dump(output, outputFile, indent = 4)
        outputFile.close()
        return
    def LoadNetwork(self, filename: str) -> None:
        """
            Chargement du reseau de neurone depuis un fichier
            Params:
                fileame : Nom du fichier source
        """
        inputFile = open(filename)                                                                                              # Ouverture du fichier
        inputData = json.load(inputFile)                                                                                        # Chargement du json
        inputFile.close()                                                                                                       # Fermeture du fichier

        if inputData["nbMaxNeuronPerLayer"] > self.__nbMaxNeuronPerLayer:                                                       # Si le nombre de neurone max par couche du neurone sauvegardé est supérieur au réseau actuel
            raise Exception("Invalid max neuron per layer, this network have "+str(self.__nbMaxNeuronPerLayer)+" but the loaded network have "+str(inputData["nbMaxNeuronPerLayer"])+"\nPlease create a network with max neuron greater or egal to "+str(inputData["nbMaxNeuronPerLayer"]))

        #Paramettrage du reseau
        self.__nbInputNeuron = np.int32(inputData["nbInput"])                                                                   # Nombre de neurone d'entrée
        self.__nbHiddenLayer = 0                                                                                                # Nombre de couche cachée (car on les ajoutent par la suite)
        self.__nbOutputNeuron = np.int32(inputData["nbOutput"])                                                                 # Nombre de neurone de sortie

        #Preparation des couches caché
        self.__hiddenLayerNbNeuron = np.zeros((0), dtype=np.int)                                                                # Nombre de neurone de chaque couche
        self.__hiddenLayerActivationFunction = np.zeros((0), dtype=np.int)                                                      # [ nCouche ]
        self.__hiddenLayerTypeList = np.zeros((0), dtype=np.int)                                                                # [ nCouche ]
        self.__hiddenLayerWeightList = np.zeros((0, self.__nbMaxNeuronPerLayer, self.__nbMaxNeuronPerLayer), dtype=np.float64)  # [ nCouche * [ nNeuron * [ nInput ] ] ]
        self.__hiddenLayerBiais = np.zeros((0, self.__nbMaxNeuronPerLayer), dtype=np.float64)                                   # [ nCouche * [ nNeuron ] ]
        self.__hiddenLayerAlpha = np.ones((0, self.__nbMaxNeuronPerLayer), dtype=np.float64)                                    # [ nCouche * [ nNeuron ] ]

        #Preparation de la couche de sortie
        self.__outputActivationFunction = np.int(inputData["outputLayer"]["activationFunction"])                                # Fonction d'activation des neurones de sortie
        self.__outputLayerType = np.int(inputData["outputLayer"]["type"])                                                       # Type de neurone de sortie
        self.__outputLayerWeightList = np.zeros((self.__nbMaxNeuronPerLayer, self.__nbMaxNeuronPerLayer), dtype=np.float64)     # Liste des poids d'entrée

        #Pour chaque couche caché mettre les informations
        for xLayer in range(int(inputData["nbHiddenLayer"])):                                                                   # Pour chaque couche caché
            #Ajout d'une couche
            self.AddLayer(inputData["hiddenLayers"]["nbNeuron"][xLayer], inputData["hiddenLayers"]["type"][xLayer], inputData["hiddenLayers"]["activationFunction"][xLayer])
            
            #Pour chaque neurone
            for xNeuron in range(inputData["nbMaxNeuronPerLayer"]):                                                                 # Pour chaque neurone de la couche
                #Poids
                for xWeight in range(inputData["nbMaxNeuronPerLayer"]):                                                                 # Pour chaque neurone d'entrée de la couche
                    self.__hiddenLayerWeightList[xLayer][xNeuron][xWeight] = inputData["hiddenLayers"]["weightList"][xLayer][xNeuron][xWeight]  # On change la valeur du poids d'entrée
                
                #Biais
                self.__hiddenLayerBiais[xLayer][xNeuron] = inputData["hiddenLayers"]["biais"][xLayer][xNeuron]                          # On change la valeur du biais

                #Alpha
                self.__hiddenLayerAlpha[xLayer][xNeuron] = inputData["hiddenLayers"]["alpha"][xLayer][xNeuron]                          # On change la valeur de l'alpha
        
        #Couche de sortie
        for xNeuron in range(int(inputData["nbMaxNeuronPerLayer"])):                                                            # Pour chaque neurone de sortie
            for xWeight in range(int(inputData["nbMaxNeuronPerLayer"])):                                                            # Pour chaque neurone d'entrée
                self.__outputLayerWeightList[xNeuron][xWeight] = inputData["outputLayer"]["weightList"][xNeuron][xWeight]               # On enregistre le poids
            self.__outputLayerBiais[xNeuron] = inputData["outputLayer"]["biais"][xNeuron]                                           # On enregistre le biais
            self.__outputLayerAlpha[xNeuron] = inputData["outputLayer"]["alpha"][xNeuron]                                           # On enregistre l'alpha

        return

    def Train(self, dataset: list[list[float]], expected: list[list[float]], nbEpoch: int, learningRate: float, acceleration: bool = False, display: bool = False) -> None:
        """
            Entrainement du reseau de neurone
            Params:
                dataset         :   Liste de liste d'entrées
                expected        :   Liste de liste de sortie attendue en liaison avec la dataset
                nbEpoch         :   Nombre d'entrainement sur toute la dataset
                learningRate    :   Taux d'apprentissage (0 = aucun apprentissage, 1 = Trop d'apprentissage)
                acceleration    :   Accélération materiel
                display         :   Si on affiche
        """
        temp = np.zeros((len(dataset), self.__nbMaxNeuronPerLayer), dtype=np.float64)                           # Table temporaire
        for xData in range(len(dataset)):                                                                       # Pour chaque données d'entrée
            for xValue in range(self.__nbInputNeuron):                                                              # Pour chaque entrée
                temp[xData][xValue] = dataset[xData][xValue]                                                            # Definition de la valeur
        dataset = temp                                                                                          # Definition de la table

        temp = np.zeros((len(expected), self.__nbMaxNeuronPerLayer))                                            # Table temporaire
        for xData in range(len(expected)):                                                                      # Pour chaque données attendue en sortie
            for xValue in range(self.__nbOutputNeuron):                                                             # Pour chaque sortie
                temp[xData][xValue] = expected[xData][xValue]                                                           # Definition de la valeur
        expected = temp                                                                                         # Definition de la table


        if acceleration:                                                                                        # Si on utilise l'accélération materiel
            dataset = np.array(dataset, dtype=np.float64)                                                           # Conversion en tableau numpy
            expected = np.array(expected, dtype=np.float64)                                                         # Conversion en tableau numpy
            self.__hiddenLayerAlpha, self.__hiddenLayerWeightList, self.__hiddenLayerBiais, self.__outputLayerAlpha, self.__outputLayerWeightList, self.__outputLayerBiais = TrainGPU(  # Entrainement
                dataset,
                expected,
                nbEpoch,
                learningRate,
                self.__nbMaxNeuronPerLayer,
                self.__nbInputNeuron,
                self.__nbHiddenLayer,
                self.__hiddenLayerNbNeuron,
                self.__hiddenLayerActivationFunction,
                self.__hiddenLayerTypeList,
                self.__hiddenLayerAlpha,
                self.__hiddenLayerWeightList,
                self.__hiddenLayerBiais,
                self.__nbOutputNeuron,
                self.__outputActivationFunction,
                self.__outputLayerType,
                self.__outputLayerAlpha,
                self.__outputLayerWeightList,
                self.__outputLayerBiais
            )
        else:                                                                                                   # Sinon
            for epoch in range(nbEpoch):                                                                            # Pour chaque epoch
                cout = 0                                                                                                # Cout
                for xInput in range(len(dataset)):                                                                      # Pour chaque entrée
                    self.__FeedForward(dataset[xInput])                                                                     # Calcul de la sortie
                    self.__BackPropagation(expected[xInput])                                                                # Calcul de l'erreur
                    self.__Correction(dataset[xInput], learningRate)                                                        # Correction des poids
                    cout += np.mean(self.__outputLayerCout[:self.__nbOutputNeuron])                                         # Calcul du cout
                if display: print("#", epoch, "/", nbEpoch, " Cout =>", cout)                                           # Affichage du cout

        return
    def Predict(self, inputs: list) -> list:
        """
            Prédiction du reseau de neurone
            Params:
                inputs : Liste des entrées
                acceleration : Acceleration materiel
            Return:
                Liste des valeurs de sortie
        """
        inputs = np.array(inputs, dtype=np.float64)
        inputs.resize((self.__nbMaxNeuronPerLayer), refcheck=False)
        self.__FeedForward(inputs)
        return self.__outputLayerValue[:self.__nbOutputNeuron]


    def __FeedForward(self, inputs: np.ndarray) -> None:
        """
            Propagation des valeurs des neurones
            Params: 
                inputs : Liste des entrées de données
        """
        # HIDDEN LAYER
        for xHiddenLayer in range(self.__nbHiddenLayer):                                                                # Pour chaque couche cachée
            if self.__hiddenLayerTypeList[xHiddenLayer] == LAYER_HIDDEN:                                                    # Si c'est des neurones cachée
                if xHiddenLayer == 0:                                                                                           # Si on est a la premiere couche
                    self.__hiddenLayerValueList[xHiddenLayer] = self.__PreActivation(inputs, self.__hiddenLayerWeightList[xHiddenLayer], self.__nbInputNeuron, self.__hiddenLayerNbNeuron[xHiddenLayer])    # Pre-Activation
                else:                                                                                                           # Sinon
                    self.__hiddenLayerValueList[xHiddenLayer] = self.__PreActivation(self.__hiddenLayerValueList[xHiddenLayer-1], self.__hiddenLayerWeightList[xHiddenLayer], self.__hiddenLayerNbNeuron[xHiddenLayer-1], self.__hiddenLayerNbNeuron[xHiddenLayer]) # Pre-Activation
                self.__hiddenLayerValueList[xHiddenLayer] += self.__hiddenLayerBiais[xHiddenLayer]                              # Ajout des taux d'activations
                self.__hiddenLayerValueList[xHiddenLayer] = ActivationFunction(self.__hiddenLayerActivationFunction[xHiddenLayer], self.__hiddenLayerValueList[xHiddenLayer], False, self.__hiddenLayerAlpha[xHiddenLayer]) # Activation du neurone

        # OUTPUT LAYER
        if self.__outputLayerType == LAYER_OUTPUT:                                                                      # Si c'est des neurones de sortie
            if self.__nbHiddenLayer == 0:                                                                                   # Si il n'y a pas de couche cachée
                self.__outputLayerValue = self.__PreActivation(inputs, self.__outputLayerWeightList, self.__nbInputNeuron, self.__nbOutputNeuron)   # Pre-Activation
            else:                                                                                                           # Sinon
                self.__outputLayerValue = self.__PreActivation(self.__hiddenLayerValueList[-1], self.__outputLayerWeightList, self.__hiddenLayerNbNeuron[-1], self.__nbOutputNeuron)    # Pre-Activation
            self.__outputLayerValue += self.__outputLayerBiais                                                              # Ajout des taux d'activations
            self.__outputLayerValue = ActivationFunction(self.__outputActivationFunction, self.__outputLayerValue, False, self.__outputLayerAlpha)  # Activation des neurones        

        return
    def __BackPropagation(self, expected: np.ndarray) -> None:
        """
            Calcul des taux d'erreur des neurones
            Params:
                expected : Liste des valeurs attendue en sortie
        """        
        if self.__outputLayerType  == LAYER_OUTPUT:                                                                 # Si les neurones sont des neurones de sortie
            self.__outputLayerCout = expected - self.__outputLayerValue                                                 # Calcul de l'erreur
            self.__outputLayerDelta = self.__outputLayerCout * ActivationFunction(self.__outputActivationFunction, self.__outputLayerValue, True, self.__outputLayerAlpha)  # Calcul du taux d'erreur
        
        for xLayer in reversed(range(self.__nbHiddenLayer)):                                                        # Pour chaque couche en partant de la fin
            if self.__hiddenLayerTypeList[xLayer] == LAYER_HIDDEN:                                                      # Si la couche sont des neurones cachée      
                if xLayer == self.__nbHiddenLayer-1:                                                                        # Si la couche actuel est la derniere cachée
                    for xNeuron in range(self.__hiddenLayerNbNeuron[xLayer]):                                                   # Pour chaque neurone de la couche actuel
                        self.__hiddenLayerCoutList[xLayer][xNeuron] = 0                                                             # On reset le cout
                        for xOutputNeuron in range(self.__nbOutputNeuron):                                                          # Pour chaque neurone de sortie
                            self.__hiddenLayerCoutList[xLayer][xNeuron] += self.__outputLayerDelta[xOutputNeuron] * self.__outputLayerWeightList[xOutputNeuron][xNeuron]    # Calcul du taux d'erreur
                else:                                                                                                       # Sinon
                    for xNeuron in range(self.__hiddenLayerNbNeuron[xLayer]):                                                   # Pour chaque neurone de la couche actuel
                        self.__hiddenLayerCoutList[xLayer][xNeuron] = 0                                                             # On reset le cout
                        for xOutputNeuron in range(self.__hiddenLayerNbNeuron[xLayer+1]):                                           # Pour chaque neurone de sortie
                            self.__hiddenLayerCoutList[xLayer][xNeuron] += self.__hiddenLayerDeltaList[xLayer+1][xOutputNeuron] * self.__outputLayerWeightList[xOutputNeuron][xNeuron]  # Calcul du taux d'erreur
                self.__hiddenLayerDeltaList[xLayer] = self.__hiddenLayerCoutList[xLayer] * ActivationFunction(self.__hiddenLayerActivationFunction[xLayer], self.__hiddenLayerValueList[xLayer], True, self.__hiddenLayerAlpha[xLayer]) # Calcul de l'importance de chaque erreur
        return
    def __Correction(self, inputList: np.ndarray, learningRate: float) -> None:
        """
            Correction des poids
            Params:
                inputs          :   Données d'entrée
                learningRate    :   Taux d'apprentissage
        """
        # HIDDEN LAYER
        for xLayer in range(self.__nbHiddenLayer):                                                                          #Pour chaque couche du reseau
            if self.__hiddenLayerTypeList[xLayer] == LAYER_HIDDEN:                                                              # Si la couche sont des neurones cachées
                if xLayer == 0:                                                                                                     # Si c'est la premiere couche
                    for xNeuron in range(self.__hiddenLayerNbNeuron[xLayer]):                                                           # Pour chaque neurones de la couche
                        for xInputNeuron in range(self.__nbInputNeuron):                                                                    # Pour chaque neurone d'entrée
                            self.__hiddenLayerWeightList[xLayer][xNeuron][xInputNeuron] += learningRate * inputList[xInputNeuron] * self.__hiddenLayerDeltaList[xLayer][xNeuron]    # Application de la correction
                        self.__hiddenLayerBiais[xLayer][xNeuron] += learningRate * self.__hiddenLayerDeltaList[xLayer][xNeuron]             # Modification du biais
                else:                                                                                                               # Sinon ce n'est pas la premiere couche
                    for xNeuron in range(self.__hiddenLayerNbNeuron[xLayer]):                                                           # Pour chaque neurones de la couche
                        for xInputNeuron in range(self.__hiddenLayerNbNeuron[xLayer-1]):                                                    # Pour chaque neurone d'entrée
                            self.__hiddenLayerWeightList[xLayer][xNeuron][xInputNeuron] += learningRate * self.__hiddenLayerValueList[xLayer-1][xInputNeuron] * self.__hiddenLayerDeltaList[xLayer][xNeuron]    # Application de la correction
                        self.__hiddenLayerBiais[xLayer][xNeuron] += learningRate * self.__hiddenLayerDeltaList[xLayer][xNeuron]             # Modification du biais
        
        # OUTPUT LAYER
        if self.__outputLayerType == LAYER_OUTPUT:                                                                          # Si la couche de sortie sont des neurones de sortie
            for xNeuron in range(self.__nbOutputNeuron):                                                                        # Pour chaque neurone de la liste actuel
                if self.__nbHiddenLayer == 0:                                                                                       # Si il n'y a pas de couche cachée
                    for xInputNeuron in range(self.__nbInputNeuron):                                                                    # Pour chaque neurone d'entrée de la couche actuel
                        self.__outputLayerWeightList[xNeuron][xInputNeuron] += learningRate * inputList[xInputNeuron] * self.__outputLayerDelta[xNeuron] # Correction du poids
                    self.__outputLayerBiais[xNeuron] += learningRate * self.__outputLayerDelta[xNeuron]                                 # Mise a jour du biais
                else:                                                                                                               # Sinon
                    for xInputNeuron in range(self.__hiddenLayerNbNeuron[-1]):                                                          # Pour chaque neurone d'entrée de la couche actuel
                        self.__outputLayerWeightList[xNeuron][xInputNeuron] += learningRate * self.__hiddenLayerValueList[-1][xInputNeuron] * self.__outputLayerDelta[xNeuron] # Correction du poids
                    self.__outputLayerBiais[xNeuron] += learningRate * self.__outputLayerDelta[xNeuron]                                 # Mise a jour du biais
        return

    def __PreActivation(self, inputList: np.ndarray, weightList: np.ndarray, nbInput: int, nbNeuron: int) -> np.ndarray:
        """
            Pre-Activation du neurone
            Params:
                inputList   :   Liste des valeurs d'entrée
                weightList  :   Liste des poids d'entrée
                nbInput     :   Nombre de neurone d'entrée
                nbNeuron    :   Nombre de neurone de la couche actuel
            Return:
                Liste de valeur
        """
        outputValue = np.zeros((self.__nbMaxNeuronPerLayer), dtype=np.float64)                                                              # Liste de valeur de sortie
        for xNeuron in range(nbNeuron):                                                                                                     # Pour chaque neurone
            for xInput in range(nbInput):                                                                                                       # Pour chaque entrée
                outputValue[xNeuron] += inputList[xInput] * weightList[xNeuron][xInput]                                                             # On additionne le produit du poids et de la valeur
        return outputValue                                                                                                                  # On retourne la liste de valeur

    def __LinkOutputLayerWeight(self) -> None:
        """
            Création des poids entre la couche de sortie et la derniere couche caché (ou l'input)
        """

        newWeightList = 2*np.random.random((self.__nbMaxNeuronPerLayer, self.__nbMaxNeuronPerLayer))-1                                      # Generation de tout les poids de la couche de sortie
        if self.__nbHiddenLayer == 0:                                                                                                       # Si il n'y a pas de couche cachée
            for xNeuron in range(self.__nbOutputNeuron):                                                                                        # Pour chaque neurone de la couche de sortie
                for inNeuron in range(self.__nbInputNeuron, self.__nbMaxNeuronPerLayer):                                                            # Pour chaque poids d'entrée inutile
                    newWeightList[xNeuron][inNeuron] = 0                                                                                                # Remise a 0 du poids inutile
            for xNeuron in range(self.__nbOutputNeuron, self.__nbMaxNeuronPerLayer):                                                            # Pour chaque neurone inutile de la couche
                newWeightList[xNeuron].fill(0)                                                                                                      # Remise a 0 de tout les poids inutiles
        else:                                                                                                                               # Sinon il y a des couches cachée
            for xNeuron in range(self.__nbOutputNeuron):                                                                                        # Pour chaque neurone de la couche de sortie
                for inNeuron in range(self.__hiddenLayerNbNeuron[self.__nbHiddenLayer-1], self.__nbMaxNeuronPerLayer):                              # Pour chaque poids d'entrée inutile
                    newWeightList[xNeuron][inNeuron] = 0                                                                                                # Remise a 0 du poids inutile
            for xNeuron in range(self.__nbOutputNeuron, self.__nbMaxNeuronPerLayer):                                                            # Pour chaque neurone inutile de la couche
                newWeightList[xNeuron].fill(0)                                                                                                      # Remise a 0 de tout les poids inutiles
        self.__outputLayerWeightList = newWeightList                                                                                        # On ajoute la liste des poids
        return

@njit
def TrainGPU(
        dataset: np.ndarray,
        expected: np.ndarray,
        nbEpoch: np.int,
        learningrate: np.float64,

        nbMaxNeuronPerLayer: np.int,
        nbInputNeuron: np.int,

        nbHiddenLayer: np.int,
        hiddenListNbNeuron: np.ndarray,
        hiddenListActivationFunction: np.ndarray,
        hiddenListType: np.ndarray,
        hiddenLayerAlpha: np.ndarray,
        hiddenLayerWeightList: np.ndarray,
        hiddenLayerBiais: np.ndarray,

        nbOutputNeuron: np.ndarray,
        outputActivationFunction: np.ndarray,
        outputLayerType: np.int,
        outputLayerAlpha: np.ndarray,
        outputLayerWeightList: np.ndarray,
        outputLayerBiais: np.ndarray
        ) -> tuple:
    """
        Entrainement au GPU
        Return:
            (
                hiddenLayerAlpha,
                hiddenLayerWeightList,
                hiddenLayerBiais,

                outputLayerAlpha,
                outputLayerWeightList,
                outputLayerBiais
            )
    """
    
    hiddenLayerValueList = np.zeros((nbHiddenLayer, nbMaxNeuronPerLayer), dtype=np.float64)                                                 # [ nCouche * [ nNeuron ] ]
    hiddenLayerCoutList = np.zeros((nbHiddenLayer, nbMaxNeuronPerLayer), dtype=np.float64)                                                  # [ nCouche * [ nNeuron ] ]
    hiddenLayerDeltaList = np.zeros((nbHiddenLayer, nbMaxNeuronPerLayer), dtype=np.float64)                                                 # [ nCouche * [ nNeuron ] ]
    
    outputLayerValue = np.zeros(nbMaxNeuronPerLayer)                                                                                        # Liste des valeur de chaque neurone de sortie
    outputLayerCout = np.zeros(nbMaxNeuronPerLayer)                                                                                         # Liste des erreur de chaque neurone de la liste de sortie
    outputLayerDelta = np.zeros(nbMaxNeuronPerLayer)                                                                                        # Liste des erreur de chaque neurone de la liste de sortie

    for epoch in range(nbEpoch):                                                                                                            # Pour chaque entrainement
        for xInput in range(len(dataset)):                                                                                                      # Pour chaque entrées
            #---FEEDFORWARD---#
            # Hidden layer
            for xLayer in range(nbHiddenLayer):                                                                                                     # Pour chaque couche caché
                if hiddenListType[xLayer] == LAYER_HIDDEN:                                                                                              # Si les nerones de la couche sont des neurones caché
                    if xLayer == 0:                                                                                                                         # Si premiere couche
                        for xNeuron in range(hiddenListNbNeuron[xLayer]):                                                                                       # Pour chaque neurone de la couche
                            hiddenLayerValueList[xLayer][xNeuron] = 0                                                                                               # Reset de la valeur
                            for xWeight in range(nbInputNeuron):                                                                                                    # Pour chaque neurone d'entrée
                                hiddenLayerValueList[xLayer][xNeuron] += dataset[xInput][xWeight] * hiddenLayerWeightList[xLayer][xNeuron][xWeight]                     # Multiplication de l'entrée par le poid
                    else:                                                                                                                                   # Sinon
                        for xNeuron in range(hiddenListNbNeuron[xLayer]):                                                                                       # Pour chaque neurone de la couche
                            hiddenLayerValueList[xLayer][xNeuron] = 0                                                                                               # Reset de la valeur
                            for xWeight in range(hiddenListNbNeuron[xLayer-1]):                                                                                     # Pour chaque neurone d'entrée
                                hiddenLayerValueList[xLayer][xNeuron] += hiddenLayerValueList[xLayer-1][xWeight] * hiddenLayerWeightList[xLayer][xNeuron][xWeight]      # Multiplication de l'entrée par le poid
                    hiddenLayerValueList[xLayer] += hiddenLayerBiais[xLayer]                                                                                # Ajout du taux d'activation
                    for xNeuron in range(hiddenListNbNeuron[xLayer]):                                                                                       # Pour chaque neurone de la couche actuel
                        hiddenLayerValueList[xLayer][xNeuron] = ActivationFunctionGPU(hiddenListActivationFunction[xLayer], hiddenLayerValueList[xLayer][xNeuron], hiddenLayerAlpha[xLayer][xNeuron], False) # Activation du neurone
            
            # Output layer
            if outputLayerType == LAYER_OUTPUT:                                                                                                     # Si les neurones de la couche de sortie sont des neurones de sortie
                if nbHiddenLayer == 0:                                                                                                                  # Si il n'y a pas de couche caché
                    for xNeuron in range(nbOutputNeuron):                                                                                                   # Pour chaque neurone de sortie
                        outputLayerValue[xNeuron] = 0                                                                                                           # Reset du neurone
                        for xWeight in range(nbInputNeuron):                                                                                                    # Pour chaque neurone entrant
                            outputLayerValue[xNeuron] += dataset[xInput][xWeight] * outputLayerWeightList[xNeuron][xWeight]                                         # Pré activation
                else:                                                                                                                                   # Sinon
                    for xNeuron in range(nbOutputNeuron):                                                                                                   # Pour chaque neurone de sortie
                        outputLayerValue[xNeuron] = 0                                                                                                           # Reset du neurone
                        for xWeight in range(hiddenListNbNeuron[-1]):                                                                                           # Pour chaque neurone entrant
                            outputLayerValue[xNeuron] += hiddenLayerValueList[-1][xWeight] * outputLayerWeightList[xNeuron][xWeight]                                # Pré activation
                outputLayerValue += outputLayerBiais                                                                                                        # Ajout du taux d'activation
                for xNeuron in range(nbOutputNeuron):                                                                                                       # Pour chaque neurone
                    outputLayerValue[xNeuron] = ActivationFunctionGPU(outputActivationFunction, outputLayerValue[xNeuron], outputLayerAlpha[xNeuron], False)    # Activation de la couche de sortie            
            #-----------------#

            #---BACK_PROPAGATION---#
            # Output layer
            if outputLayerType == LAYER_OUTPUT:                                                                                                     # Si le neurones de la couche de sortie sont des neurones de sortie
                outputLayerCout = expected[xInput] - outputLayerValue                                                                                   # Calcul de l'erreur
                for xNeuron in range(nbOutputNeuron):                                                                                                   # Pour chaque neurone
                    outputLayerDelta[xNeuron] = outputLayerCout[xNeuron] * ActivationFunctionGPU(outputActivationFunction, outputLayerValue[xNeuron], outputLayerAlpha[xNeuron], True) # Calcul de l'importance de l'erreur

            # Hidden layers
            for xLayer in range(nbHiddenLayer-1, -1, -1):                                                                                           # Pour chaque couche en partant de la fin
                if hiddenListType[xLayer] == LAYER_HIDDEN:                                                                                              # Si la couche actuel est de type caché
                    if xLayer == nbHiddenLayer-1:                                                                                                           # Si la couche actuel est la derniere caché
                        for xNeuron in range(hiddenListNbNeuron[xLayer]):                                                                                       # Pour chaque neurone de la couche
                            hiddenLayerCoutList[xLayer][xNeuron] = 0                                                                                                # Reset du taux d'erreur
                            for xOutput in range(nbOutputNeuron):                                                                                                   # Pour chaque poid de neurone de sortie
                                hiddenLayerCoutList[xLayer][xNeuron] += outputLayerDelta[xOutput] * outputLayerWeightList[xOutput][xNeuron]                             # On multiplie le taux d'erreur du neurone de sortie et le poids le relient
                    else:                                                                                                                                   #Sinon
                        for xNeuron in range(hiddenListNbNeuron[xLayer]):                                                                                       # Pour chaque neurone de la couche
                            hiddenLayerCoutList[xLayer][xNeuron] = 0                                                                                                # Reset du taux d'erreur
                            for xOutput in range(hiddenListNbNeuron[xLayer+1]):                                                                                     # Pour chaque neurone de sortie
                                hiddenLayerCoutList[xLayer][xNeuron] += hiddenLayerDeltaList[xLayer+1][xOutput] * hiddenLayerWeightList[xLayer+1][xOutput][xNeuron]     # On multiplie le taux d'erreur du neurone de sortie et le poids le relient
                    for xNeuron in range(hiddenListNbNeuron[xLayer]):                                                                                       # Pour chaque neurone
                        hiddenLayerDeltaList[xLayer][xNeuron] = hiddenLayerCoutList[xLayer][xNeuron] * ActivationFunctionGPU(hiddenListActivationFunction[xLayer], hiddenLayerValueList[xLayer][xNeuron], hiddenLayerAlpha[xLayer][xNeuron], True) # Calcul de l'importance de chaque erreur
            #----------------------#

            #---CORRECTION---#
            # Hidden layers
            for xLayer in range(nbHiddenLayer):                                                                                                     # Pour chaque couche
                if hiddenListType[xLayer] == LAYER_HIDDEN:                                                                                              # Si la couche actuel est de type caché
                    if xLayer == 0:                                                                                                                         # Si c'est la premiere couche
                        for xNeuron in range(hiddenListNbNeuron[xLayer]):                                                                                       # Pour chaque neurone de la liste actuel
                            for xInputNeuron in range(nbInputNeuron):                                                                                               # Pour chaque neurone d'entrée de la couche actuel
                                hiddenLayerWeightList[xLayer][xNeuron][xInputNeuron] += learningrate * dataset[xInput][xInputNeuron] * hiddenLayerDeltaList[xLayer][xNeuron] # Correction du poids
                            hiddenLayerBiais[xLayer][xNeuron] += learningrate * hiddenLayerDeltaList[xLayer][xNeuron]                                               # Mise a jour du biais
                    else:                                                                                                                                   # Sinon
                        for xNeuron in range(hiddenListNbNeuron[xLayer]):                                                                                       # Pour chaque neurone de la liste actuel
                            for xInputNeuron in range(hiddenListNbNeuron[xLayer-1]):                                                                                # Pour chaque neurone d'entrée de la couche actuel
                                hiddenLayerWeightList[xLayer][xNeuron][xInputNeuron] += learningrate * hiddenLayerValueList[xLayer-1][xInputNeuron] * hiddenLayerDeltaList[xLayer][xNeuron] # Correction du poids
                            hiddenLayerBiais[xLayer][xNeuron] += learningrate * hiddenLayerDeltaList[xLayer][xNeuron]                                               # Mise a jour du biais
                
            # Output layer
            if outputLayerType == LAYER_OUTPUT:                                                                                                     # Si la couche de sortie est de type output
                for xNeuron in range(nbOutputNeuron):                                                                                                   # Pour chaque neurone de la liste actuel
                    if nbHiddenLayer == 0:                                                                                                                  # Si il n'y a pas de couche caché
                        for xInputNeuron in range(nbInputNeuron):                                                                                               # Pour chaque neurone d'entrée de la couche actuel
                            outputLayerWeightList[xNeuron][xInputNeuron] += learningrate * dataset[xInput][xInputNeuron] * outputLayerDelta[xNeuron]                # Correction du poids
                        outputLayerBiais[xNeuron] += learningrate * outputLayerDelta[xNeuron]                                                                   # Mise a jour du biais
                    else:                                                                                                                                   # Sinon
                        for xInputNeuron in range(hiddenListNbNeuron[-1]):                                                                                      # Pour chaque neurone d'entrée de la couche actuel
                            outputLayerWeightList[xNeuron][xInputNeuron] += learningrate * hiddenLayerValueList[-1][xInputNeuron] * outputLayerDelta[xNeuron]       # Correction du poids
                        outputLayerBiais[xNeuron] += learningrate * outputLayerDelta[xNeuron]                                                                   # Mise a jour du biais
            #----------------#
    return hiddenLayerAlpha, hiddenLayerWeightList, hiddenLayerBiais, outputLayerAlpha, outputLayerWeightList, outputLayerBiais

