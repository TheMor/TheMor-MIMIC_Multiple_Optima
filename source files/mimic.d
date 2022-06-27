module mimic;


import std.container.rbtree;
import std.typecons : Tuple;

public import fitness_functions;


class MIMIC
{
private:
    const size_t mu;
    const size_t lambda;
    const FitnessFunction fitnessFunction;
    const size_t maximumNumberOfIterations;
    
    Tuple!(bool, "succeeded", Individual, "bestIndividual", size_t, "bestIteration", size_t, "numberOfIterations",
        size_t[], "permutation", double[][], "probabilities", bool, "tooLow", Individual, "bestNow") returnTuple;
    size_t[] permutation;
    // Each entry denotes the probabilities to sample a 1 conditional that the bit before was 0 or 1.
    double[][] probabilities;
    
public:
    this(const size_t mu, const size_t lambda, const FitnessFunction fitnessFunction,
         const size_t maximumNumberOfIterations = size_t.max)
    {
        assert(mu >= 1, "The parameter mu needs to be at least 1");
        assert(lambda >= 1, "The parameter lambda needs to be at least 1");
        assert(mu <= lambda, "The parameter mu is larger than lambda.");
        
        this.mu = mu;
        this.lambda = lambda;
        this.fitnessFunction = fitnessFunction;
        this.maximumNumberOfIterations = maximumNumberOfIterations;
        
        this.permutation = new size_t[this.fitnessFunction.dimension];
        this.probabilities = new double[][](this.fitnessFunction.dimension, 2);
    }
    
    auto getMu() const
    {
        return this.mu;
    }
    
    auto getLambda() const
    {
        return this.lambda;
    }
    
    auto run()
    {
        auto currentNumberOfIterations = 0;
        auto population = this.generateInitialPopulation;
        
        import std.algorithm.sorting : sort;
        population.sort!((a, b) => this.compareIndividuals(a, b));
        
        auto bestSoFar = population[0].dup;
        auto bestIteration = 0;
        auto probabilityTooLow = false;
        
        while(!this.foundOptimum(population) && (currentNumberOfIterations < this.maximumNumberOfIterations)
              && !probabilityTooLow)
        {
            population = population[0 .. this.mu];
            
            import std.range : iota;
            // The set contains all of the indices that are not part of the permutation yet.
            auto setOfIndices = redBlackTree(iota(this.fitnessFunction.dimension));
            // The index denotes the index in the permutation at which a new element needs to be inserted.
            size_t currentIndexOfPermutation = 0;
            
            this.determineFirstElementOfPermutation(population, setOfIndices, currentIndexOfPermutation);
            
            while(!setOfIndices.empty)
            {
                this.determineNextElementOfPermutation(population, setOfIndices, currentIndexOfPermutation);
            }
            
            this.roundProbabilities(probabilityTooLow);
            
            population = this.generateNewPopulation;
            population.sort!((a, b) => this.compareIndividuals(a, b));
            if(this.compareIndividuals(population[0], bestSoFar))
            {
                bestSoFar = population[0].dup;
                bestIteration = currentNumberOfIterations;
            }
            
            currentNumberOfIterations++;
        }
        
        this.returnTuple.succeeded = this.foundOptimum(population);
        this.returnTuple.bestIndividual = bestSoFar;
        this.returnTuple.bestIteration = bestIteration;
        this.returnTuple.numberOfIterations = currentNumberOfIterations;
        this.returnTuple.permutation = this.permutation;
        this.returnTuple.probabilities = this.probabilities;
        this.returnTuple.tooLow = probabilityTooLow;
        this.returnTuple.bestNow = population[0].dup;
        
        return this.returnTuple;
    }
    
private:
    Individual[] generateInitialPopulation() const
    {
        Individual generateIndividual()
        {
            auto individual = Individual(new bool[this.fitnessFunction.dimension]);
            
            import std.random : uniform;
            foreach(ref bit; individual)
            {
                bit = cast(bool) uniform(0, 2);
            }
            
            return individual;
        }
        
        auto population = new Individual[this.lambda];
        
        foreach(ref individual; population)
        {
            individual = generateIndividual;
        }
        
        return population;
    }
    
    Individual[] generateNewPopulation() const
    {
        Individual generateIndividualFromPermutation()
        {
            auto individual = Individual(new bool[this.fitnessFunction.dimension]);
            
            import std.random : uniform01;
            
            // Determine the first bit.
            auto currentIndex = this.permutation[0];
            auto currentBit = 0;
            if(uniform01 < this.probabilities[currentIndex][0])
            {
                currentBit = 1;
            }
            
            individual[currentIndex] = cast(bool) currentBit;
            
            // Determine the other bits.
            foreach(index; this.permutation[1 .. $])
            {
                auto previousBit = currentBit;
                currentBit = 0;
                if(uniform01 < this.probabilities[index][previousBit])
                {
                    currentBit = 1;
                }
                
                individual[index] = cast(bool) currentBit;
            }
            
            return individual;
        }
        
        auto population = new Individual[this.lambda];
        
        foreach(ref individual; population)
        {
            individual = generateIndividualFromPermutation;
        }
        
        return population;
    }
    
    bool compareIndividuals(const Individual individual1, const Individual individual2) const
    {
        auto fitness = &this.fitnessFunction.evaluate;
        
        return fitness(individual1) > fitness(individual2);
    }
    
    bool foundOptimum(Individual[] sortedPopulation) const
    {
        return this.fitnessFunction.evaluate(sortedPopulation[0]) == this.fitnessFunction.optimalFitnessValue;
    }
    
    double extendedBinaryLogarithm(double numerus) const
    {
        if(numerus == 0.0)
        {
            return 0.0;
        }
        else
        {
            import std.math : log2;
            return log2(numerus);
        }
    }
    
    double entropy(const size_t index, const Individual[] population) const
    {
        assert(population.length == this.mu, "The population does not have size mu.");
        assert(index < population[0].length, "The index is too large.");
        
        auto numberOfOnes = this.numberOfOnesAtPosition(index, population);
        auto fractionOfOnes = (cast(double) numberOfOnes) / this.mu;
        auto fractionOfZeros = 1.0 - fractionOfOnes;
        
        return - fractionOfOnes * this.extendedBinaryLogarithm(fractionOfOnes)
               - fractionOfZeros * this.extendedBinaryLogarithm(fractionOfZeros);
    }
    
    double conditionalEntropy(const size_t index, const size_t conditionIndex, const Individual[] population) const
    {
        assert(population.length == this.mu, "The population does not have size mu.");
        assert(index < population[0].length, "The index is too large.");
        assert(conditionIndex < population[0].length, "The condition index is too large.");
        
        // Number of 1s and 0s at the condition index.
        auto numberOfOnes = this.numberOfOnesAtPosition(conditionIndex, population);
        auto numberOfZeros = this.mu - numberOfOnes;
        
        // The number of all four intersection combinations.
        // The first number denotes the number of the current index, the second of the condition.
        auto numberOfOneOnes = this.numberOfOneOnesAtPositions(index, conditionIndex, population);
        auto numberOfOneZeros = this.numberOfOneZerosAtPositions(index, conditionIndex, population);
        
        // Calculate the respective probabilities.
        auto fractionOfOnes = (cast(double) numberOfOnes) / this.mu;
        auto fractionOfZeros = 1.0 - fractionOfOnes;
        
        // These are all conditional probabilities!
        auto fractionOfOneOnes = (numberOfOnes == 0.0) ? 0.0 : (cast(double) numberOfOneOnes) / numberOfOnes;
        auto fractionOfZeroOnes = 1.0 - fractionOfOneOnes;
        
        auto fractionOfOneZeros = (numberOfZeros == 0.0) ? 0.0 : (cast(double) numberOfOneZeros) / numberOfZeros;
        auto fractionOfZeroZeros = 1.0 - fractionOfOneZeros;
        
        // Make sure to get the unconditional probabilities from the conditional ones.
        return - fractionOfZeroZeros * fractionOfZeros * this.extendedBinaryLogarithm(fractionOfZeroZeros)
               - fractionOfOneZeros * fractionOfZeros * this.extendedBinaryLogarithm(fractionOfOneZeros)
               - fractionOfZeroOnes * fractionOfOnes * this.extendedBinaryLogarithm(fractionOfZeroOnes)
               - fractionOfOneOnes * fractionOfOnes * this.extendedBinaryLogarithm(fractionOfOneOnes);
    }
    
    int numberOfOnesAtPosition(const size_t index, const Individual[] population) const
    {
        assert(population.length == this.mu, "The population does not have size mu.");
        assert(index < population[0].length, "The index is too large.");
        
        auto numberOfOnes = 0;
        foreach(const ref individual; population)
        {
            numberOfOnes += individual[index];
        }
        
        return numberOfOnes;
    }
    
    int numberOfOneOnesAtPositions(const size_t index, const size_t conditionIndex,
        const Individual[] population) const
    {
        assert(population.length == this.mu, "The population does not have size mu.");
        assert(index < population[0].length, "The index is too large.");
        assert(conditionIndex < population[0].length, "The condition index is too large.");
        
        auto numberOfOneOnes = 0;
        foreach(const ref individual; population)
        {
            if((individual[index] == 1) && (individual[conditionIndex] == 1))
            {
                numberOfOneOnes++;
            }
        }
        
        return numberOfOneOnes;
    }
    
    int numberOfOneZerosAtPositions(const size_t index, const size_t conditionIndex,
        const Individual[] population) const
    {
        assert(population.length == this.mu, "The population does not have size mu.");
        assert(index < population[0].length, "The index is too large.");
        assert(conditionIndex < population[0].length, "The condition index is too large.");
        
        auto numberOfOneZeros = 0;
        foreach(const ref individual; population)
        {
            if((individual[index] == 1) && (individual[conditionIndex] == 0))
            {
                numberOfOneZeros++;
            }
        }
        
        return numberOfOneZeros;
    }
    
    void determineFirstElementOfPermutation(const Individual[] population,
        RedBlackTree!(size_t, "a < b", false) setOfIndices, ref size_t currentIndexOfPermutation)
    {
        size_t minIndex = size_t.max;
        auto minimumEntropy = double.infinity;
        
        // Shuffle the set in order to generate uniform random tie-breaking.
        import std.random : randomShuffle;
        import std.range : array;
        foreach(const index; setOfIndices.array.randomShuffle)
        {
            auto currentEntropy = this.entropy(index, population);
            if(currentEntropy < minimumEntropy)
            {
                minIndex = index;
                minimumEntropy = currentEntropy;
            }
        }
        
        this.permutation[currentIndexOfPermutation] = minIndex;
        setOfIndices.removeKey(minIndex);
        currentIndexOfPermutation++;
        
        // Calculate the probability of the first position.
        auto numberOfOnes = this.numberOfOnesAtPosition(minIndex, population);
        this.probabilities[minIndex][0] = (cast(double) numberOfOnes) / this.mu;
        this.probabilities[minIndex][1] = this.probabilities[minIndex][0];
    }
    
    void determineNextElementOfPermutation(const Individual[] population,
        RedBlackTree!(size_t, "a < b", false) setOfIndices, ref size_t currentIndexOfPermutation)
    {
        assert(currentIndexOfPermutation > 0, "Current index is still at 0.");
        
        auto prevIndex = this.permutation[currentIndexOfPermutation - 1];
        
        size_t minIndex = size_t.max;
        auto minimumEntropy = double.infinity;
        
        // Shuffle the set in order to generate uniform random tie-breaking.
        import std.random : randomShuffle;
        import std.range : array;
        foreach(const index; setOfIndices.array.randomShuffle)
        {
            auto currentEntropy = this.conditionalEntropy(index, prevIndex, population);
            if(currentEntropy < minimumEntropy)
            {
                minIndex = index;
                minimumEntropy = currentEntropy;
            }
        }
        
        this.permutation[currentIndexOfPermutation] = minIndex;
        setOfIndices.removeKey(minIndex);
        currentIndexOfPermutation++;
        
        // Calculate the probability of the current position.
        // Number of 1s and 0s at the condition index.
        auto numberOfOnes = this.numberOfOnesAtPosition(prevIndex, population);
        auto numberOfZeros = this.mu - numberOfOnes;
        
        // The first number denotes the number of the current index, the second of the condition.
        auto numberOfOneOnes = this.numberOfOneOnesAtPositions(minIndex, prevIndex, population);
        auto numberOfOneZeros = this.numberOfOneZerosAtPositions(minIndex, prevIndex, population);
        
        // Calculate the conditional probabilities to sample a 1.
        this.probabilities[minIndex][0] = (numberOfZeros == 0.0) ? 0.5 :
                                                                   (cast(double) numberOfOneZeros) / numberOfZeros;
        this.probabilities[minIndex][1] = (numberOfOnes == 0.0) ? 0.5 :
                                                                  (cast(double) numberOfOneOnes) / numberOfOnes;
    }
    
    void roundProbabilities(ref bool probabilityTooLow)
    {
        auto lowerBorder = 1.0 / this.fitnessFunction.dimension;
        auto upperBorder = 1.0 - lowerBorder;
        
        foreach(firstIndex; 0 .. this.permutation.length)
        {
            foreach(secondIndex; 0 .. 2)
            {
                auto probability = this.probabilities[firstIndex][secondIndex];
                
                if((secondIndex == 1) && (probability < 0.25))
                {
                    probabilityTooLow = true;
                }
                
                if(probability > upperBorder)
                {
                    probability = upperBorder;
                }
                if(probability < lowerBorder)
                {
                    probability = lowerBorder;
                }
                
                this.probabilities[firstIndex][secondIndex] = probability;
            }
        }
    }
}