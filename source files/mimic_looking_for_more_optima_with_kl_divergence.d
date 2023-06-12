module mimic_looking_for_more_optima_with_kl_divergence;


import std.array : array;
import std.container.rbtree;
import std.range : iota;
import std.typecons : Tuple;

public import fitness_functions;


class MIMIC
{
private:
    const size_t mu;
    const size_t lambda;
    const FitnessFunction fitnessFunction;
    const size_t maximumNumberOfIterations;

    Tuple!(bool, "succeeded", bool, "tooLow", double[], "divergences") returnTuple;
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

        this.permutation = iota(this.fitnessFunction.dimension).array;
        this.probabilities = new double[][](this.fitnessFunction.dimension, 2);
        foreach(ref pair; probabilities)
        {
            pair = [0.5, 0.5];
        }
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
        auto probabilityTooLow = false;
        double[] divergences = [];

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
            }

            currentNumberOfIterations++;
            divergences ~= this.klDivergence;
        }

        // Search for additional optima if an optimum was found in the first place.
        if(this.foundOptimum(population))
        {
            foreach(_; 0 .. currentNumberOfIterations)
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
                }

                divergences ~= this.klDivergence;
            }
        }

        this.returnTuple.succeeded = this.foundOptimum(population);
        this.returnTuple.tooLow = probabilityTooLow;
        this.returnTuple.divergences = divergences;

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

    // Functions for calculating the KL divergence.
    double firstSum() const
    {
        const n = this.fitnessFunction.dimension;
        const margin = 1.0 / n;

        // The probability that x_i and x_j occur with the stated values. (We assume that j = i - 1,
        // since we are in an ideal model and use the identity as permutation.)
        double intersectionProbability(size_t i, byte x_i, byte x_j)
        {
            if(i == 0) // The unconditional probability of x_i occurring, divided by 2 (since it occurs twice).
                       // It occurs once for x_j = 0 and once for x_j = 1.
            {
                return 0.25;
            }

            const j = i - 1;
            if((i / 2) == (j / 2)) // The indices are in the same block
            {
                return 0.5 * ((x_i == x_j) ? 1 - margin : margin);
            }
            return 0.25;
        }

        // The probability that x_i has the stated value, conditional that x_j has the stated value.
        // (We assume that j = i - 1, as above.)
        double conditionalProbability(size_t i, byte x_i, byte x_j)
        {
            if(i == 0) // The unconditional probability of x_i occurring.
            {
                return 0.5;
            }

            const j = i - 1;
            if((i / 2) == (j / 2)) // The indices are in the same block
            {
                return (x_i == x_j) ? 1 - margin : margin;
            }
            return 0.5;
        }

        // Compute the sum.
        double sum = 0.0;
        foreach(i; 0 .. n)
        {
            foreach(byte x_i; 0 .. 2)
            {
                foreach(byte x_j; 0 .. 2)
                {
                    sum += intersectionProbability(i, x_i, x_j)
                        * this.extendedBinaryLogarithm(conditionalProbability(i, x_i, x_j));
                }
            }
        }
        return sum;
    }

    double secondSum() const
    {
        const n = this.fitnessFunction.dimension;
        const margin = 1.0 / n;

        // The probability that x_i and x_j occur with the stated values.
        double intersectionProbability(size_t i, byte x_i, int j, byte x_j)
        {
            if(j == -1) // The unconditional probability of x_i occurring, divided by 2 (since it occurs twice).
            {
                return 0.25;
            }

            if((i / 2) == (j / 2)) // The indices are in the same block
            {
                return 0.5 * ((x_i == x_j) ? 1.0 - margin : margin);
            }
            return 0.25;
        }

        double conditionalProbability(size_t i, byte x_i, int j, byte x_j)
        {
            const p_i = this.probabilities[i][0];
            if(j == -1) // The unconditional probability of x_i occurring.
            {
                return (x_i == 1) ? p_i : 1.0 - p_i;
            }

            const p_ij = this.probabilities[i][x_j];
            return (x_i == 1) ? p_ij : 1.0 - p_ij;
        }

        // Compute the sum.
        double sum = 0.0;
        foreach(index; 0 .. this.fitnessFunction.dimension)
        {
            foreach(byte x_i; 0 .. 2)
            {
                foreach(byte x_j; 0 .. 2)
                {
                    const i = this.permutation[index];
                    const j = cast(int) ((index != 0) ? this.permutation[index - 1] : -1);

                    sum += intersectionProbability(i, x_i, j, x_j)
                        * this.extendedBinaryLogarithm(conditionalProbability(i, x_i, j, x_j));
                }
            }
        }
        return sum;
    }

    // Does not check for absolute continuity! That is, if the current probability distribution has
    // events that occur with probability 0 which occur with positive probability in an ideal model,
    // then the uotput of the function is incorrect (that is, not positive infinity). However, due
    // to the margins, this cannot happen here.
    double klDivergence() const
    {
        return this.firstSum - this.secondSum;
    }
}