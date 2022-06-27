module fitness_functions;


import std.bitmanip : BitArray;

alias Individual = BitArray;
alias FitnessFunctionReturnType = size_t;
alias FitnessEvaluator = FitnessFunctionReturnType delegate(const Individual);


class FitnessFunction
{
private:
    const size_t n;
    const FitnessEvaluator evaluator;
    const FitnessFunctionReturnType maximumFitness;
    
public:
    this(const size_t n, const FitnessEvaluator evaluator, const FitnessFunctionReturnType maximumFitness)
    {
        this.n = n;
        this.evaluator = evaluator;
        this.maximumFitness = maximumFitness;
    }
    
    FitnessFunctionReturnType evaluate(const Individual individual) const
    {
        assert(individual.length == this.n, "The length of the individual does not match the dimension of the function.");
        
        return this.evaluator(individual);
    }
    
    size_t dimension() const
    {
        return this.n;
    }
    
    FitnessFunctionReturnType optimalFitnessValue() const
    {
        return this.maximumFitness;
    }
}

// Function for creating a UniformBlocksMax function.
FitnessFunction createUniformBlocksMaxWithBlockSize(const size_t k, const size_t n)
{
    // Returns whether a block of size k is all 1s or all 0s.
    bool hasUniformBlockAtIndex(const Individual individual, const size_t startIndex)
    {
        assert(startIndex < individual.length, "The start index is too large.");
        
        auto sumOfOnesInCurrentBlock = 0;
        
        foreach(index; 0 .. k)
        {
            sumOfOnesInCurrentBlock += individual[startIndex + index];
        }
        
        return (sumOfOnesInCurrentBlock == k) || (sumOfOnesInCurrentBlock == 0);
    }
    
    FitnessFunctionReturnType uniformBlocksMax(const Individual individual)
    {
        assert(k <= individual.length, "The individual is shorter than the block length.");
        assert(individual.length % k == 0, "The length of the individual is not a multiple of the block size.");
        
        FitnessFunctionReturnType numberOfBlocks = individual.length / k;
        FitnessFunctionReturnType numberOfCorrectBlocks = 0;
        
        // Determine the number of correct blocks.
        foreach(blockIndex; 0 .. numberOfBlocks)
        {
            const firstIndex = k * blockIndex;
            
            if(hasUniformBlockAtIndex(individual, firstIndex))
            {
                numberOfCorrectBlocks++;
            }
        }
        
        return numberOfCorrectBlocks;
    }
    
    return new FitnessFunction(n, &uniformBlocksMax, n / k);
}