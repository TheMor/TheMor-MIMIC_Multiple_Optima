module evaluate_number_of_different_optima;


alias Individual = int[];


void evaluateModels(const string inputDataPath, const string outputDataPath, const int[] nValues,
    const int numberOfTrials)
{
    const dataSeparator = "::";

    import std.stdio : File;
    auto outputFile = File(outputDataPath ~ "evaluationOfNumberOfDifferentOptima.txt", "w");
    outputFile.writeln("numerOfOptima", dataSeparator, "numberOfUniqueOptima");

    foreach(n; nValues)
    {
        import std.algorithm : map, filter, sort, uniq;
        import std.array : array, split;
        import std.conv : text, to;
        import std.file : readText;
        import std.range : dropOne, walkLength;
        import std.string : splitLines;

        auto data = (inputDataPath ~ "n=" ~ n.text ~ ".txt").readText
            .splitLines
            .dropOne
            .map!(line => line.split(dataSeparator))
            .filter!(line => line[0].to!(bool) == true)   // Only get the data where an optimum was found.
            .array;
        const numberOfSuccesses = data.length;

        auto optimalIndividualSets = new Individual[][](numberOfSuccesses);
        auto cardinalitiesOfOptimalSets = new size_t[](numberOfSuccesses);
        auto cardinalitiesOfUniqueOptimalSets = new size_t[](numberOfSuccesses);
        foreach(runIndex; 0 .. numberOfSuccesses)
        {
            auto optimalIndividuals = data[runIndex][1].to!(Individual[]).dup;

            cardinalitiesOfOptimalSets[runIndex] = optimalIndividuals.length;
            optimalIndividualSets[runIndex] = optimalIndividuals;

            // Remove duplicates by sorting and then removing consecutive duplicates.
            cardinalitiesOfUniqueOptimalSets[runIndex] = optimalIndividuals.sort!(compareIndividuals)
                .uniq
                .walkLength;
        }

        outputFile.writeln(cardinalitiesOfOptimalSets, dataSeparator, cardinalitiesOfUniqueOptimalSets);
    }

    outputFile.close;
}

// Returns true if the first individual is lexicographically smaller, false otherwise.
bool compareIndividuals(const ref Individual individual1, const ref Individual individual2)
{
    assert(individual1.length == individual2.length, "The individuals to compare do not have equal length.");

    foreach(index; 0 .. individual1.length)
    {
        if(individual1[index] < individual2[index])
        {
            return true;
        }
        if(individual1[index] > individual2[index])
        {
            return false;
        }
    }

    return false;
}

void main()
{
    import std.path : dirSeparator;

    // Declare all important variables here.
    const dataFolderName = "UBM" ~ dirSeparator ~ "additionalOptima";
    const nValues = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200];
    const numberOfTrials = 100;
    // Do not touch the rest.

    const inputDataPath = ".." ~ dirSeparator ~ "tests" ~ dirSeparator ~ dataFolderName ~ dirSeparator;
    const outputDataPath = ".." ~ dirSeparator ~ "data" ~ dirSeparator ~ dataFolderName ~ dirSeparator;

    // Create the directory if needed.
    import std.file : mkdirRecurse;
    outputDataPath.mkdirRecurse();

    evaluateModels(inputDataPath, outputDataPath, nValues, numberOfTrials);
}