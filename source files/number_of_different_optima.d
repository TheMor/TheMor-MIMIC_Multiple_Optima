module number_of_different_optima;


import mimic_looking_for_more_optima;


void runTest(const size_t mu, const size_t lambda, const FitnessFunction fitnessFunction,
    const size_t maximumNumberOfIterations, const size_t numberOfTrials, const string path)
{
    const n = fitnessFunction.dimension;

    import std.stdio : File;
    import std.conv : text;
    const fileName = path ~ "n=" ~ n.text ~ ".txt";
    auto file = File(fileName, "w");
    const dataSeparator = "::";
    file.writeln("success", dataSeparator, "lambda", dataSeparator, "numberOfTotalFitnessEvaluations", dataSeparator,
        "bestIndividuals", dataSeparator, "numberOfOptimaInIteration");
    file.close;
    scope(exit)
    {
        file.close;
    }

    import std.parallelism : parallel;
    import std.range : iota;
    foreach(_; iota(0, numberOfTrials).parallel)
    {
        auto mimic = new MIMIC(mu, lambda, fitnessFunction, maximumNumberOfIterations);
        auto result = mimic.run;
        synchronized
        {
            file.open(fileName, "a");
            file.writeln(result.succeeded, dataSeparator, lambda, dataSeparator, 2 * lambda * result.numberOfIterations,
                dataSeparator, result.bestIndividuals, dataSeparator, result.numberOfOptimaInIteration);
        }
    }
}

void main()
{
    import std.path : dirSeparator;

    // Declare all important variables here.
    const directoryName = "UBM" ~ dirSeparator ~ "additionalOptima";
    const nMin = 50;
    const nMax = 200;
    const nStepSize = 10;
    const numberOfTrials = 100;
    const maximumNumberOfIterations = 50_000;
    // Do not touch the rest.

    const pathPrefix = ".." ~ dirSeparator ~ "tests"~ dirSeparator;
    const path = pathPrefix ~ directoryName ~ dirSeparator;

    // Create the directory if needed.
    import std.file : mkdirRecurse;
    path.mkdirRecurse;

    // Generate the test cases.
    import std.parallelism : parallel;
    import std.range : iota;
    foreach(n; iota(nMin, nMax + 1, nStepSize).parallel)
    {
        import std.math : E, log, sqrt;
        const lambda = cast(int) (12 * n * log(n));
        const mu = cast(int) (lambda / 8); // 1/8 = 2(1/4)^2

        // For LeadingUniformBlocks.
        const UBM = createUniformBlocksMaxWithBlockSize(2, n);

        runTest(mu, lambda, UBM, maximumNumberOfIterations, numberOfTrials, path);
    }
}