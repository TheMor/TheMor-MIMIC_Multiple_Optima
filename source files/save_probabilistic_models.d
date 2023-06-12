module save_probabilistic_models;


import mimic;


void runTest(const size_t mu, const size_t lambda, const FitnessFunction fitnessFunction,
    const size_t maximumNumberOfIterations, const size_t numberOfTrials, const string path)
{
    const n = fitnessFunction.dimension;

    import std.stdio : File;
    import std.conv : text;
    const fileName = path ~ "n=" ~ n.text ~ ".txt";
    auto file = File(fileName, "w");
    const dataSeparator = "::";
    file.writeln("success", dataSeparator, "permutation", dataSeparator, "probabilities");
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
            file.writeln(result.succeeded, dataSeparator, result.permutation, dataSeparator, result.probabilities);
        }
    }
}

void main()
{
    import std.path : dirSeparator;
    import std.math : E, log;

    // Declare all important variables here.
    const directoryName = "UBM" ~ dirSeparator ~ "models";
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
        const lambda = cast(int) (12 * n * log(n));
        const mu = cast(int) (lambda / 8);
        const UBM = createUniformBlocksMaxWithBlockSize(2, n);

        runTest(mu, lambda, UBM, maximumNumberOfIterations, numberOfTrials, path);
    }
}