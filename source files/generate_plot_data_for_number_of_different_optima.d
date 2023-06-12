module generate_plot_data_for_number_of_different_optima;


void evaluateModels(const string inputDataPath, const string outputDataPath, const int[] nValues,
    const int numberOfTrials)
{
    const dataSeparator = "::";

    const lowerWhisker = 0.1;
    const lowerQuantile = 0.25;
    const upperWhisker = 1.0 - lowerWhisker;
    const upperQuantile = 1.0 - lowerQuantile;

    const lowerWhiskerIndex = cast(int)(lowerWhisker * numberOfTrials);
    const upperWhiskerIndex = cast(int)(upperWhisker * numberOfTrials);
    const lowerQuantileIndex = cast(int)(lowerQuantile * numberOfTrials);
    const upperQuantileIndex = cast(int)(upperQuantile * numberOfTrials);
    const medianIndex = numberOfTrials / 2;

    import std.stdio : File;
    auto outputFile = File(outputDataPath ~ "numberOfDifferentOptimaBoxPlotData.txt", "w");
    scope(exit)
    {
        outputFile.close;
    }

    outputFile.writeln("n lw lq med uq uw uniqueness_ratio");

    import std.algorithm : map, count, sort;
    import std.array : array, split;
    import std.conv : text, to;
    import std.file : readText;
    import std.range : dropOne, zip;
    import std.string : splitLines;

    auto data = (inputDataPath ~ "evaluationOfNumberOfDifferentOptima.txt").readText
        .splitLines
        .dropOne
        .map!(line => line.split(dataSeparator).map!(to!(int[])))
        .array;

    foreach(index, n; nValues)
    {
        const allOptima = data[index][0];
        auto allUniqueOptima = data[index][1];
        const numberOfDataPoints = allOptima.length;

        // Calculate how many of the runs had 100 % unique optima.
        const numberOfAllUniques = zip(allOptima, allUniqueOptima).count!(pair => pair[0] == pair[1]);

        // Sort the unique optima in order to read the boxplot data.
        allUniqueOptima.sort();

        outputFile.writeln
            (
                n, " ",
                allUniqueOptima[lowerWhiskerIndex], " ",
                allUniqueOptima[lowerQuantileIndex], " ",
                allUniqueOptima[medianIndex], " ",
                allUniqueOptima[upperQuantileIndex], " ",
                allUniqueOptima[upperWhiskerIndex], " ",
                cast(int)(100 * cast(double)(numberOfAllUniques) / numberOfTrials)
            );
    }
}

void main()
{
    import std.path : dirSeparator;

    // Declare all important variables here.
    const dataFolderName = "UBM" ~ dirSeparator ~ "additionalOptima";
    const nValues = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200];
    const numberOfTrials = 100;
    // Do not touch the rest.

    const inputDataPath = ".." ~ dirSeparator ~ "data" ~ dirSeparator ~ dataFolderName ~ dirSeparator;
    const outputDataPath = ".." ~ dirSeparator ~ "plots" ~ dirSeparator ~ dataFolderName ~ dirSeparator;

    // Create the directory if needed.
    import std.file : mkdirRecurse;
    outputDataPath.mkdirRecurse();

    evaluateModels(inputDataPath, outputDataPath, nValues, numberOfTrials);
}