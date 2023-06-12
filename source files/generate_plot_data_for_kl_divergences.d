module generate_plot_data_for_kl_divergences;


void evaluateModels(const string inputDataPath, const string outputDataPath, const int[] nValues)
{
    const dataSeparator = "::";

    const lowerWhisker = 0.1;
    const lowerQuantile = 0.25;
    const upperWhisker = 1.0 - lowerWhisker;
    const upperQuantile = 1.0 - lowerQuantile;

    foreach(n; nValues)
    {
        import std.conv : text;
        import std.stdio : File;
        auto nString = "n=" ~ n.text;
        auto outputFile = File(outputDataPath ~ "klDivergencePlotData_" ~ nString ~ ".txt", "w");
        scope(exit)
        {
            outputFile.close;
        }

        outputFile.writeln("iteration lw lq med uq uw");

        import std.algorithm : map, count, sort;
        import std.array : array, split;
        import std.conv : text, to;
        import std.file : readText;
        import std.range : dropOne, zip;
        import std.string : splitLines;

        auto data = (inputDataPath ~ nString ~ ".txt").readText
            .splitLines
            .dropOne
            .map!(line => line.split(dataSeparator).dropOne.map!(to!(double[]))[0])
            .array;

        double[][] klsInIteration;
        foreach(divergences; data)
        {
            foreach(i, kl; divergences)
            {
                if(i < klsInIteration.length)
                {
                    klsInIteration[i] ~= kl;
                }
                else
                {
                    klsInIteration ~= [kl];
                }
            }
        }

        foreach(i, kls; klsInIteration)
        {
            auto numberOfTrials = kls.length;

            kls.sort;
            const lowerWhiskerIndex = cast(int)(lowerWhisker * numberOfTrials);
            const upperWhiskerIndex = cast(int)(upperWhisker * numberOfTrials);
            const lowerQuantileIndex = cast(int)(lowerQuantile * numberOfTrials);
            const upperQuantileIndex = cast(int)(upperQuantile * numberOfTrials);
            const medianIndex = numberOfTrials / 2;

            outputFile.writeln
                (
                    i + 1, " ",
                    kls[lowerWhiskerIndex], " ",
                    kls[lowerQuantileIndex], " ",
                    kls[medianIndex], " ",
                    kls[upperQuantileIndex], " ",
                    kls[upperWhiskerIndex], " ",
                );
        }
    }
}

void main()
{
    import std.path : dirSeparator;

    // Declare all important variables here.
    const dataFolderName = "UBM" ~ dirSeparator ~ "klDivergence";
    const nValues = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200];
    // Do not touch the rest.

    const inputDataPath = ".." ~ dirSeparator ~ "tests" ~ dirSeparator ~ dataFolderName ~ dirSeparator;
    const outputDataPath = ".." ~ dirSeparator ~ "plots" ~ dirSeparator ~ dataFolderName ~ dirSeparator;

    // Create the directory if needed.
    import std.file : mkdirRecurse;
    outputDataPath.mkdirRecurse();

    evaluateModels(inputDataPath, outputDataPath, nValues);
}