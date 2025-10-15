/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

// Authors: Matteo Venanzi and John Guiver
// Refactored entry point for .NET 8.0

namespace BCCWordsRelease
{
    using BCCWordsRelease.Core;
    using BCCWordsRelease.Data;
    using System;
    using System.IO;
    using System.Collections.Generic;

    /// <summary>
    /// Main program entry point for BCCWords sentiment analysis.
    /// </summary>
    public class Program
    {
        /// <summary>
        /// Main method to run the BCCWords crowdsourcing experiment.
        /// </summary>
        /// <param name="args">Command-line arguments.</param>
        public static void Main(string[] args)
        {
            // Parse command-line arguments
            string inputFile = ParseArguments(args);
            
            if (inputFile == null)
            {
                return; // Help was displayed or error occurred
            }

            try
            {
                Console.WriteLine("================================");
                Console.WriteLine("BCCWords: Bayesian Text Sentiment Analysis");
                Console.WriteLine("Using Crowdsourced Annotations");
                Console.WriteLine("================================\n");

                // Load data from TSV file
                Console.WriteLine($"[1/5] Loading data from {Path.GetFileName(inputFile)}...");
                var data = Datum.LoadData(inputFile);
                Console.WriteLine($"      Loaded {data.Count} data points");

                // Validate data quality
                Console.WriteLine($"\n[2/5] Validating data quality...");
                var validationResult = DataValidator.ValidateData(data);
                
                if (!validationResult.IsValid)
                {
                    DataValidator.PrintValidationResults(validationResult);
                    Console.WriteLine("\nPlease fix the data errors and try again.");
                    Environment.Exit(1);
                }
                
                if (validationResult.Warnings.Count > 0)
                {
                    DataValidator.PrintValidationResults(validationResult);
                    Console.Write("Continue anyway? (y/n): ");
                    string response = Console.ReadLine()?.Trim().ToLower();
                    if (response != "y" && response != "yes")
                    {
                        Console.WriteLine("Aborted by user.");
                        Environment.Exit(0);
                    }
                    Console.WriteLine();
                }
                else
                {
                    Console.WriteLine("      ✓ All validation checks passed");
                }

                // Build vocabulary from subset using TF-IDF
                Console.WriteLine($"\n[3/5] Building vocabulary from data subset...");
                var vocabulary = ResultsWords.BuildVocabularyOnSubdata((List<Datum>)data);
                Console.WriteLine($"      Vocabulary size: {vocabulary.Count} terms\n");

                // Create model and results container
                Console.WriteLine("[4/5] Creating BCCWords model and running inference...");
                BCCWords model = new BCCWords();
                ResultsWords resultsWords = new ResultsWords(data, vocabulary);
                DataMappingWords mapping = resultsWords.Mapping as DataMappingWords;

                if (mapping != null)
                {
                    resultsWords = new ResultsWords(data, vocabulary);
                    resultsWords.RunBCCWords(
                        modelName: "BCCwords",
                        data: data,
                        fullData: data,
                        model: model,
                        mode: Results.RunMode.ClearResults,
                        calculateAccuracy: true
                    );
                    Console.WriteLine("      Inference complete\n");

                    // Write results
                    Console.WriteLine("[5/5] Outputting results...\n");
                    Console.WriteLine("================================");
                    Console.WriteLine("RESULTS");
                    Console.WriteLine("================================\n");

                    using (var writer = new StreamWriter(Console.OpenStandardOutput()))
                    {
                        resultsWords.WriteResults(
                            writer: writer,
                            writeCommunityParameters: false,
                            writeWorkerParameters: false,
                            writeWorkerCommunities: false,
                            writeProbWords: true
                        );
                        writer.Flush();
                    }

                    Console.WriteLine("\n================================");
                    Console.WriteLine("Analysis Complete!");
                    Console.WriteLine("================================");
                }
                else
                {
                    Console.WriteLine("ERROR: Failed to create data mapping.");
                    Environment.Exit(1);
                }
            }
            catch (FileNotFoundException ex)
            {
                Console.WriteLine($"\nERROR: File not found - {ex.Message}");
                Console.WriteLine("Please ensure the input file exists and the path is correct.");
                Console.WriteLine("Also ensure stopwords.txt exists in the application directory.");
                Environment.Exit(1);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\nERROR: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
                Environment.Exit(1);
            }
        }

        /// <summary>
        /// Parses command-line arguments and returns the input file path.
        /// </summary>
        /// <param name="args">Command-line arguments.</param>
        /// <returns>The input file path, or null if help was displayed or an error occurred.</returns>
        private static string ParseArguments(string[] args)
        {
            // Check for help flag
            if (args.Length == 0 || args[0] == "-h" || args[0] == "--help" || args[0] == "/?" || args[0] == "-?")
            {
                ShowHelp();
                return null;
            }

            // If one argument provided, treat it as the input file
            if (args.Length == 1)
            {
                string inputFile = args[0];
                
                if (!File.Exists(inputFile))
                {
                    Console.WriteLine($"ERROR: Input file '{inputFile}' not found.\n");
                    ShowHelp();
                    return null;
                }
                
                return inputFile;
            }

            // If more than one argument, check for named arguments
            if (args.Length >= 2)
            {
                for (int i = 0; i < args.Length - 1; i++)
                {
                    if (args[i] == "-i" || args[i] == "--input")
                    {
                        string inputFile = args[i + 1];
                        
                        if (!File.Exists(inputFile))
                        {
                            Console.WriteLine($"ERROR: Input file '{inputFile}' not found.\n");
                            ShowHelp();
                            return null;
                        }
                        
                        return inputFile;
                    }
                }
            }

            // Invalid arguments
            Console.WriteLine("ERROR: Invalid arguments.\n");
            ShowHelp();
            return null;
        }

        /// <summary>
        /// Displays help information about program usage.
        /// </summary>
        private static void ShowHelp()
        {
            Console.WriteLine("BCCWords - Bayesian Classifier Combination with Words");
            Console.WriteLine("======================================================\n");
            Console.WriteLine("DESCRIPTION:");
            Console.WriteLine("  A probabilistic model for sentiment analysis using crowdsourced annotations.");
            Console.WriteLine("  BCCWords combines multiple worker labels to infer true sentiment labels,");
            Console.WriteLine("  while modeling both worker reliability and word-level sentiment indicators.\n");
            Console.WriteLine("USAGE:");
            Console.WriteLine("  BCCWords [options] <input-file>");
            Console.WriteLine("  BCCWords -i <input-file>\n");
            Console.WriteLine("ARGUMENTS:");
            Console.WriteLine("  <input-file>              Path to the input data file (TSV format)");
            Console.WriteLine("  -i, --input <file>        Path to the input data file (alternative syntax)\n");
            Console.WriteLine("OPTIONS:");
            Console.WriteLine("  -h, --help, -?, /?        Display this help message\n");
            Console.WriteLine("INPUT FILE FORMAT:");
            Console.WriteLine("  Tab-separated values (TSV) with the following columns:");
            Console.WriteLine("    1. TaskId       - Unique identifier for each text item");
            Console.WriteLine("    2. WorkerId     - Unique identifier for each annotator");
            Console.WriteLine("    3. WorkerLabel  - Label assigned by the worker (e.g., 0 or 1)");
            Console.WriteLine("    4. GoldLabel    - True label (optional, for evaluation)");
            Console.WriteLine("    5. BodyText     - The text content to analyze\n");
            Console.WriteLine("EXAMPLES:");
            Console.WriteLine("  BCCWords sample_data.txt");
            Console.WriteLine("  BCCWords -i data/tweets.txt");
            Console.WriteLine("  BCCWords --input /path/to/annotations.tsv\n");
            Console.WriteLine("REQUIREMENTS:");
            Console.WriteLine("  - stopwords.txt must exist in the application directory");
            Console.WriteLine("  - Input file must be in valid TSV format\n");
        }
    }
}
