/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace BCCWordsRelease.Data
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text.RegularExpressions;

    /// <summary>
    /// Validates data quality and format for BCCWords model.
    /// </summary>
    public class DataValidator
    {
        /// <summary>
        /// Validation result containing errors and warnings.
        /// </summary>
        public class ValidationResult
        {
            public bool IsValid => Errors.Count == 0;
            public List<string> Errors { get; set; } = new List<string>();
            public List<string> Warnings { get; set; } = new List<string>();
            public int ValidRecords { get; set; }
            public int TotalRecords { get; set; }
        }

        /// <summary>
        /// Validates a collection of data records.
        /// </summary>
        /// <param name="data">The data to validate.</param>
        /// <param name="labelValues">Expected label values (e.g., {0, 1} for binary).</param>
        /// <param name="minTextLength">Minimum text length after processing.</param>
        /// <returns>Validation result with errors and warnings.</returns>
        public static ValidationResult ValidateData(
            IList<Datum> data, 
            HashSet<int> labelValues = null, 
            int minTextLength = 1)
        {
            if (labelValues == null)
            {
                // Default to binary classification (0 and 1)
                labelValues = new HashSet<int> { 0, 1 };
            }

            var result = new ValidationResult
            {
                TotalRecords = data.Count
            };

            if (data.Count == 0)
            {
                result.Errors.Add("No data records found in file.");
                return result;
            }

            // Track IDs for validation
            var workerIds = new HashSet<string>();
            var taskIds = new HashSet<string>();
            var recordsPerTask = new Dictionary<string, int>();

            for (int i = 0; i < data.Count; i++)
            {
                var datum = data[i];
                int lineNumber = i + 1;

                // Validate WorkerId
                if (string.IsNullOrWhiteSpace(datum.WorkerId))
                {
                    result.Errors.Add($"Line {lineNumber}: WorkerId is empty or whitespace.");
                }
                else
                {
                    workerIds.Add(datum.WorkerId);
                }

                // Validate TaskId
                if (string.IsNullOrWhiteSpace(datum.TaskId))
                {
                    result.Errors.Add($"Line {lineNumber}: TaskId is empty or whitespace.");
                }
                else
                {
                    taskIds.Add(datum.TaskId);
                    
                    // Track how many annotations per task
                    if (!recordsPerTask.ContainsKey(datum.TaskId))
                        recordsPerTask[datum.TaskId] = 0;
                    recordsPerTask[datum.TaskId]++;
                }

                // Validate WorkerLabel
                if (!labelValues.Contains(datum.WorkerLabel))
                {
                    result.Errors.Add(
                        $"Line {lineNumber}: WorkerLabel '{datum.WorkerLabel}' is not a valid label. " +
                        $"Expected one of: {string.Join(", ", labelValues.OrderBy(x => x))}");
                }

                // Validate GoldLabel (if provided)
                if (datum.GoldLabel.HasValue)
                {
                    if (!labelValues.Contains(datum.GoldLabel.Value))
                    {
                        result.Errors.Add(
                            $"Line {lineNumber}: GoldLabel '{datum.GoldLabel.Value}' is not a valid label. " +
                            $"Expected one of: {string.Join(", ", labelValues.OrderBy(x => x))}");
                    }
                }

                // Validate BodyText
                if (string.IsNullOrWhiteSpace(datum.BodyText))
                {
                    result.Errors.Add($"Line {lineNumber}: BodyText is empty or whitespace.");
                }
                else
                {
                    // Check if text has meaningful content (not just punctuation/numbers)
                    string cleanText = Regex.Replace(datum.BodyText, @"[^a-zA-Z\s]", "").Trim();
                    if (string.IsNullOrWhiteSpace(cleanText))
                    {
                        result.Warnings.Add(
                            $"Line {lineNumber}: BodyText contains no alphabetic characters (only punctuation/numbers). " +
                            $"This may result in empty text after preprocessing.");
                    }
                    else if (cleanText.Length < minTextLength)
                    {
                        result.Warnings.Add(
                            $"Line {lineNumber}: BodyText is very short ({cleanText.Length} characters after cleaning). " +
                            $"This may not provide enough information for classification.");
                    }

                    // Check for extremely long text (might be a data error)
                    if (datum.BodyText.Length > 10000)
                    {
                        result.Warnings.Add(
                            $"Line {lineNumber}: BodyText is very long ({datum.BodyText.Length} characters). " +
                            $"This might be a data formatting error.");
                    }
                }

                // Count valid records (no errors on this line)
                bool hasErrorsOnThisLine = result.Errors.Any(e => e.StartsWith($"Line {lineNumber}:"));
                if (!hasErrorsOnThisLine)
                {
                    result.ValidRecords++;
                }
            }

            // Dataset-level validations
            if (workerIds.Count == 0)
            {
                result.Errors.Add("No valid workers found in dataset.");
            }
            else if (workerIds.Count == 1)
            {
                result.Warnings.Add(
                    "Only one unique worker found. BCCWords is designed for multiple workers. " +
                    "Worker reliability modeling may not be meaningful.");
            }

            if (taskIds.Count == 0)
            {
                result.Errors.Add("No valid tasks found in dataset.");
            }
            else if (taskIds.Count == 1)
            {
                result.Warnings.Add(
                    "Only one unique task found. The model requires multiple tasks for training.");
            }

            // Check for tasks with only one annotation
            int tasksWithOneAnnotation = recordsPerTask.Values.Count(c => c == 1);
            if (tasksWithOneAnnotation > 0)
            {
                result.Warnings.Add(
                    $"{tasksWithOneAnnotation} task(s) have only one annotation. " +
                    $"BCCWords works best with multiple annotations per task to assess worker reliability.");
            }

            // Check for label distribution
            var labelCounts = data.GroupBy(d => d.WorkerLabel)
                                  .ToDictionary(g => g.Key, g => g.Count());
            
            if (labelCounts.Count == 1)
            {
                result.Warnings.Add(
                    $"All annotations have the same label ({labelCounts.First().Key}). " +
                    $"The model may not learn meaningful patterns.");
            }
            else
            {
                // Check for severe class imbalance (>95% one class)
                int maxCount = labelCounts.Values.Max();
                double maxProportion = (double)maxCount / data.Count;
                if (maxProportion > 0.95)
                {
                    result.Warnings.Add(
                        $"Severe class imbalance detected: {maxProportion:P1} of annotations are class " +
                        $"{labelCounts.First(kvp => kvp.Value == maxCount).Key}. " +
                        $"This may affect model performance.");
                }
            }

            return result;
        }

        /// <summary>
        /// Validates text after stop word removal and preprocessing.
        /// </summary>
        /// <param name="processedTexts">Dictionary mapping TaskId to processed text terms.</param>
        /// <returns>Validation result for processed texts.</returns>
        public static ValidationResult ValidateProcessedTexts(Dictionary<string, List<string>> processedTexts)
        {
            var result = new ValidationResult
            {
                TotalRecords = processedTexts.Count
            };

            int emptyTexts = 0;
            int veryShortTexts = 0;

            foreach (var kvp in processedTexts)
            {
                string taskId = kvp.Key;
                var terms = kvp.Value;

                if (terms == null || terms.Count == 0)
                {
                    emptyTexts++;
                    result.Warnings.Add(
                        $"Task {taskId}: Text is empty after preprocessing (stop word removal, stemming). " +
                        $"This task will have no word features.");
                }
                else if (terms.Count == 1)
                {
                    veryShortTexts++;
                    result.Warnings.Add(
                        $"Task {taskId}: Only one term remains after preprocessing. " +
                        $"Consider reviewing stop word list or text quality.");
                }
                else
                {
                    result.ValidRecords++;
                }
            }

            if (emptyTexts > 0)
            {
                result.Warnings.Add(
                    $"WARNING: {emptyTexts} out of {processedTexts.Count} texts are empty after preprocessing. " +
                    $"These texts will have no word features for classification.");
            }

            if (veryShortTexts > 0)
            {
                result.Warnings.Add(
                    $"{veryShortTexts} text(s) have only 1 term after preprocessing. " +
                    $"Limited features may affect classification quality.");
            }

            return result;
        }

        /// <summary>
        /// Prints validation results to console.
        /// </summary>
        public static void PrintValidationResults(ValidationResult result)
        {
            Console.WriteLine("\n--- Data Validation Results ---");
            Console.WriteLine($"Total Records: {result.TotalRecords}");
            Console.WriteLine($"Valid Records: {result.ValidRecords}");
            
            if (result.Errors.Count > 0)
            {
                Console.WriteLine($"\n❌ ERRORS ({result.Errors.Count}):");
                foreach (var error in result.Errors.Take(20)) // Limit to first 20 to avoid spam
                {
                    Console.WriteLine($"  • {error}");
                }
                if (result.Errors.Count > 20)
                {
                    Console.WriteLine($"  ... and {result.Errors.Count - 20} more errors.");
                }
            }
            
            if (result.Warnings.Count > 0)
            {
                Console.WriteLine($"\n⚠️  WARNINGS ({result.Warnings.Count}):");
                foreach (var warning in result.Warnings.Take(10)) // Limit to first 10
                {
                    Console.WriteLine($"  • {warning}");
                }
                if (result.Warnings.Count > 10)
                {
                    Console.WriteLine($"  ... and {result.Warnings.Count - 10} more warnings.");
                }
            }
            
            if (result.IsValid && result.Warnings.Count == 0)
            {
                Console.WriteLine("✓ All validation checks passed!");
            }
            else if (result.IsValid)
            {
                Console.WriteLine("\n✓ Data is valid, but please review warnings above.");
            }
            else
            {
                Console.WriteLine("\n✗ Data validation FAILED. Please fix errors before proceeding.");
            }
            
            Console.WriteLine("-------------------------------\n");
        }
    }
}

