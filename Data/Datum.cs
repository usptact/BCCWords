/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace BCCWordsRelease.Data
{
    using System;
    using System.Collections.Generic;
    using System.IO;

    /// <summary>
    /// Represents a single data point in the crowdsourcing dataset.
    /// </summary>
    public class Datum
    {
        /// <summary>
        /// The worker id.
        /// </summary>
        public string WorkerId;

        /// <summary>
        /// The task id.
        /// </summary>
        public string TaskId;

        /// <summary>
        /// The worker's label.
        /// </summary>
        public int WorkerLabel;

        /// <summary>
        /// The task's gold label (optional).
        /// </summary>
        public int? GoldLabel;

        /// <summary>
        /// The body text of the document (optional - only for text sentiment labelling tasks).
        /// </summary>
        public string BodyText;

        /// <summary>
        /// Loads the data file in the format (worker id, task id, worker label, ?gold label).
        /// </summary>
        /// <param name="filename">The data file.</param>
        /// <param name="maxLength"></param>
        /// <returns>The list of parsed data.</returns>
        public static IList<Datum> LoadData(string filename, int maxLength = short.MaxValue)
        {
            var result = new List<Datum>();
            int lineNumber = 0;
            
            using (var reader = new StreamReader(filename))
            {
                string line;
                while ((line = reader.ReadLine()) != null && result.Count < maxLength)
                {
                    lineNumber++;
                    
                    // Skip empty lines
                    if (string.IsNullOrWhiteSpace(line))
                        continue;
                    
                    var strarr = line.Split('\t');
                    int length = strarr.Length;

                    // Validate minimum number of fields
                    if (length < 4)
                    {
                        throw new FormatException(
                            $"Line {lineNumber}: Invalid format. Expected at least 4 tab-separated fields " +
                            $"(WorkerId, TaskId, WorkerLabel, BodyText), but found {length}. " +
                            $"Line content: '{line}'");
                    }

                    try
                    {
                        var datum = new Datum
                        {
                            WorkerId = strarr[0],
                            TaskId = strarr[1],
                            WorkerLabel = int.Parse(strarr[2]),
                            BodyText = strarr[3]
                        };

                        if (length >= 5 && !string.IsNullOrWhiteSpace(strarr[4]))
                            datum.GoldLabel = int.Parse(strarr[4]);
                        else
                            datum.GoldLabel = null;

                        result.Add(datum);
                    }
                    catch (FormatException ex) when (ex.Message.Contains("Input string was not in a correct format"))
                    {
                        throw new FormatException(
                            $"Line {lineNumber}: Failed to parse numeric value. " +
                            $"WorkerLabel and GoldLabel must be integers. " +
                            $"Line content: '{line}'", ex);
                    }
                }
            }

            return result;
        }
    }
}
