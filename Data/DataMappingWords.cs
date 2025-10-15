/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace BCCWordsRelease.Data
{
    using BCCWordsRelease.TextProcessing;
    using Microsoft.ML.Probabilistic.Utilities;
    using System;
    using System.Collections.Generic;
    using System.Linq;

    /// <summary>
    /// Data mapping class for text-based tasks. Extends DataMapping with vocabulary management.
    /// </summary>
    public class DataMappingWords : DataMapping
    {
        /// <summary>
        /// The vocabulary
        /// </summary>
        public List<string> Vocabulary;

        /// <summary>
        /// The size of the vocabulary.
        /// </summary>
        public int WordCount
        {
            get
            {
                return Vocabulary.Count();
            }
        }

        /// <summary>
        /// The mapping from index to terms.
        /// </summary>
        public Dictionary<int, string> WordIndexToTerm;

        /// <summary>
        /// The word indices for each task.
        /// </summary>
        public int[][] WordIndicesPerTaskIndex;

        /// <summary>
        /// The word counts for each task.
        /// </summary>
        public int[] WordCountsPerTaskIndex;

        public Dictionary<string, string> CFLabelName = new Dictionary<string, string>()
            {
                { "0", "negative" },
                { "1", "positive" },
                { "2", "neutral" },
                { "3", "not related" },
                { "4", "unknown" },
            };

        public Dictionary<string, string> SPLabelName = new Dictionary<string, string>()
            {
                { "0", "ham" },
                { "1", "spam" }
            };

        /// <summary>
        /// Creates a data mapping.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <param name="vocabulary">The vocabulary</param>
        /// <param name="numCommunities">The number of communities.</param>
        /// <param name="labelMin">The lower bound of the labels range.</param>
        /// <param name="labelMax">The upper bound of the labels range.</param>
        public DataMappingWords(IEnumerable<Datum> data, List<string> vocabulary, int numCommunities = -1, int labelMin = int.MaxValue, int labelMax = int.MinValue)
            : base(data, numCommunities, labelMin, labelMax)
        {
            Vocabulary = vocabulary;
            WordIndexToTerm = vocabulary.Select((term, i) => new { Key = i, Value = term }).ToDictionary(v => v.Key, v => v.Value);

            var groupedRandomisedData = data.GroupBy(d => d.TaskId).OrderBy(g => g.Key);
            string[] corpus = Util.ArrayInit(TaskCount, t => (string)null);
            foreach (var kvp in groupedRandomisedData)
            {
                corpus[TaskIdToIndex[kvp.Key]] = kvp.First().BodyText;
            }

            WordIndicesPerTaskIndex = TFIDFProcessor.GetWordIndexStemmedDocs(corpus, Vocabulary);
            WordCountsPerTaskIndex = WordIndicesPerTaskIndex.Select(t => t.Length).ToArray();
        }
    }
}

