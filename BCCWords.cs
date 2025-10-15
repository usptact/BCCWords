/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

// Authors: Matteo Venanzi and John Guiver

/* Language Understanding in the Wild: Combining Crowdsourcing and Machine Learning
* 
* Software to run the experiment presented in the paper "Language Understanding in the Wind: Combining Crowdsourcing and Machine Learning" by Simpsons et. al, WWW15
To run it:
- Replace <your-data-file> with a TSV with fields <WorkerId, TaskId, Worker label, Text, Gold label (optional)
- Replace <your-stop-words-file> with a TSV with the list of stop words, one for each line
*/

namespace BCCWordsRelease
{
    using Iveonik.Stemmers;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Models;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;
    using Microsoft.ML.Probabilistic.Algorithms;
    using BCCWordsRelease.TextProcessing;
    using BCCWordsRelease.Utilities;
    using BCCWordsRelease.Core;
    using BCCWordsRelease.Data;
    using System;
    using System.Collections.Generic;
    using System.Collections.ObjectModel;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    using System.Text.RegularExpressions;
    using Range = Microsoft.ML.Probabilistic.Models.Range;


    /// <summary>
    /// Results class containing posteriors and predictions of BCCWords.
    /// </summary>
    public class ResultsWords : Results
    {
        /// <summary>
        /// The posterior of the word probabilities for each true label.
        /// </summary>
        public Dirichlet[] ProbWords
        {
            get;
            private set;
        }

        /// <summary>
        /// The vocabulary
        /// </summary>
        public List<string> Vocabulary
        {
            get;
            set;
        }

        /// <summary>
        /// Creates an object for storing the inference results of BCCWords
        /// </summary>
        /// <param name="data">The data</param>
        /// <param name="vocabulary">The vocabulary</param>
        public ResultsWords(IList<Datum> data, List<string> vocabulary)
        {
            if (vocabulary == null)
            {
                // Build vocabulary
                Console.Write("Building vocabulary...");
                Stopwatch stopwatch = new Stopwatch();
                stopwatch.Start();
                string[] corpus = data.Select(d => d.BodyText).Distinct().ToArray();
                Vocabulary = BuildVocabularyFromCorpus(corpus);
                Console.WriteLine("done. Elapsed time: {0}", stopwatch.Elapsed);
            }

            // Build data mapping
            Vocabulary = vocabulary;
            this.Mapping = new DataMappingWords(data, vocabulary);
            this.GoldLabels = Mapping.GetGoldLabelsPerTaskId();
        }

        /// <summary>
        /// Runs the majority vote method on the data.
        /// </summary>
        /// <param name="modelName"></param>
        /// <param name="data">The data</param>
        /// <param name="mode"></param>
        /// <param name="calculateAccuracy">Compute the accuracy (true).</param>
        /// <param name="fullData"></param>
        /// <param name="model"></param>
        /// <param name="useMajorityVote"></param>
        /// <param name="useRandomLabel"></param>
        /// <returns>The updated results</returns>
        public void RunBCCWords(string modelName,
            IList<Datum> data,
            IList<Datum> fullData,
            BCCWords model,
            RunMode mode,
            bool calculateAccuracy,
            bool useMajorityVote = false,
            bool useRandomLabel = false)
        {
            DataMappingWords MappingWords = null;
            if (FullMapping == null)
                FullMapping = new DataMapping(fullData);

            if (Mapping == null)
            {
                // Build vocabulary
                Console.Write("Building vocabulary...");
                Stopwatch stopwatch = new Stopwatch();
                stopwatch.Start();
                string[] corpus = data.Select(d => d.BodyText).Distinct().ToArray();
                Vocabulary = BuildVocabularyFromCorpus(corpus);
                Console.WriteLine("done. Elapsed time: {0}", stopwatch.Elapsed);

                // Build data mapping
                this.Mapping = new DataMappingWords(data, MappingWords.Vocabulary);
                MappingWords = Mapping as DataMappingWords;
                this.GoldLabels = MappingWords.GetGoldLabelsPerTaskId();
            }

            MappingWords = Mapping as DataMappingWords;
            int[] trueLabels = null;
            if (useMajorityVote)
            {
                if (MappingWords != null)
                {
                    var majorityLabel = MappingWords.GetMajorityVotesPerTaskId(data);
                    trueLabels = Util.ArrayInit(FullMapping.TaskCount, i => majorityLabel.ContainsKey(Mapping.TaskIndexToId[i]) ? (int)majorityLabel[Mapping.TaskIndexToId[i]] : Rand.Int(Mapping.LabelMin, Mapping.LabelMax + 1));
                    data = MappingWords.BuildDataFromAssignedLabels(majorityLabel, data);
                }
            }

            if (useRandomLabel)
            {
                var randomLabels = MappingWords.GetRandomLabelPerTaskId(data);
                data = MappingWords.BuildDataFromAssignedLabels(randomLabels, data);
            }

            var labelsPerWorkerIndex = MappingWords.GetLabelsPerWorkerIndex(data);
            var taskIndicesPerWorkerIndex = MappingWords.GetTaskIndicesPerWorkerIndex(data);

            // Create model
            ClearResults();
            model.CreateModel(MappingWords.TaskCount, MappingWords.LabelCount, MappingWords.WordCount);

            // Run model inference
            BCCWordsPosteriors posteriors = model.InferPosteriors(labelsPerWorkerIndex, taskIndicesPerWorkerIndex, MappingWords.WordIndicesPerTaskIndex, MappingWords.WordCountsPerTaskIndex, trueLabels);

            // Update results
            UpdateResults(posteriors, mode);

            // Compute accuracy
            if (calculateAccuracy)
            {
                UpdateAccuracy();
            }
        }

        /// <summary>
        /// Select high TFIDF terms
        /// </summary>
        /// <param name="corpus">array of terms</param>
        /// <param name="tfidf_threshold">TFIDF threshold</param>
        /// <returns></returns>
        private static List<string> BuildVocabularyFromCorpus(string[] corpus, double tfidf_threshold = 0.8)
        {
            List<string> vocabulary;
            double[][] inputs = TFIDFProcessor.Transform(corpus, out vocabulary, 0);
            inputs = TFIDFProcessor.Normalize(inputs);

            // Select high TF_IDF terms
            List<string> vocabularyTfidf = new List<string>();
            for (int index = 0; index < inputs.Length; index++)
            {
                var sortedTerms = inputs[index].Select((x, i) => new KeyValuePair<string, double>(vocabulary[i], x)).OrderByDescending(x => x.Value).ToList();
                vocabularyTfidf.AddRange(sortedTerms.Where(entry => entry.Value > tfidf_threshold).Select(k => k.Key).ToList());
            }
            return vocabulary.Distinct().ToList();
        }

        protected override void ClearResults()
        {
            BackgroundLabelProb = Dirichlet.Uniform(Mapping.LabelCount);
            WorkerConfusionMatrix = new Dictionary<string, Dirichlet[]>();
            WorkerPrediction = new Dictionary<string, Dictionary<String, Discrete>>();
            WorkerCommunity = new Dictionary<string, Discrete>();
            TrueLabel = new Dictionary<string, Discrete>();
            PredictedLabel = new Dictionary<string, int?>();
            TrueLabelConstraint = new Dictionary<string, Discrete>();
            CommunityConfusionMatrix = null;
            WorkerScoreMatrixConstraint = new Dictionary<string, VectorGaussian[]>();
            CommunityProb = null;
            CommunityScoreMatrix = null;
            CommunityConstraint = new Dictionary<string, Discrete>();
            LookAheadTrueLabel = new Dictionary<string, Discrete>();
            LookAheadWorkerConfusionMatrix = new Dictionary<string, Dirichlet[]>();
            ModelEvidence = new Bernoulli(0.5);
            ProbWords = null;
        }

        /// <summary>
        /// Writes various results to a StreamWriter.
        /// </summary>
        /// <param name="writer">A StreamWriter instance.</param>
        /// <param name="writeCommunityParameters">Set true to write community parameters.</param>
        /// <param name="writeWorkerParameters">Set true to write worker parameters.</param>
        /// <param name="writeWorkerCommunities">Set true to write worker communities.</param>
        /// <param name="writeProbWords">Set true to write word probabilities</param>
        /// <param name="topWords">Number of words to select</param>
        public void WriteResults(StreamWriter writer, bool writeCommunityParameters, bool writeWorkerParameters, bool writeWorkerCommunities, bool writeProbWords, int topWords = 30)
        {
            base.WriteResults(writer, writeCommunityParameters, writeWorkerCommunities, writeWorkerCommunities);
            DataMappingWords MappingWords = Mapping as DataMappingWords;
            if (writeProbWords && this.ProbWords != null)
            {
                int NumClasses = ProbWords.Length;
                for (int c = 0; c < NumClasses; c++)
                {
                    if (MappingWords != null && MappingWords.WorkerCount > 300) // Assume it's CF
                        writer.WriteLine("Class {0}", MappingWords.CFLabelName[c.ToString()]);
                    else
                        if (MappingWords != null)
                        writer.WriteLine("Class {0}", MappingWords.SPLabelName[c.ToString()]);

                    Vector probs = ProbWords[c].GetMean();
                    var probsDictionary = probs.Select((value, index) => new KeyValuePair<string, double>(MappingWords.Vocabulary[index], Math.Log(value))).OrderByDescending(x => x.Value).ToArray();

                    for (int w = 0; w < topWords; w++)
                    {
                        writer.WriteLine($"\t{probsDictionary[w].Key}: \t{probsDictionary[w].Value:0.000}");
                    }
                }
            }
        }

        /// <summary>
        /// Build a vocabulary of terms for a subset of text snippets extracted from the data
        /// </summary>
        /// <param name="data">the data</param>
        /// <returns></returns>
        public static List<string> BuildVocabularyOnSubdata(List<Datum> data)
        {
            Console.WriteLine("Building vocabulary");
            var subData = data.Where((k, i) => i < 20000).ToList();
            string[] corpus = subData.Select(d => d.BodyText).Distinct().ToArray();
            var vocabularyOnSubData = BuildVocabularyFromCorpus(corpus);
            
            // Return vocabulary, limited to 6 terms or less if vocabulary is smaller
            int vocabCount = Math.Min(vocabularyOnSubData.Count, 6);
            return vocabularyOnSubData.GetRange(0, vocabCount);
        }
    }


    /// <summary>
    /// The BCCWords model
    /// </summary>
    public class BCCWords : BCC
    {
        // Add extra ranges
        private Range w;
        private Range nw;

        // Model evidence
        private Variable<bool> evidence;

        // Additional variables for BCCWords
        private VariableArray<Vector> ProbWord;
        private VariableArray<int> WordCount;
        private VariableArray<VariableArray<int>, int[][]> Words;
        private Variable<Dirichlet> ProbWordPrior;

        public void CreateModel(int NumTasks, int NumClasses, int VocabSize, int numBatches = 3)
        {
            WorkerCount = Variable.New<int>().Named("WorkerCount");

            // Set up inference engine
            Engine = new InferenceEngine(new VariationalMessagePassing())
            {
                ShowFactorGraph = false,
                ShowWarnings = true,
                ShowProgress = false
            };

            // Set engine flags
            Engine.Compiler.WriteSourceFiles = true;
            Engine.Compiler.UseParallelForLoops = true;

            evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);

            // Set up ranges
            n = new Range(NumTasks).Named("N");
            c = new Range(NumClasses).Named("C");
            k = new Range(WorkerCount).Named("K");
            WorkerTaskCount = Variable.Array<int>(k).Named("WorkerTaskCount");
            kn = new Range(WorkerTaskCount[k]).Named("KN");
            WorkerTaskIndex = Variable.Array(Variable.Array<int>(kn), k).Named("Task");
            WorkerTaskIndex.SetValueRange(n);

            // Initialise truth
            BackgroundLabelProbPrior = Variable.New<Dirichlet>().Named("TruthProbPrior");
            BackgroundLabelProb = Variable<Vector>.Random(BackgroundLabelProbPrior).Named("TruthProb");
            BackgroundLabelProb.SetValueRange(c);

            // Truth distributions
            TrueLabel = Variable.Array<int>(n).Named("Truth");
            TrueLabel[n] = Variable.Discrete(BackgroundLabelProb).ForEach(n);

            //VocabSize = Variable.New<int>();
            w = new Range(VocabSize).Named("W");
            ProbWord = Variable.Array<Vector>(c).Named("ProbWord");
            ProbWord.SetValueRange(w);
            WordCount = Variable.Array<int>(n).Named("WordCount");
            nw = new Range(WordCount[n]).Named("WN");
            Words = Variable.Array(Variable.Array<int>(nw), n).Named("Word");
            ProbWordPrior = Variable.New<Dirichlet>().Named("ProbWordPrior");
            ProbWord[c] = Variable<Vector>.Random(ProbWordPrior).ForEach(c);

            // Initialise user profiles
            ConfusionMatrixPrior = Variable.Array(Variable.Array<Dirichlet>(c), k).Named("WorkerConfusionMatrixPrior");
            WorkerConfusionMatrix = Variable.Array(Variable.Array<Vector>(c), k).Named("WorkerConfusionMatrix");
            WorkerConfusionMatrix[k][c] = Variable<Vector>.Random(ConfusionMatrixPrior[k][c]);
            WorkerConfusionMatrix.SetValueRange(c);

            // Vote distributions
            WorkerLabel = Variable.Array(Variable.Array<int>(kn), k).Named("WorkerLabel");

            using (Variable.ForEach(k))
            {
                var trueLabel = Variable.Subarray(TrueLabel, WorkerTaskIndex[k]).Named("TrueLabelSubarray");
                trueLabel.SetValueRange(c);
                using (Variable.ForEach(kn))
                {
                    using (Variable.Switch(trueLabel[kn]))
                    {
                        WorkerLabel[k][kn] = Variable.Discrete(WorkerConfusionMatrix[k][trueLabel[kn]]);
                    }
                }
            }

            // Words inference
            using (Variable.ForEach(n))
            {
                using (Variable.Switch(TrueLabel[n]))
                {
                    Words[n][nw] = Variable.Discrete(ProbWord[TrueLabel[n]]).ForEach(nw);
                }
            }
            block.CloseBlock();
        }

        private void ObserveCrowdLabels(int[][] workerLabel, int[][] workerTaskIndex)
        {
            BackgroundLabelProbPrior.ObservedValue = Dirichlet.Uniform(c.SizeAsInt);
            WorkerCount.ObservedValue = workerLabel.Length;
            WorkerLabel.ObservedValue = workerLabel;
            WorkerTaskCount.ObservedValue = workerTaskIndex.Select(tasks => tasks.Length).ToArray();
            WorkerTaskIndex.ObservedValue = workerTaskIndex;
            SetBiasedPriors(WorkerCount.ObservedValue);
        }

        private void ObserveWords(int[][] words, int[] wordCounts)
        {
            Words.ObservedValue = words;
            WordCount.ObservedValue = wordCounts;
        }

        private void ObserveTrueLabels(int[] trueLabels)
        {
            TrueLabel.ObservedValue = trueLabels;
        }

        public void SetBiasedPriors(int workerCount)
        {
            // uniform over true values
            BackgroundLabelProbPrior.ObservedValue = Dirichlet.Uniform(c.SizeAsInt);
            ConfusionMatrixPrior.ObservedValue = Util.ArrayInit(workerCount, input => Util.ArrayInit(c.SizeAsInt, l => new Dirichlet(Util.ArrayInit(c.SizeAsInt, l1 => l1 == l ? 5.5 : 1))));
            ProbWordPrior.ObservedValue = Dirichlet.Symmetric(w.SizeAsInt, 1);
        }

        /* Inference */
public BCCWordsPosteriors InferPosteriors(
            int[][] workerLabel, int[][] workerTaskIndex, int[][] words, int[] wordCounts, int[] trueLabels = null,
            int numIterations = 35)
        {
            ObserveCrowdLabels(workerLabel, workerTaskIndex);

            ObserveWords(words, wordCounts);

            if (trueLabels != null)
            {
                ObserveTrueLabels(trueLabels);
            }

            BCCWordsPosteriors posteriors = new BCCWordsPosteriors();

            Console.WriteLine("\n***** BCC Words *****\n");
            for (int it = 1; it <= numIterations; it++)
            {
                Engine.NumberOfIterations = it;
                posteriors.TrueLabel = Engine.Infer<Discrete[]>(TrueLabel);
                posteriors.WorkerConfusionMatrix = Engine.Infer<Dirichlet[][]>(WorkerConfusionMatrix);
                posteriors.BackgroundLabelProb = Engine.Infer<Dirichlet>(BackgroundLabelProb);
                posteriors.ProbWordPosterior = Engine.Infer<Dirichlet[]>(ProbWord);
                Console.WriteLine("Iteration {0}:\t{1:0.0000}", it, posteriors.TrueLabel[0]);
            }

            posteriors.Evidence = Engine.Infer<Bernoulli>(evidence);
            return posteriors;
        }
    }

    /// <summary>
    /// BCCWords posterior object.
    /// </summary>
    [Serializable]
    public class BCCWordsPosteriors : BCCPosteriors
    {
        /// <summary>
        /// The Dirichlet posteriors of the word probabilities for each true label value.
        /// </summary>
        public Dirichlet[] ProbWordPosterior;

    }

    /// <summary>
    /// The BCC posteriors class.
    /// </summary>
    [Serializable]
    public class BCCPosteriors
    {
        /// <summary>
        /// The probabilities that generate the true labels of all the tasks.
        /// </summary>
        public Dirichlet BackgroundLabelProb;

        /// <summary>
        /// The probabilities of the true label of each task.
        /// </summary>
        public Discrete[] TrueLabel;

        /// <summary>
        /// The Dirichlet parameters of the confusion matrix of each worker.
        /// </summary>
        public Dirichlet[][] WorkerConfusionMatrix;

        /// <summary>
        /// The predictive probabilities of the worker's labels.
        /// </summary>
        public Discrete[][] WorkerPrediction;

        /// <summary>
        /// The true label constraint used in online training.
        /// </summary>
        public Discrete[] TrueLabelConstraint;

        /// <summary>
        /// The model evidence.
        /// </summary>
        public Bernoulli Evidence;
    }

    /// <summary>
    /// The BCC model class.
    /// </summary>
    public class BCC
    {
        /// <summary>
        /// The number of label values.
        /// </summary>
        public int LabelCount => c?.SizeAsInt ?? 0;

        /// <summary>
        /// The number of tasks.
        /// </summary>
        public int TaskCount => n?.SizeAsInt ?? 0;

        // Ranges
        protected Range n;
        protected Range k;
        protected Range c;
        protected Range kn;

        // Variables in the model
        protected Variable<int> WorkerCount;
        protected VariableArray<int> TrueLabel;
        protected VariableArray<int> WorkerTaskCount;
        protected VariableArray<VariableArray<int>, int[][]> WorkerTaskIndex;
        protected VariableArray<VariableArray<int>, int[][]> WorkerLabel;
        protected Variable<Vector> BackgroundLabelProb;
        protected VariableArray<VariableArray<Vector>, Vector[][]> WorkerConfusionMatrix;
        protected Variable<bool> Evidence;

        // Prior distributions
        protected Variable<Dirichlet> BackgroundLabelProbPrior;
        protected VariableArray<VariableArray<Dirichlet>, Dirichlet[][]> ConfusionMatrixPrior;
        protected VariableArray<Discrete> TrueLabelConstraint;
        protected Variable<Bernoulli> EvidencePrior;

        // Inference engine
        protected InferenceEngine Engine;

        // Hyperparameters and inference settings
        public double InitialWorkerBelief
        {
            get;
            set;
        }

        /// <summary>
        /// Returns the confusion matrix prior of each worker.
        /// </summary>
        /// <returns>The confusion matrix prior of each worker.</returns>
        public Dirichlet[] GetConfusionMatrixPrior()
        {
            var confusionMatrixPrior = new Dirichlet[LabelCount];
            for (int d = 0; d < LabelCount; d++)
            {
                confusionMatrixPrior[d] = new Dirichlet(Util.ArrayInit(LabelCount, i => i == d ? (InitialWorkerBelief / (1 - InitialWorkerBelief)) * (LabelCount - 1) : 1.0));
            }

            return confusionMatrixPrior;
        }
    }
}