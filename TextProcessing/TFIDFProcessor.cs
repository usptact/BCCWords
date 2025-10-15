/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace BCCWordsRelease.TextProcessing
{
    using Iveonik.Stemmers;
    using Microsoft.ML.Probabilistic.Utilities;
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Text.RegularExpressions;

    /// <summary>
    /// Performs TF-IDF (Term Frequency * Inverse Document Frequency) transformation on documents.
    /// Includes tokenization, stemming, stop word removal, and vocabulary building.
    /// </summary>
    public static class TFIDFProcessor
    {
        /// <summary>
        /// Document vocabulary, containing each word's IDF value.
        /// </summary>
        private static Dictionary<string, double> _vocabularyIDF = new Dictionary<string, double>();

        /// <summary>
        /// Transforms a list of documents into their associated TF*IDF values.
        /// If a vocabulary does not yet exist, one will be created, based upon the documents' words.
        /// </summary>
        /// <param name="documents">Array of document strings</param>
        /// <param name="vocabulary">Output vocabulary of terms</param>
        /// <param name="vocabularyThreshold">Minimum number of occurrences of the term within all documents</param>
        /// <returns>TF-IDF vectors for each document</returns>
        public static double[][] Transform(string[] documents, out List<string> vocabulary, int vocabularyThreshold = 3)
        {
            List<List<string>> stemmedDocs;

            // Get the vocabulary and stem the documents at the same time.
            vocabulary = GetVocabulary(documents, out stemmedDocs, vocabularyThreshold);

            if (_vocabularyIDF.Count == 0)
            {
                // Calculate the IDF for each vocabulary term.
                _vocabularyIDF = vocabulary.ToDictionary(term => term, term =>
                {
                    double numberOfDocsContainingTerm = stemmedDocs.Count(d => d.Contains(term));
                    return Math.Log(stemmedDocs.Count / (1 + numberOfDocsContainingTerm));
                });
            }

            // Transform each document into a vector of tfidf values.
            return TransformToTFIDFVectors(stemmedDocs, _vocabularyIDF);
        }

        /// <summary>
        /// Converts a list of stemmed documents and their vocabulary + IDF values into TF*IDF vectors.
        /// </summary>
        private static double[][] TransformToTFIDFVectors(List<List<string>> stemmedDocs, Dictionary<string, double> vocabularyIDF)
        {
            List<List<double>> vectors = new List<List<double>>();
            foreach (var doc in stemmedDocs)
            {
                List<double> vector = new List<double>();

                foreach (var vocab in vocabularyIDF)
                {
                    // Term frequency = count how many times the term appears in this document.
                    double tf = doc.Where(d => d == vocab.Key).Count();
                    double tfidf = tf * vocab.Value;

                    vector.Add(tfidf);
                }

                vectors.Add(vector);
            }

            return vectors.Select(v => v.ToArray()).ToArray();
        }

        /// <summary>
        /// Normalizes a TF*IDF array of vectors using L2-Norm.
        /// Xi = Xi / Sqrt(X0^2 + X1^2 + .. + Xn^2)
        /// </summary>
        public static double[][] Normalize(double[][] vectors)
        {
            List<double[]> normalizedVectors = new List<double[]>();
            foreach (var vector in vectors)
            {
                var normalized = Normalize(vector);
                normalizedVectors.Add(normalized);
            }

            return normalizedVectors.ToArray();
        }

        /// <summary>
        /// Normalizes a TF*IDF vector using L2-Norm.
        /// </summary>
        public static double[] Normalize(double[] vector)
        {
            List<double> result = new List<double>();

            double sumSquared = 0;
            foreach (var value in vector)
            {
                sumSquared += value * value;
            }

            double SqrtSumSquared = Math.Sqrt(sumSquared);

            foreach (var value in vector)
            {
                // L2-norm: Xi = Xi / Sqrt(X0^2 + X1^2 + .. + Xn^2)
                result.Add(value / SqrtSumSquared);
            }

            return result.ToArray();
        }

        /// <summary>
        /// Parses and tokenizes documents, returning a vocabulary of words.
        /// Applies stemming and removes stop words.
        /// </summary>
        private static List<string> GetVocabulary(string[] docs, out List<List<string>> stemmedDocs, int vocabularyThreshold)
        {
            List<string> vocabulary = new List<string>();
            Dictionary<string, int> wordCountList = new Dictionary<string, int>();
            stemmedDocs = new List<List<string>>();
            var stopWordsFile = File.ReadAllLines(@"stopwords.txt");
            var stopWordsList = new List<string>(stopWordsFile).ToArray();
            int docIndex = 0;
            List<string> words = new List<string>();

            foreach (var doc in docs)
            {
                List<string> stemmedDoc = new List<string>();

                docIndex++;

                if (docIndex % 10000 == 0)
                {
                    Console.WriteLine("Processing " + docIndex + "/" + docs.Length);
                }

                string[] parts2 = Tokenize(doc.ToLower());

                foreach (string part in parts2)
                {
                    // Strip non-alphanumeric characters.
                    string stripped = Regex.Replace(part, "[^a-zA-Z0-9]", "");

                    if (!stopWordsList.Contains(stripped.ToLower()))
                    {
                        try
                        {
                            var tmp = new EnglishStemmer();
                            string stem = tmp.Stem(stripped);

                            words.Add(stem);

                            if (stem.Length > 0)
                            {
                                // Build the word count list.
                                if (wordCountList.ContainsKey(stem))
                                {
                                    wordCountList[stem]++;
                                }
                                else
                                {
                                    wordCountList.Add(stem, 0);
                                }

                                stemmedDoc.Add(stem);
                            }
                        }
                        catch
                        {
                            // ignored
                        }
                    }
                }

                stemmedDocs.Add(stemmedDoc);
            }

            // Get the top words.
            var vocabList = wordCountList.Where(w => w.Value >= vocabularyThreshold);
            foreach (var item in vocabList)
            {
                vocabulary.Add(item.Key);
            }

            return vocabulary;
        }

        /// <summary>
        /// Gets word indices for stemmed documents based on a vocabulary.
        /// </summary>
        public static int[][] GetWordIndexStemmedDocs(string[] docs, List<string> vocabulary)
        {
            List<int>[] wordIndex = Util.ArrayInit(docs.Length, d => new List<int>());

            int docIndex = 0;

            foreach (var doc in docs)
            {
                if (doc != null)
                {
                    string[] parts2 = Tokenize(doc.ToLower());

                    List<int> wordIndexDoc = new List<int>();
                    foreach (string part in parts2)
                    {
                        // Strip non-alphanumeric characters.
                        string stripped = Regex.Replace(part, "[^a-zA-Z0-9]", "");

                        try
                        {
                            var tmp = new EnglishStemmer();
                            string stem = tmp.Stem(stripped);

                            if (vocabulary.Contains(stem))
                            {
                                wordIndexDoc.Add(vocabulary.IndexOf(stem));
                            }
                        }
                        catch
                        {
                            // ignored
                        }
                    }

                    wordIndex[docIndex] = (wordIndexDoc.Distinct().ToList());
                    docIndex++;
                }
            }

            return wordIndex.Select(list => list.Select(index => index).ToArray()).ToArray();
        }

        /// <summary>
        /// Tokenizes a string, returning its list of words.
        /// Strips HTML, numbers, URLs, email addresses, etc.
        /// </summary>
        private static string[] Tokenize(string text)
        {
            // Strip all HTML.
            text = Regex.Replace(text, "<[^<>]+>", "");

            // Strip numbers.
            text = Regex.Replace(text, "[0-9]+", "number");

            // Strip urls.
            text = Regex.Replace(text, @"(http|https)://[^\s]*", "httpaddr");

            // Strip email addresses.
            text = Regex.Replace(text, @"[^\s]+@[^\s]+", "emailaddr");

            // Strip dollar sign.
            text = Regex.Replace(text, "[$]+", "dollar");

            // Strip usernames.
            text = Regex.Replace(text, @"@[^\s]+", "username");

            // Tokenize and also get rid of any punctuation
            return text.Split(" @$/#.-:&*+=[]?!(){},''\">_<;%\\".ToCharArray());
        }
    }
}

