# BCCWords: Bayesian Text Sentiment Analysis using Crowdsourced Annotations

## Overview

BCCWords (Bayesian Classifier Combination with Words) is a sophisticated probabilistic model for automated text sentiment analysis that learns from crowdsourced human annotations. This implementation is based on the research presented in:

> **Edwin Simpson, Matteo Venanzi, Steven Reece, Pushmeet Kohli, John Guiver, Stephen Roberts and Nicholas R. Jennings** (2015)  
> "Language Understanding in the Wild: Combining Crowdsourcing and Machine Learning"  
> In: 24th International World Wide Web Conference (WWW 2015)

**Original Microsoft Research Blog Post:** [BCCWords: Bayesian Text Sentiment Analysis](https://learn.microsoft.com/en-us/archive/blogs/infernet_team_blog/bccwords-bayesian-text-sentiment-analysis-using-crowdsourced-annotations)

## The Problem

The challenge addressed by BCCWords is: **How do we build an automated tool for text sentiment analysis that learns from crowdsourced human annotations?**

The problem involves classifying the sentiment of a large corpus (hundreds of thousands) of text snippets, such as tweets, using only a small set of crowdsourced sentiment labels provided by human annotators.

### Real-World Applications

1. **Weather Sentiment Classification from Twitter** - Understanding public sentiment about weather conditions
2. **Disaster Response Applications** - The Ushahidi-Haiti project received 40,000 emergency reports in the first week from victims of the 2010 Haiti earthquake
3. **Social Media Monitoring** - Large-scale sentiment analysis of user-generated content
4. **Market Research** - Understanding customer opinions from crowdsourced reviews

## Key Challenges

BCCWords addresses three fundamental challenges in crowdsourced text classification:

### 1. Annotator Reliability Variability

- Each annotator may have different reliabilities of labeling tweets correctly depending on the content
- Interpreting sentiment or relevance of text is highly subjective
- Variations in annotators' skill levels result in disagreement amongst annotators
- The same text can be interpreted differently by different workers

### 2. Incomplete and Conflicting Labels

- Large volumes of tweets overwhelm small numbers of dedicated expert labelers
- Human labels may not cover the whole set of tweets
- Some tweets may have only one label, multiple conflicting labels, or none at all
- Need to handle sparse labeling scenarios effectively

### 3. Word Distribution Varies by Sentiment

- Each distinct term has different probabilities of appearing in tweets of different sentiment classes
- Example: Words like "Good" and "Nice" are more likely in positive sentiment tweets
- Must leverage language models inferred from aggregated crowdsourced labels
- Need to classify the entire corpus, not just labeled examples

## Model Architecture

### Extending Bayesian Classifier Combination (BCC)

BCCWords extends the core Bayesian Classifier Combination (BCC) model by adding language model learning for automated text classification.

**BCC Foundation:**
- Represents the reliability of each annotator through a confusion matrix
- Expresses labeling probabilities for each possible sentiment class
- Aggregates crowdsourced labels probabilistically

**BCCWords Enhancement:**
- Adds word probability distributions conditioned on sentiment classes
- Learns which words are discriminative for different sentiments
- Combines annotator reliability with language understanding

### Model Components

The model represents the following entities:

- **K** = Number of workers (annotators)
- **N** = Number of tweets (tasks)
- **C** = Number of sentiment classes (e.g., positive, negative, neutral, not related, unknown)
- **D** = Size of dictionary (all unique words in tweets after stemming and stop word removal)

### Infer.NET Implementation

#### Ranges Definition

```csharp
Range n = new Range(N);  // Tweet range
Range k = new Range(K);  // Worker range
Range c = new Range(C);  // Class range
Range d = new Range(D);  // Dictionary range
```

#### Handling Sparsity

To deal with sparsity in worker labels and tweet brevity:

```csharp
VariableArray<int> WorkerTaskCount = Variable.Array<int>(k);
VariableArray<int> WordCount = Variable.Array<int>(n);

Range kn = new Range(WorkerTaskCount[k]);  // Subset of tweets labeled by worker k
Range nw = new Range(WordCount[n]);         // Subset of words in tweet n
```

#### Latent Variables with Dirichlet Priors

**Worker Confusion Matrices:**
```csharp
WorkerConfusionMatrix = Variable.Array(Variable.Array<Vector>(c), k);
WorkerConfusionMatrix[k][c] = Variable<Vector>.Random(ConfusionMatrixPrior[k][c]);
```

**Word Probability Distributions:**
```csharp
var ProbWord = Variable.Array<Vector>(c);
ProbWord[c] = Variable<Vector>.Random(ProbWordPrior).ForEach(c);
```

#### Inference Structure

**Label Inference:**
```csharp
using (Variable.ForEach(k)) {
    var tweetTrueLabel = Variable.Subarray(TweetTrueLabel, WorkerTaskIndex[k]);
    tweetTrueLabel.SetValueRange(c);
    
    using (Variable.ForEach(kn)) {
        using (Variable.Switch(tweetTrueLabel[kn])) {
            WorkerLabel[k][kn] = Variable.Discrete(WorkerConfusionMatrix[k][tweetTrueLabel[kn]]);
        }
    }
}
```

**Words Inference:**
```csharp
using (Variable.ForEach(n)) {
    using (Variable.Switch(tweetTrueLabel[n])) {
        Words[n][nw] = Variable.Discrete(ProbWord[TrueLabel[n]]).ForEach(nw);
    }
}
```

#### Observations

```csharp
int[][] workerLabel;  // Worker labels indexed by worker
int[][] words;        // Word indices indexed by tweet

WorkerLabel.ObservedValue = workerLabel;
Words.ObservedValue = words;
```

#### Querying Posteriors

```csharp
Discrete[] TweetTrueLabel = Engine.Infer<Discrete[]>(TrueLabel);
Dirichlet[][] WorkerConfusionMatrix = Engine.Infer<Dirichlet[][]>(WorkerConfusionMatrix);
Dirichlet[] ProbWordPosterior = Engine.Infer<Dirichlet[]>(ProbWord);
```

## Model Outputs

BCCWords provides three key outputs:

1. **Estimated Confusion Matrix for Each Worker**
   - Shows how reliable each annotator is
   - Reveals systematic biases in labeling patterns
   - Helps identify expert vs. novice annotators

2. **True Label for Each Tweet**
   - Probabilistic classification of each tweet's sentiment
   - Aggregates noisy labels into confident predictions
   - Works even with sparse or conflicting annotations

3. **Word Probabilities for Each Sentiment Class**
   - Distribution of words conditioned on sentiment
   - Identifies discriminative words for each class
   - Enables classification of unlabeled tweets

## Experimental Results

### Dataset: CrowdFlower Weather Sentiment

The model was applied to the CrowdFlower dataset containing:
- Up to 5 weather sentiment annotations per tweet
- Tens of thousands of tweets
- Thousands of workers
- 5 sentiment classes: **neutral, positive, negative, not related, unknown**

### Word Distribution Findings

#### Most Probable Words by Class

The model correctly identified correlations between word usage and sentiment:

- **Positive Class:** Words with positive connotations (e.g., "beautiful", "perfect")
- **Negative Class:** Words with negative connotations (including explicit negative language)
- **Neutral Class:** Mix of descriptive and factual terms
- **Not Related Class:** Random words unrelated to weather sentiment
- **Unknown Class:** Context-dependent words like "complain", "snowstorm", "warm"

#### Most Discriminative Words

By normalizing word probabilities across classes, BCCWords identifies the most discriminative terms:

- **Positive Sentiment:** "beautiful", "perfect" - highly specific to positive tweets
- **Negative Sentiment:** "stayinghometweet", "dammit" - strongly associated with negative weather experiences
- **Context-Dependent:** Words like "complain", "snowstorm", "warm" don't necessarily imply a particular sentiment as interpretation depends heavily on context

### Key Insights

1. **Subjectivity Handling:** The model successfully captures that sentiment interpretation is subjective and varies by annotator
2. **Sparse Data:** Performs well even with incomplete labeling coverage
3. **Language Understanding:** Automatically learns which words are indicative of different sentiments
4. **Scalability:** Can process large corpora by learning from a small set of labeled examples

## Technical Implementation Details

### Text Preprocessing Pipeline

1. **Tokenization:** Split text into individual words
2. **Stop Word Removal:** Filter out common words using `stopwords.txt`
3. **Stemming:** Reduce words to their root form using Porter stemmer
4. **Vocabulary Building:** Create dictionary of unique terms
5. **TF-IDF Selection:** Select high TF-IDF terms for vocabulary (threshold: 0.8)

### Inference Engine Configuration

```csharp
Engine = new InferenceEngine(new VariationalMessagePassing())
{
    ShowFactorGraph = false,
    ShowWarnings = true,
    ShowProgress = false
};
```

Uses **Variational Message Passing** algorithm for efficient approximate inference in large-scale probabilistic models.

### Data Format

**Input TSV Format:**
```
WorkerId    TaskId    WorkerLabel    Text    GoldLabel(optional)
```

- **WorkerId:** Unique identifier for the annotator
- **TaskId:** Unique identifier for the tweet/text
- **WorkerLabel:** Sentiment label assigned by worker
- **Text:** The actual tweet or text content
- **GoldLabel:** Optional ground truth for evaluation

## Model Assumptions

### Probabilistic Assumptions

1. **Worker Labels are Conditionally Independent:**
   - Given the true label, worker labels are drawn from categorical distributions
   - Parameters specified by worker's confusion matrix rows

2. **Words are Conditionally Independent:**
   - Given the tweet's true sentiment, words are drawn from categorical distributions
   - Parameters conditioned on the sentiment class

3. **Conjugate Priors:**
   - Dirichlet priors for confusion matrices
   - Dirichlet priors for word distributions
   - Enables efficient Bayesian inference

## Advantages of BCCWords

1. **Handles Noisy Labels:** Aggregates multiple noisy annotations into reliable predictions
2. **Works with Sparse Data:** Effective even when not all tweets are labeled
3. **Learns from Disagreement:** Uses disagreement patterns to estimate annotator reliability
4. **Transfers Knowledge:** Language model learned from labeled data applies to unlabeled tweets
5. **Uncertainty Quantification:** Provides probability distributions, not just point estimates
6. **Scalable:** Can process large corpora efficiently using Infer.NET

## Comparison with Alternative Approaches

### vs. Majority Vote
- **Majority Vote:** Simple but doesn't account for annotator reliability
- **BCCWords:** Weights annotations by estimated annotator expertise

### vs. Supervised Learning
- **Supervised Learning:** Requires large amounts of gold-standard labels
- **BCCWords:** Works with noisy crowdsourced labels and sparse annotations

### vs. Simple Language Models
- **Simple Models:** Ignore annotation quality and disagreement
- **BCCWords:** Jointly models annotator reliability and language understanding

## Practical Applications

### When to Use BCCWords

✅ Large text corpus with limited expert annotations  
✅ Multiple annotators with varying reliability  
✅ Conflicting or noisy labels  
✅ Need to classify unlabeled data using learned patterns  
✅ Subjective classification tasks (sentiment, relevance, quality)  

### When Not to Use BCCWords

❌ Small datasets (simpler models may suffice)  
❌ All data labeled by single expert (no need for aggregation)  
❌ Non-subjective classification (e.g., spam detection with clear rules)  
❌ Real-time applications (inference can be computationally intensive)  

## Implementation Notes for This Codebase

### Key Classes

- **`WWW15_BCCWordsExperiment`:** Main entry point for running experiments
- **`BCCWords`:** Core model implementation extending BCC
- **`ResultsWords`:** Manages results and posterior distributions
- **`DataMappingWords`:** Maps raw data to model variables
- **`Datum`:** Represents individual data points (tweets with labels)
- **`TFIDFClass`:** Text preprocessing and vocabulary building

### Running the Model

```csharp
var data = Datum.LoadData(@"sample_data.txt");
var vocabulary = ResultsWords.BuildVocabularyOnSubdata(data);

BCCWords model = new BCCWords();
ResultsWords resultsWords = new ResultsWords(data, vocabulary);

resultsWords.RunBCCWords("BCCwords", data, data, model, 
    Results.RunMode.ClearResults, calculateAccuracy: true);

resultsWords.WriteResults(writer, writeProbWords: true);
```

### Configuration Parameters

- **Vocabulary Size:** Currently limited to 6 terms (configurable)
- **TF-IDF Threshold:** 0.8 (higher = more selective vocabulary)
- **Number of Batches:** 3 (for memory management)
- **Inference Algorithm:** Variational Message Passing

## References

1. Simpson, E., Venanzi, M., Reece, S., Kohli, P., Guiver, J., Roberts, S., & Jennings, N. R. (2015). Language Understanding in the Wild: Combining Crowdsourcing and Machine Learning. WWW 2015.

2. Microsoft Research Blog: [BCCWords: Bayesian Text Sentiment Analysis using Crowdsourced Annotations](https://learn.microsoft.com/en-us/archive/blogs/infernet_team_blog/bccwords-bayesian-text-sentiment-analysis-using-crowdsourced-annotations)

3. CrowdFlower Weather Sentiment Dataset: [Crowdsourcing at Scale Challenge](http://www.crowdflower.com/blog/2013/12/crowdsourcing-at-scale-shared-task-challenge-winners)

4. Ushahidi-Haiti Project: [How Social Media Can Inform UN Assessments](https://www.linkedin.com/pulse/how-social-media-can-inform-un-assessments-during-major-patrick-meier)

## Further Reading

- **Infer.NET Documentation:** [dotnet.github.io/infer](https://dotnet.github.io/infer)
- **Bayesian Classifier Combination (BCC):** The foundation model that BCCWords extends
- **Variational Message Passing:** The inference algorithm used in this implementation
- **Crowdsourcing for Machine Learning:** Techniques for aggregating noisy labels

---

## Technical Support

For questions about this implementation:
- Review the comprehensive README.md in this repository
- Examine the inline code documentation in BCCWords.cs
- Refer to the Infer.NET user guide for probabilistic programming concepts

**Version:** Updated for .NET 8.0 with Microsoft.ML.Probabilistic 0.4.2402.2904  
**Authors:** Matteo Venanzi and John Guiver (Original), Modernized for .NET 8.0  
**Last Updated:** 2025
