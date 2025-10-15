# BCCWords: Bayesian Text Sentiment Analysis using Crowdsourced Annotations

A .NET 8.0 implementation of the BCCWords model for sentiment analysis that learns from crowdsourced annotations with varying annotator reliability.

## Quick Start

### Prerequisites

- .NET 8.0 SDK
- Microsoft.ML.Probabilistic (Infer.NET) package

### Installation & Running

```bash
# Clone and navigate to the repository
git clone <repository-url>
cd BCCWords

# Build and run
dotnet build
dotnet run
```

The application will process `sample_data.txt` and output sentiment analysis results with word probabilities per class.

## Data Format

Input data must be in TSV (Tab-Separated Values) format:

```
WorkerId    TaskId    WorkerLabel    Text    GoldLabel(optional)
```

**Fields:**
- **WorkerId**: Unique annotator identifier
- **TaskId**: Unique text snippet identifier  
- **WorkerLabel**: Sentiment label assigned by the worker (0 or 1)
- **Text**: Text content to analyze
- **GoldLabel**: Optional ground truth for evaluation

## Model Output

The BCCWords model provides:

- **True Label Predictions**: Estimated sentiment class for each text
- **Worker Reliability**: Confusion matrices showing annotator accuracy patterns
- **Word Probabilities**: Most discriminative words for each sentiment class (top 30 by default)
- **Model Evidence**: Overall model quality metric

## Overview

BCCWords (Bayesian Classifier Combination with Words) addresses the challenge of building reliable sentiment classifiers from crowdsourced data where annotators have varying reliability levels. It combines:

1. **Worker Reliability Model**: Confusion matrices representing each annotator's labeling patterns
2. **Language Model**: Word probability distributions conditioned on sentiment classes

### Key Features

- **Annotator Reliability Modeling**: Accounts for varying skill levels and interpretation differences
- **Sparse Labeling Support**: Handles incomplete, conflicting, or missing labels
- **Language Model Integration**: Uses TF-IDF, stemming, and stop word removal
- **Bayesian Inference**: Provides uncertainty quantification via variational message passing

## Use Cases

- Social media sentiment analysis (Twitter, reviews)
- Disaster response and emergency report processing
- Market research and opinion mining
- Content moderation

## Technical Architecture

### Model Components

- **Worker Confusion Matrices**: Model P(worker_label | true_class) for each annotator
- **Word Probability Distributions**: Learn P(word | true_class) using Dirichlet priors
- **True Label Inference**: Estimate true sentiment using variational message passing

### Text Processing Pipeline

1. **Tokenization**: Split text into words
2. **Stemming**: Porter stemming to root forms
3. **Stop Word Removal**: Filter common non-informative words
4. **TF-IDF Selection**: Select most discriminative terms

## Project Structure

```
BCCWords/
├── Program.cs                      # Application entry point
├── BCCWords.cs                     # Core Infer.NET models
├── Data/                           # Data models and mapping
│   ├── Datum.cs
│   ├── DataMapping.cs
│   └── DataMappingWords.cs
├── Core/                           # Results management
│   └── Results.cs
├── TextProcessing/                 # NLP utilities
│   └── TFIDFProcessor.cs
└── Utilities/                      # Performance metrics
    ├── ConfusionMatrix.cs
    └── ROCCurve.cs
```

## Performance Considerations

- **Vocabulary Size**: Limited to 6 terms in sample for efficiency (configurable)
- **Inference**: Uses variational message passing for scalability
- **Convergence**: 35 iterations by default

## Research Background

Based on research presented in:

> Edwin Simpson, Matteo Venanzi, Steven Reece, Pushmeet Kohli, John Guiver, Stephen Roberts and Nicholas R. Jennings, (2015) "Language Understanding in the Wild: Combining Crowdsourcing and Machine Learning." WWW 2015.

For detailed information, see:
- [Microsoft Research Blog Post](https://learn.microsoft.com/en-us/archive/blogs/infernet_team_blog/bccwords-bayesian-text-sentiment-analysis-using-crowdsourced-annotations)
- [BCCWords-Model-Documentation.md](BCCWords-Model-Documentation.md) in this repository

## Technical Details

### Infer.NET Integration

Uses Microsoft.ML.Probabilistic for probabilistic programming:

- **Variational Message Passing**: Efficient approximate inference
- **Dirichlet Priors**: Probability distribution modeling
- **Discrete Variables**: Categorical sentiment classification

### Model Extensions

The BCCWords model extends the base BCC (Bayesian Classifier Combination) model by adding:
- Word-level features via TF-IDF
- Per-class word probability distributions
- Joint inference over labels and language model

## Contributing

This project has been modernized to .NET 8.0 with the latest Infer.NET framework. Contributions welcome for:

- Performance optimizations
- Multi-language support
- Additional text preprocessing features
- Integration with ML.NET

## License

Based on Microsoft Research work. See original copyright and licensing terms.

## Acknowledgments

- Microsoft Research for BCCWords model and Infer.NET framework
- .NET Foundation for open-source Infer.NET
- Research community for advancing crowdsourced ML techniques
