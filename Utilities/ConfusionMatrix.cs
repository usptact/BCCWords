/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace BCCWordsRelease.Utilities
{
    /// <summary>
    /// Represents a 2x2 confusion matrix for binary classification evaluation.
    /// </summary>
    public class ConfusionMatrix
    {
        // 2x2 confusion matrix components
        private int truePositives;
        private int trueNegatives;
        private int falsePositives;
        private int falseNegatives;

        /// <summary>
        /// Constructs a new Confusion Matrix.
        /// </summary>
        /// <param name="truePositives">Number of true positives</param>
        /// <param name="trueNegatives">Number of true negatives</param>
        /// <param name="falsePositives">Number of false positives</param>
        /// <param name="falseNegatives">Number of false negatives</param>
        public ConfusionMatrix(int truePositives, int trueNegatives,
            int falsePositives, int falseNegatives)
        {
            this.truePositives = truePositives;
            this.trueNegatives = trueNegatives;
            this.falsePositives = falsePositives;
            this.falseNegatives = falseNegatives;
        }

        /// <summary>
        /// Sensitivity, also known as True Positive Rate (Recall).
        /// </summary>
        /// <remarks>
        /// Sensitivity = TPR = TP / (TP + FN)
        /// </remarks>
        public double Sensitivity => (double)truePositives / (truePositives + falseNegatives);

        /// <summary>
        /// Specificity, also known as True Negative Rate.
        /// </summary>
        /// <remarks>
        /// Specificity = TNR = TN / (FP + TN)
        /// or also as: TNR = (1 - False Positive Rate)
        /// </remarks>
        public double Specificity => (double)trueNegatives / (trueNegatives + falsePositives);

        /// <summary>
        /// False Positive Rate, also known as false alarm rate.
        /// </summary>
        /// <remarks>
        /// FPR = FP / (FP + TN)
        /// or also as: FPR = (1 - specificity)
        /// </remarks>
        public double FalsePositiveRate => (double)falsePositives / (falsePositives + trueNegatives);
    }
}

