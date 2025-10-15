/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace BCCWordsRelease.Utilities
{
    using System;
    using System.Collections.Generic;
    using System.Collections.ObjectModel;

    /// <summary>
    /// Receiver Operating Characteristic (ROC) curve for binary classification evaluation.
    /// </summary>
    public class ReceiverOperatingCharacteristic
    {
        private double area;
        private double[] measurement;
        private double[] prediction;
        private int positiveCount;
        private int negativeCount;
        private double dtrue;
        private double dfalse;

        public PointCollection collection;

        /// <summary>
        /// Constructs a new Receiver Operating Characteristic model.
        /// </summary>
        /// <param name="measurement">Binary values (e.g., 0 and 1) indicating negative and positive cases.</param>
        /// <param name="prediction">Continuous values approximating the measurement array.</param>
        public ReceiverOperatingCharacteristic(double[] measurement, double[] prediction)
        {
            this.measurement = measurement;
            this.prediction = prediction;

            // Determine which numbers correspond to each binary category
            dtrue = dfalse = measurement[0];
            for (int i = 1; i < measurement.Length; i++)
            {
                if (dtrue < measurement[i])
                    dtrue = measurement[i];
                if (dfalse > measurement[i])
                    dfalse = measurement[i];
            }

            // Count the real number of positive and negative cases
            for (int i = 0; i < measurement.Length; i++)
            {
                if (measurement[i] == dtrue)
                    this.positiveCount++;
            }

            this.negativeCount = this.measurement.Length - this.positiveCount;
        }

        /// <summary>
        /// Computes a ROC curve with 1/increment points.
        /// </summary>
        public void Compute(double increment)
        {
            List<Point> points = new List<Point>();
            double cutoff;

            for (cutoff = dfalse; cutoff <= dtrue; cutoff += increment)
            {
                points.Add(ComputePoint(cutoff));
            }
            if (cutoff < dtrue) points.Add(ComputePoint(dtrue));

            points.Sort((a, b) => a.Specificity.CompareTo(b.Specificity));
            this.collection = new PointCollection(points.ToArray());
            this.area = calculateAreaUnderCurve();
            calculateStandardError();
        }

        Point ComputePoint(double threshold)
        {
            int truePositives = 0;
            int trueNegatives = 0;

            for (int i = 0; i < this.measurement.Length; i++)
            {
                bool measured = (this.measurement[i] == dtrue);
                bool predicted = (this.prediction[i] >= threshold);

                if (predicted == measured)
                {
                    if (predicted)
                        truePositives++;
                    else trueNegatives++;
                }
            }

            int falsePositives = negativeCount - trueNegatives;
            int falseNegatives = positiveCount - truePositives;

            return new Point(this, threshold,
                truePositives, trueNegatives,
                falsePositives, falseNegatives);
        }

        private double calculateAreaUnderCurve()
        {
            double sum = 0.0;

            for (int i = 0; i < collection.Count - 1; i++)
            {
                var tpz = collection[i].Sensitivity + collection[i + 1].Sensitivity;
                tpz = tpz * (collection[i].FalsePositiveRate - collection[i + 1].FalsePositiveRate) / 2.0;
                sum += tpz;
            }
            return sum;
        }

        private double calculateStandardError()
        {
            double A = area;
            int Na = positiveCount;
            int Nn = negativeCount;

            double Q1 = A / (2.0 - A);
            double Q2 = 2 * A * A / (1.0 + A);

            return Math.Sqrt((A * (1.0 - A) +
                (Na - 1.0) * (Q1 - A * A) +
                (Nn - 1.0) * (Q2 - A * A)) / (Na * Nn));
        }

        /// <summary>
        /// ROC Curve Point.
        /// </summary>
        public class Point : ConfusionMatrix
        {
            private double cutoff;

            internal Point(ReceiverOperatingCharacteristic curve, double cutoff,
                int truePositives, int trueNegatives, int falsePositives, int falseNegatives)
                : base(truePositives, trueNegatives, falsePositives, falseNegatives)
            {
                this.cutoff = cutoff;
            }

            public double Cutoff
            {
                get { return cutoff; }
            }
        }

        /// <summary>
        /// Collection of ROC Curve points.
        /// </summary>
        public class PointCollection : ReadOnlyCollection<Point>
        {
            internal PointCollection(Point[] points)
                : base(points)
            {
            }
        }
    }
}

