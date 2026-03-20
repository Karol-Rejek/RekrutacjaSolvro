#include "metrics.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <vector>

ClassificationMetrics ClassificationMetrics::compute(const torch::Tensor& preds, const torch::Tensor& targets, const torch::Tensor& probs, int64_t numClasses)
{
    ClassificationMetrics m;
    int64_t N = preds.size(0);

    m.confusionMatrix = torch::zeros({ numClasses, numClasses }, torch::kLong);
    for (int64_t i = 0; i < N; ++i)
    {
        int64_t t = targets[i].item<int64_t>();
        int64_t p = preds[i].item<int64_t>();
        m.confusionMatrix[t][p] += 1;
    }

    m.accuracy = preds.eq(targets).sum().item<double>() / static_cast<double>(N);

    double precSum = 0.0, recSum = 0.0, f1Sum = 0.0;
    for (int64_t c = 0; c < numClasses; ++c)
    {
        double tp = m.confusionMatrix[c][c].item<double>();
        double fp = m.confusionMatrix.select(0, c).sum().item<double>() - tp;
        double fn = m.confusionMatrix.select(1, c).sum().item<double>() - tp;

        double prec = (tp + fp > 0.0) ? tp / (tp + fp) : 0.0;
        double rec = (tp + fn > 0.0) ? tp / (tp + fn) : 0.0;
        double f1 = (prec + rec > 0.0) ? 2.0 * prec * rec / (prec + rec) : 0.0;

        precSum += prec;
        recSum += rec;
        f1Sum += f1;
    }
    m.precision = precSum / static_cast<double>(numClasses);
    m.recall = recSum / static_cast<double>(numClasses);
    m.f1Score = f1Sum / static_cast<double>(numClasses);

    int64_t n = probs.size(0);
    std::vector<std::pair<float, int64_t>> scored(n);
    for (int64_t i = 0; i < n; ++i)
    {
        scored[i] = { probs[i].item<float>(), targets[i].item<int64_t>() };
    }

    // Sortuj malejąco po prawdopodobieństwie
    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

    double pos = static_cast<double>(std::count_if(scored.begin(), scored.end(), [](const auto& s) { return s.second == 1; }));
    double neg = static_cast<double>(n) - pos;

    if (pos == 0.0 || neg == 0.0)
    {
        m.rocAuc = 0.0;
        return m;
    }

    double tpCum = 0.0, fpCum = 0.0, auc = 0.0;
    double prevFpr = 0.0, prevTpr = 0.0;
    for (auto& [score, label] : scored)
    {
        if (label == 1)
        {
            tpCum++;
        }
        else
        {
            fpCum++;
        }

        double tpr = tpCum / pos;
        double fpr = fpCum / neg;

        auc += (fpr - prevFpr) * (tpr + prevTpr) / 2.0;
        prevTpr = tpr;
        prevFpr = fpr;
    }
    m.rocAuc = auc;

    return m;
}

std::string ClassificationMetrics::toString() const
{
    std::ostringstream ss;
    ss << std::fixed;
    ss.precision(4);

    ss << " Accuracy: " << accuracy << "\n"
        << " Precision: " << precision << "  (macro-avg)\n"
        << " Recall: " << recall << "  (macro-avg)\n"
        << " F1 Score: " << f1Score << "  (macro-avg)\n"
        << " ROC-AUC: " << rocAuc << "\n";

    if (confusionMatrix.defined()) {
        ss << "  Confusion Matrix:\n" << "Pred:HandDrawn Pred:Digital\n" << "True:HandDrawn " << confusionMatrix[0][0].item<int64_t>() 
            << " " << confusionMatrix[0][1].item<int64_t>() << "\n"<< "True:Digital "<< confusionMatrix[1][0].item<int64_t>() << " " << confusionMatrix[1][1].item<int64_t>() << "\n";
    }

    return ss.str();
}
