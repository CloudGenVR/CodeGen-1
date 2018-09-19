using Microsoft.ML.Runtime.Api;

namespace CodeGen.MLNET.Prediction
{
    public class HeartPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool PredictedLabel;
    }
}
