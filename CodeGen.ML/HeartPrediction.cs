using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Runtime.Api;

namespace CodeGen.ML
{
    public class HeartPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels;
    }
}
