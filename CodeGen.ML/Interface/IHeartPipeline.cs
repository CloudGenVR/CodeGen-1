using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Runtime.Data;

namespace CodeGen.ML.Interface
{
    public interface IHeartPipeline
    {
        LearningPipeline CreatePipeline<TData>(string dataPath);

        PredictionModel TrainPipeline<TData, TPrediction>(LearningPipeline learningPipeline) where TData : class where TPrediction : class, new();

        IDataView PredictData(PredictionModel model, object sample);
    }
}
