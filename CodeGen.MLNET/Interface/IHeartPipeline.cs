using Microsoft.ML;
using Microsoft.ML.Runtime.Data;

namespace CodeGen.MLNET.Interface
{
    public interface IHeartPipeline
    {
        LearningPipeline CreatePipeline<TData>(string dataPath);

        PredictionModel TrainPipeline<TData, TPrediction>(LearningPipeline learningPipeline) where TData : class where TPrediction : class, new();

        IDataView PredictData(PredictionModel model, object sample);
    }
}
