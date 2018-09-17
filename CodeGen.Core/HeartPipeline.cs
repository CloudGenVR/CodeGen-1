using CodeGen.ML.Interface;
using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using TextLoader = Microsoft.ML.Data.TextLoader;

namespace CodeGen.Core
{
    public class HeartPipeline : IHeartPipeline
    {
        public LearningPipeline CreatePipeline<TData>(string dataPath)
        {
            var pipeline = new LearningPipeline {new TextLoader(dataPath).CreateFrom<TData>(separator: ',')};
            return pipeline;
        }

        public IDataView PredictData(PredictionModel model, object sample)
        {       
            var prediction = model.Predict((IDataView) sample);
            return prediction;
        }

        public PredictionModel TrainPipeline<TData, TPrediction>(LearningPipeline learningPipeline) where TData : class where TPrediction : class, new()
        { 
            var model = learningPipeline.Train<TData, TPrediction>();
            return model;
        }
    }
}
