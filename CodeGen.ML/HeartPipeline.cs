using System;
using System.Collections.Generic;
using System.Text;
using CodeGen.ML.Interface;
using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using TextLoader = Microsoft.ML.Data.TextLoader;

namespace CodeGen.ML
{
    public class HeartPipeline : IHeartPipeline
    {
        public LearningPipeline CreatePipeline<TData>(string dataPath)
        {
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader(dataPath).CreateFrom<TData>(separator: ','));
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
