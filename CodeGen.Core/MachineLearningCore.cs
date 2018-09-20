using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Models;

namespace CodeGen.Core
{
    public class MachineLearningCore
    {
        private readonly ILearningPipelineItem _algorythm;

      
        public MachineLearningCore(ILearningPipelineItem algorythm)
        {
            _algorythm = algorythm;
        }


        public static async Task<TPrediction> PredictionByModel<TData, TPrediction>(TData data, string modelPath) where TData : class where TPrediction : class, new()
        {
            var model = await PredictionModel.ReadAsync<TData, TPrediction>(modelPath);

            return model.Predict(data);
        }

        public static async Task<bool> TrainAndSaveModel<TData, TPrediction>(string filePath, string modelToSave) where TData : class where TPrediction : class, new()
        {
            if (File.Exists(modelToSave))
            {
                File.Delete(modelToSave);
            }
      
            var model = await CreateModelWithPipeline<TData, TPrediction>(filePath, true, ',', "PredictedLabel");

            await model.WriteAsync(modelToSave);

            return true;
        }


        private static async Task<PredictionModel> CreateModelWithPipeline<TData, TPrediction>(string dataPath, bool useHeader, char separator, string predictedLabel) where TData : class where TPrediction : class, new()
        {

            List<string> properties = typeof(TData).GetFields().Select(o => o.Name).ToList();
            try
            {
                var learningPipeline = new LearningPipeline
                {
                    new TextLoader(dataPath).CreateFrom<TData>(useHeader: true, separator: ','),
                    new ColumnConcatenator("Features", "Age", "Sex", "Cp", "TrestBps", "Chol", "Fbs", "Restecg", "Thalach", "Exang", "OldPeak", "Slope", "Ca", "Thal"),
                    new NaiveBayesClassifier
                    {
                        NormalizeFeatures = NormalizeOption.Auto,
                        Caching = CachingOptions.Memory
                    },
                };

                var model = learningPipeline.Train<TData, TPrediction>();

                return model;
            }
            catch (Exception ex)
            {
                return null;
            }

        }

    }
}
