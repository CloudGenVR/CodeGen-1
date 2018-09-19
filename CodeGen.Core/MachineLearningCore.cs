using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using CodeGen.ML;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace CodeGen.Core
{
    public class MachineLearningCore
    {
        private readonly ILearningPipelineItem _algorythm;

        public MachineLearningCore(ILearningPipelineItem algorythm)
        {
            _algorythm = algorythm;
        }


        public async Task<PredictionModel> TrainAndSaveModel<TData, TPrediction>(string filePath) where TData : class where TPrediction : class, new()
        {
            //if (File.Exists(fileSavePath))
            //{
            //    File.Delete(fileSavePath);
            //}

            var fields = new List<string> { "Age","Sex", "Cp", "TrestBps", "Chol", "Fbs", "Restecg", "Thalach", "Exang", "OldPeak", "Slope", "Ca", "Thal", "Num" };
       
            var model = await CreateModelWithPipeline<TData, TPrediction>(filePath, true);
            return model;
        }


        public async Task<PredictionModel> CreateModelWithPipeline<TData, TPrediction>(string dataPath, bool useHeader) where TData : class where TPrediction : class, new()
        {

            List<string> properties = typeof(TData).GetFields().Select(o => o.Name).ToList();


            var learningPipeline = new LearningPipeline();
            learningPipeline.Add(new TextLoader(dataPath).CreateFrom<TData>(separator: ','));
          
           
            learningPipeline.Add(new ColumnConcatenator( "Features",  "Age","Sex", "Cp", "TrestBps", "Chol", "Fbs", "Restecg", "Thalach", "Exang", "OldPeak", "Slope", "Ca", "Thal"));
            
            //learningPipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            learningPipeline.Add(_algorythm);


            var model = learningPipeline.Train<TData, TPrediction>();
            return model;
        }

    }
}
