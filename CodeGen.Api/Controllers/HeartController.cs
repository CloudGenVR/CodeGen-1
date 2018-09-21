using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using CodeGen.Core;
using CodeGen.Core.Model;
using CodeGen.MLNET.DataMapping;
using CodeGen.MLNET.Prediction;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML.Trainers;

namespace CodeGen.Api.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class HeartController : ControllerBase
    {
        //// GET api/values
        //[HttpGet]
        //public ActionResult<IEnumerable<string>> Get()
        //{
        //    return new string[] {"value1", "value2"};
        //}

        //// GET api/values/5
        //[HttpGet("{id}")]
        //public ActionResult<string> Get(int id)
        //{
        //    return "value";
        //}

        //// POST api/values
        //[HttpPost]
        //public void Post([FromBody] string value)
        //{
        //}

        //// PUT api/values/5
        //[HttpPut("{id}")]
        //public void Put(int id, [FromBody] string value)
        //{
        //}

        //// DELETE api/values/5
        //[HttpDelete("{id}")]
        //public void Delete(int id)
        //{
        //}

        
        [HttpPost("TrainAndSaveModel", Name = "TrainAndSaveModel")]
        public async Task<IActionResult> TrainAndSaveModel([FromBody] TrainRequest trainModel)
        {

            var dateString = DateTime.Now.ToString("dd_MM_yyyy_HHmm");
            // @"../CodeGen.MLNET/Data/heart-dataset.csv", "../CodeGen.MLNET/Data/heart-model-"+ dateString + ".zip",',');
            var model = await MachineLearningCore.TrainAndSaveModel<HeartData, HeartPrediction>(trainModel);
            return Ok(true);
        }

        [HttpPost("PredictionByModel", Name = "GetByRegistration")]
        public async Task<IActionResult> PredictionByModel([FromBody] HeartData predictionData, string modelFile)
        {
            modelFile = "../CodeGen.MLNET/Data/heart-model-21_09_2018_0003.zip";
            var data = new HeartData
            {
                //Age,Sex,Cp,TrestBps,Chol,Fbs,Restecg,Thalach,Exang,OldPeak,Slope,Ca,Thal,Label
                //44,1,2,120,263,0,0,173,0,0,1,0,7


                //58.0,1.0,3.0,112.0,230.0,0.0,2.0,165.0,0.0,2.5,2.0,1.0,7.0,4

                Age = 58.0f,
                Sex = 1.0f,
                Cp = 3.0f,
                TrestBps = 112.0f,
                Chol = 230.0f,
                Fbs = 0.0f,
                Restecg = 2.0f,
                Thalach = 165.0f,
                Exang = 0.0f,
                OldPeak = 2.5f,
                Slope = 2.0f,
                Ca = 1.0f,
                Thal = 7.0f,
            };
            var model = await MachineLearningCore.PredictionByModel<HeartData, HeartPrediction>(data, modelFile);
            return Ok(model);
        }
    }
}
