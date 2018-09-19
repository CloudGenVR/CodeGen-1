using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using CodeGen.Core;
using CodeGen.ML;
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


        [HttpPost]
        public async Task Post([FromBody] string dataPath)
        {
            var mlCore = new MachineLearningCore(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 }); 
            var model = await mlCore.TrainAndSaveModel<HeartData, HeartPrediction>(@"../CodeGen.ML/Data/heartdata.csv");
        }
    }
}
